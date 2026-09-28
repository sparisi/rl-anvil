"""Rho-ablation heatmaps from saved replay memories: true visit counts against
pseudocounts, every row recomputed on one memory.

Written for one figure: the rho ablation of configs/plots/gcrl/ablation_rho.yaml.
It reads the `configs:` schema plot_results.py takes, one row per entry, and the
entry whose hyperparameter is null is the reference: its own visit counts are
that row, and each other entry is a `pseudocount(radius=<its value>)` over the
SAME memory, labelled by its `plot.label`. So the figure compares the values of
that hyperparameter on one set of samples rather than on whatever data each run
happened to collect. Nothing about it is specific to `rho` beyond the name: the
hyperparameter is whichever key the entries name besides `algorithm`, and the
reference and the ablations may be the same algorithm, since what tells them
apart is the null rather than the config file.

--plot_config names a config, a bare name looked for in configs/plots, or a
directory of them; every *.yaml in it is drawn, except those whose first
non-empty line is `# skip`.

`rng_seed` in a plot config is not read: --rng_seed picks which seed's memory is
drawn, for every config alike.

--colorbar draws the colour scale under the panels it applies to. How many panels
a bar covers follows --shared_vmap; unset, it is one bar per panel. Panels are
drawn larger so the bar and its ticks stay legible.

A run's memory is <data_dir>/<cfg_dir>/<rng_seed>/memory.npz, with the seed
coming from --rng_seed (default 0).

Pseudocounts come from `pseudocount` in src/pseudocount.py.

A run is the environment and the algorithm whose YAMLs it matches on every key
they declare, apart from the plot config's `ignored_cfg_keys`. A run matching
neither is "unknown" and is not drawn.

Only `visit_count` is drawn.

Output goes to <data_dir>/<output>/heatmaps/<config_stem>/<key>/.

Example

    python plot_heatmaps_rho.py -f data_gcrl -p gcrl/ablation_rho --shared_vmap=same_figure --log_scale -v
"""

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

from src.utils.plot import (
    assign_groups, ensure_font, env_groups, flatten_cfg, is_null,
    load_config_group, ignored_cfg_keys, load_plot_configs, memory_fractions,
    plot_entries, plot_labels, split_config_label, tex_kwargs as _tex, value_str,
)
from src.utils.heatmaps import (
    CBAR_SUBPLOT_SIZE, add_border, add_colorbar, arr_range as _arr_range, auto_dpi,
    cbar_groups, cbar_hspace, compute_maps_all_fractions, env_context, orient,
    pooled_range, scale_funcs, scale_key, viridis_black_bad,
)
from src.pseudocount import pseudocount_prefixes, robust_scale

ensure_font()
font_size = 7
plt.rcParams["font.size"] = font_size
# Panel titles and axis labels, as a multiple of font_size, as in
# plot_heatmaps.py: they name a panel a few centimetres across, so they are set
# well above the tick and colour-bar text.
LABEL_SCALE = 2.2

SUBPLOT_SIZE = 1.5   # inches per panel, raised to CBAR_SUBPLOT_SIZE under --colorbar

MEMORY_KEYS = ["visit_count"]

# Rows sharing a colour range. Every row is a count -- a visit count or a
# pseudocount -- so there is one group; the machinery around it takes a group
# per range a figure may hold.
SCALE_GROUPS = ("count",)

# Only these fields of a memory are actually consumed; skip the rest so
# we don't decompress them on load.
NEEDED_MEM_KEYS = {"obs"}


_CMAP = viridis_black_bad()


def plot_cell(ax, arr, vmin, vmax):
    """Draw one map in one panel, compressed by --exp_scale or --log_scale.

    `arr` of None leaves an empty bordered panel, which is how a row with no
    memory behind it shows up."""

    if arr is None:
        ax.set_xticks([])
        ax.set_yticks([])
        add_border(ax)
        return
    arr, vmin, vmax = _FORWARD(arr), float(_FORWARD(vmin)), float(_FORWARD(vmax))
    ax.imshow(
        arr,
        vmin=vmin,
        vmax=vmax,
        cmap=_CMAP,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("black")
    add_border(ax)


def make_grid(nrows, ncols, subplot_size=None, hspace=0.05):
    """A Figure of `nrows` x `ncols` touching panels, as `axs[row][col]`.

    An `hspace` above the default opens a gap between the rows for a colour bar
    that ends above the bottom row, and the figure grows by exactly that gap: the
    panels keep their size rather than shrinking to make room.

    `subplot_size` is read from the module at call time, since --colorbar raises
    it and the flag is parsed after this is defined."""

    subplot_size = SUBPLOT_SIZE if subplot_size is None else subplot_size
    extra = max(hspace - 0.05, 0.0) * (nrows - 1)
    fig = plt.figure(figsize=(subplot_size * ncols, subplot_size * (nrows + extra)))
    gs = fig.add_gridspec(
        nrows,
        ncols,
        hspace=hspace,
        wspace=0.05,
        left=0,
        right=1,
        top=1,
        bottom=0,
    )
    return fig, [
        [fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)
    ]


def grid_for(cell_keys, nrows, ncols):
    """`make_grid`, with a row gap opened when the colour bars need one."""

    return make_grid(
        nrows,
        ncols,
        hspace=cbar_hspace(cell_keys, nrows) if args.colorbar else 0.05,
    )


def add_colorbars(fig, axs, cell_keys, cell_ranges):
    """One bar per set of panels drawn on a shared range (see cbar_groups).

    `cell_ranges` holds the bounds as the maps hold them; plot_cell compresses
    them on its way to the colours, so a bar is normalised on the compressed
    bounds -- its gradient is then the one on the panels -- and its ticks are
    labelled with the values they came from."""

    if not args.colorbar:
        return
    for (vmin, vmax), cells in cbar_groups(cell_keys, cell_ranges):
        add_colorbar(
            fig,
            [axs[r][c] for r, c in cells],
            _FORWARD(vmin),
            _FORWARD(vmax),
            cmap=_CMAP,
            inverse=_INVERSE,
            fontsize=font_size,
        )


def is_skipped_config(config_path) -> bool:
    """Whether a plot config is marked to be left undrawn, that is, whether its
    first non-empty line starts with `# skip` or `#skip`."""

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip().lower()
                if not stripped:
                    continue
                return stripped.startswith("# skip") or stripped.startswith("#skip")
    except OSError:
        return False
    return False


def _canon(v):
    """A hyperparameter value as the key its rows are bucketed by: None for a
    null, and what `value_str` makes of it otherwise.

    The null keeps a key of its own because the reference row is the one whose
    ablated key is null, and it is looked up by that. Every other value goes
    through the one equality the rest of the comparison uses, so a value buckets
    here exactly as it matches there."""

    if is_null(v):
        return None
    return value_str(v)


def ablation_from_entries(plot_cfg, where=""):
    """Read a plot config and return what it draws, as
    (hyperparameter key, rows, reference algorithms, ablation algorithms).

        configs:
          <group>:
            - algorithm: <stem>
              <hyperparameter>: null        # the reference: its own visit counts
              plot.label: "True Counts"
            - algorithm: <stem>
              <hyperparameter>: <value>     # pseudocount(radius=value) on that memory
              plot.label: "..."

    An entry whose hyperparameter is null is a reference row, drawn from that
    run's memory; every other entry is a pseudocount recomputed on the same
    memory, so the figure compares the values rather than the data each run
    happened to collect. That is what the script exists to show, and the reason
    the reference and the ablations may name the same algorithm: what tells them
    apart is the value, not the config file.

    Labels come from `plot.label`, one per row. `rng_seed`, `statistics` and
    `color_scheme` are not read: the first is a command-line argument, and the
    other two describe figures this script does not draw."""

    if not (plot_cfg.get("configs") or {}):
        raise SystemExit(f"{where}: no `configs:` to draw.")

    entries = plot_entries(plot_cfg)
    keys = {k for _, _, selector in entries for k in selector if k != "algorithm"}
    if len(keys) != 1:
        raise SystemExit(
            f"{where}: an ablation needs exactly one hyperparameter besides "
            f"`algorithm` across its entries, and this one names {sorted(keys)}."
        )
    hp_key = keys.pop()

    rows, ref_algos, abl_algos = [], set(), set()
    for _, label, selector in entries:
        algo = selector.get("algorithm")
        if algo is None:
            raise SystemExit(
                f"{where}: entry '{label}' names no algorithm, so there is no "
                f"run to read a memory from."
            )
        algo = str(algo)
        raw = selector.get(hp_key)
        if is_null(raw):
            ref_algos.add(algo)
            rows.append({
                "algo_id": algo,
                "v_key": None,
                "label": label,
                "kind": "true",
            })
        else:
            abl_algos.add(algo)
            rows.append({
                "algo_id": algo,
                "v_key": _canon(raw),
                "raw": raw,
                "label": label,
                "kind": "pseudo",
            })

    if not ref_algos:
        raise SystemExit(
            f"{where}: no entry sets {hp_key} to null, so there is no reference "
            f"memory to draw the visit counts from and to recompute the "
            f"pseudocounts on."
        )
    return hp_key, rows, ref_algos, abl_algos


def _build_ref_data(run_cfg, mem):
    """Extract the fields needed to recompute pseudocounts on this run's obs:
    (bin_centers, obs_goal, scale, n_bins, cartesian, reachable). Returns None
    for an environment whose counter is not binned and so has no bin centres to
    score pseudocounts at."""

    if "obs" not in mem:
        return None
    goal_idx_raw = ((run_cfg.get("agent") or {}).get("critic") or {}).get("goal_idx")
    goal_idx = slice(None) if goal_idx_raw is None else goal_idx_raw
    counter, cartesian, reachable = env_context(run_cfg["environment"])
    if counter is None or not hasattr(counter, "n_bins"):
        return None
    obs = np.asarray(mem["obs"]).astype(np.float64)
    obs_goal = obs[..., goal_idx]
    return {
        "obs_goal":    obs_goal,
        "scale":       robust_scale(obs_goal),
        "bin_centers": counter.bin_centers_raw().astype(np.float64),
        "n_bins":      tuple(counter.n_bins),
        "cartesian":   cartesian,
        "reachable":   reachable,
    }


def _radius(v_key):
    """The radius a canonical hyperparameter value stands for, or None when it is
    not numeric."""

    try:
        return float(v_key)
    except (ValueError, TypeError):
        return None


def _pseudocounts(d, radii, fractions):
    """`pseudocount(radius)` at every bin centre over the first `fraction` of the
    memory in `d` (see _build_ref_data), in the scale of the whole memory.
    `radii` is [(v_key, radius)]. Returns {(v_key, fraction): 2D map}.

    Bin centres are passed in chunks: the pairwise distances are a
    (bin centres, memory) matrix.

    One pass per chunk, not one per (radius, fraction): the distances depend on
    neither, so `pseudocount_prefixes` computes them once and thresholds and
    sums them for every pair. The fractions are prefixes of one batch rather
    than batches of their own, for the same reason."""

    x = d["bin_centers"]
    N = len(d["obs_goal"])
    chunk_m = 512

    prefixes = [max(1, int(round(N * frac))) for frac in fractions]
    chunks = [
        pseudocount_prefixes(
            x[i:i + chunk_m],
            d["obs_goal"],
            [radius for _, radius in radii],
            prefixes,
            scale=d["scale"],
        )
        for i in range(0, len(x), chunk_m)
    ]
    counts = np.concatenate(chunks, axis=-1)

    out = {}
    for i, (vk, _) in enumerate(radii):
        for j, frac in enumerate(fractions):
            out[(vk, frac)] = orient(
                counts[i, j],
                d["n_bins"],
                d["cartesian"],
                d["reachable"],
            )
    return out


# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", default="data_dir")
parser.add_argument(
    "-o", "--output",
    default="plots",
    help="Output root, under the data directory. Figures go in "
         "<root>/heatmaps/<config_stem>.",
)
parser.add_argument("-a", "--algorithms", default="configs/algorithm")
parser.add_argument("-e", "--environments", default="configs/environment")
parser.add_argument(
    "-p", "--plot_config",
    required=True,
    metavar="NAME",
    help="Plot config: which configurations become the rows, over which "
         "environments, and what to call them. A name without .yaml is looked for "
         "in configs/plots, or give a path; a directory is a set of configs and "
         "every one of them is drawn, bar those marked `# skip`.",
)
parser.add_argument(
    "--rng_seed",
    type=int,
    default=0,
    metavar="N",
    help="Which seed of each configuration to draw: "
         "<cfg_dir>/<rng_seed>/memory.npz. Default: 0.",
)
parser.add_argument("--max_dpi", type=int, default=1000)
parser.add_argument("--pixels_per_cell", type=int, default=15)
parser.add_argument(
    "--log_scale",
    action="store_true",
    help="Apply np.log1p to the maps and their bounds before plotting.",
)
parser.add_argument(
    "--exp_scale",
    type=float,
    default=None,
    help="Softer alternative to --log_scale: apply (1+x)**alpha to the maps and "
         "their bounds. alpha=1 is linear, alpha=0.5 gives sqrt-like compression, "
         "and alpha approaching 0 approaches a log scale. Overrides --log_scale "
         "when set.",
)
parser.add_argument(
    "--shared_vmap",
    choices=["all_figures", "same_figure"],
    default=None,
    help="Share vmin/vmax across heatmaps for the same (env, key). "
         "'all_figures': pool over every row in the data. 'same_figure': pool over "
         "the maps in the current figure only. Unset: each heatmap uses its own "
         "min/max.",
)
parser.add_argument(
    "--colorbar",
    action="store_true",
    help="Draw the colour scale under the panels it applies to. --shared_vmap "
         "decides how many panels a bar covers, and unset it is one per panel. "
         "Panels are drawn larger.",
)
parser.add_argument(
    "--rotate",
    action="store_true",
    help="Put the rows (the reference and the ablated values) in the columns "
         "instead.",
)
parser.add_argument(
    "--progression_step",
    type=float,
    default=1.0,
    help="Step (in (0, 1]) between memory-fraction checkpoints. 0.25 emits plots "
         "at 25/50/75/100%% of the memory, suffixed _25 _50 _75 _100. Default "
         "1.0 = a single plot from the full memory, suffixed _100.",
)
parser.add_argument(
    "-v", "--verbose",
    action="store_true",
    help="Print progress messages. The recap is always printed.",
)
args = parser.parse_args()

# A colour bar and its tick labels need more room than a panel of the default
# size leaves under it. auto_dpi divides by the same number, so a cell keeps its
# pixel budget and only the figure grows.
if args.colorbar:
    SUBPLOT_SIZE = CBAR_SUBPLOT_SIZE

# The transform the panels and their colour bars share, built once. A bar is
# normalised on the compressed bounds, so building the pair a second time where
# the bars are drawn is a second chance to disagree with what the panels used.
# It is also quiet on a bound the transform has no answer for, and returns NaN,
# which is what add_colorbar draws nothing on.
_FORWARD, _INVERSE = scale_funcs(
    log_scale=args.log_scale, exp_scale=args.exp_scale
)


def vprint(*a, **kw):
    """Print only under --verbose."""

    if args.verbose:
        print(*a, **kw)


root        = Path(args.folder)
output_root = root / args.output / "heatmaps"

try:
    FRACTIONS = memory_fractions(args.progression_step)
except ValueError as err:
    parser.error(f"--progression_step {err}")

# --- Preflight: gather the set of reference and ablation algos across ALL
# ablation configs. Only these runs need their memory loaded (avoids
# decompressing ~10s of MB per run for algos no config draws).
plot_cfgs = [
    (name, Path(path), cfg)
    for name, path, cfg in load_plot_configs(args.plot_config)
    if not is_skipped_config(path)
]
if not plot_cfgs:
    raise SystemExit(
        f"No plot config to draw for '{args.plot_config}' "
        f"(every one of them is marked `# skip`)."
    )
vprint(f"\n{len(plot_cfgs)} plot config(s): {[n for n, _, _ in plot_cfgs]}")

# What each plot config asks of the scan: the keys it exempts, the algorithms
# whose runs are its reference, and the hyperparameter it ablates. Kept per
# config rather than merged into one table -- which algorithm a run is depends on
# the keys that config exempts, so a merged answer is no config's.
#
# Only the reference runs' memory is read: the other rows are pseudocounts over
# that same memory, so decompressing one per value would be reading megabytes
# nothing is drawn from. The reference and the ablations may be the same
# algorithm -- what tells them apart is the null value of the ablated key.
_cfg_specs: list = []
# The union over the plot configs, for deciding which runs are scanned at all.
# Exempting more keys can only loosen the comparison, so a run no config can
# match under the union matches under none of them on its own. Which
# configuration a run then IS is not read off this -- see _ref_names below.
_ignored_keys: frozenset = frozenset()
for _name, _cp, _pc in plot_cfgs:
    _ignored = ignored_cfg_keys(_pc)
    _ignored_keys |= _ignored
    _hp_key, _, _refs, _ = ablation_from_entries(_pc, where=str(_cp))
    _cfg_specs.append((_ignored, set(_refs), _hp_key))
if _ignored_keys:
    vprint(f"Keys exempt while scanning: {sorted(_ignored_keys)}")


# --- Read every run's configuration ---
cfg_dirs = sorted(root.glob("*"))
flat_cfgs:   dict = {}   # cfg_dir_name -> flat cfg
run_cfgs:    dict = {}   # cfg_dir_name -> nested cfg
no_cfg_dirs: list = []

for cfg_dir in cfg_dirs:
    if not cfg_dir.is_dir():
        continue
    cfg_file = cfg_dir / "cfg.yaml"
    if not cfg_file.exists():
        no_cfg_dirs.append(cfg_dir.name)
        continue
    run_cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    run_cfgs[cfg_dir.name]  = run_cfg
    # Interpolations kept, as `composed_config` keeps them: a key dropped from
    # one side of the comparison and not the other is a key it reads as missing
    # from the run, and a dropped value reads as null -- which is what tells a
    # reference run from an ablation one below.
    flat_cfgs[cfg_dir.name] = flatten_cfg(run_cfg, drop_interpolations=False)

if not flat_cfgs:
    raise SystemExit(f"No <cfg_dir>/cfg.yaml under {root}.")

df = pd.DataFrame.from_dict(flat_cfgs, orient="index")
df.columns = pd.MultiIndex.from_tuples([("hyperparameter", c) for c in df.columns])


# --- Identify environment and algorithm ---
# Every key a YAML declares must equal what the run recorded, the swept one
# included: two algorithms differing only in it would otherwise be one, and a
# run of either could be assigned to whichever declares more keys. A run
# matching no YAML is "unknown" and is not scanned.
#
# `ignored_cfg_keys` from the plot configs is exempt: a key a launch set for
# every run alike says nothing about which configuration a run is, and the YAML
# describes one none of them were run at.
env_configs_all  = load_config_group(args.environments)
algo_configs_all = load_config_group(args.algorithms)

# The answers `assign_groups` has already given, by the exemptions that produced
# them: configs declaring the same ones get the same answer, and the frame is
# compared against every YAML to arrive at it.
_assigned: dict = {}

# `labels_of`'s own answers, so the per-name dicts are built once per set of
# exemptions as well.
_labelled: dict = {}


def labels_of(ignored):
    """Return (environment, algorithm, overrides) of every run under `ignored`, as
    three dicts keyed by cfg dir name.

    The scan and the deduplication below index runs by name rather than by row, so
    the shared `assign_groups` answer is read out as dicts here. What a run
    overrode is read off its algorithm label, so it moves with the exemptions too:
    a key exempt from the comparison is not something the run overrode."""

    if ignored not in _labelled:
        # Both groups are required: a row is named by an algorithm and a
        # hyperparameter value, so a run this script cannot name in both is a run
        # it cannot place. With no YAMLs for a group, every run is "unknown" there
        # and the scan reports it unmatched.
        env, algo, env_stem, _matched = assign_groups(
            df,
            env_configs_all,
            algo_configs_all,
            ignored,
            env_id_fallback=False,
            algo_optional=False,
            cache=_assigned,
        )
        _labelled[ignored] = (
            env_stem.to_dict(),
            algo.map(lambda v: split_config_label(str(v))[0]).to_dict(),
            {
                name: split_config_label(str(a))[1]
                for name, a in zip(df.index, algo)
            },
        )
    return _labelled[ignored]


# The union of every plot config's exemptions, for deciding which runs are scanned
# at all (see _ignored_keys above).
#
# `overrides_of` is empty: `assign_configs` labels a run with a config name alone,
# so there is no parenthesised tail for `split_config_label` to return. The
# off-ablation check that reads it therefore passes everything, and a run
# overriding a key the ablation does not vary shares a row with the clean one.
env_of, algo_of, overrides_of = labels_of(_ignored_keys)


# Whose memory the scan reads: the runs some plot config would draw as its
# reference. Asked once per config under that config's own exemptions, because
# which algorithm a run is moves with them -- a run one config calls a reference
# has to be read whatever another config calls it. The predicate is the one
# `is_reference` applies below, so the scan reads exactly what the drawing loop
# goes on to ask for.
_ref_names: set = set()
for _ignored, _refs, _hp_key in _cfg_specs:
    _, _cfg_algo, _ = labels_of(_ignored)
    _ref_names.update(
        name for name, flat in flat_cfgs.items()
        if _cfg_algo[name] in _refs and is_null(flat.get(_hp_key))
    )


# --- Load memory of reference runs, compute maps ---
# One entry per cfg dir (not per (env, algo) — an ablation sweep has multiple
# runs per (env, algo), one per hp value). Deduplication into (env, algo, hp)
# happens per plot config, since hp_key varies by config.
scanned: dict = {}   # cfg_dir_name -> {"env", "algo", "flat", "maps", "ref_data"}
no_memory_dirs:  list = []
no_counter_dirs: list = []
unmatched:       list = []

with tqdm(
    list(flat_cfgs), desc="Scanning", unit="cfg", disable=not args.verbose
) as pbar:
    for name in pbar:
        env_name  = env_of[name]
        algo_name = algo_of[name]
        if env_name == "unknown" or algo_name == "unknown":
            unmatched.append(name)
            continue

        pbar.set_postfix_str(f"{env_name}/{algo_name}")

        run_cfg  = run_cfgs[name]
        is_ref = name in _ref_names
        mem_file = root / name / str(args.rng_seed) / "memory.npz"
        maps_per_frac = None
        ref_data = None

        if is_ref:
            mem = None
            if not mem_file.exists():
                no_memory_dirs.append(name)
            else:
                try:
                    with np.load(mem_file) as data:
                        mem = {k: data[k] for k in data.files if k in NEEDED_MEM_KEYS}
                except Exception as e:
                    vprint(f"  WARNING: skipping corrupt {mem_file}: {e}")

            if mem is not None:
                try:
                    maps_per_frac = compute_maps_all_fractions(run_cfg, mem, FRACTIONS)
                except Exception as e:
                    vprint(
                        f"  WARNING: compute_maps failed for {name} "
                        f"({env_name}/{algo_name}): {e}"
                    )
                else:
                    if maps_per_frac is None:
                        no_counter_dirs.append(name)
                    else:
                        ref_data = _build_ref_data(run_cfg, mem)

        scanned[name] = {
            "env":       env_name,
            "algo":      algo_name,
            "overrides": overrides_of[name],
            "flat":      flat_cfgs[name],
            "maps":     maps_per_frac,
            "ref_data": ref_data,
        }

# --- Recap ---
# Read off the union labelling by name, not off df[ENV]/df[ALGO]: `assign_groups`
# writes those on every call, so by now they hold whichever exemption set was
# labelled last. The recap is about the runs the scan kept, which is the union's
# question.
_union_env, _union_algo = _assigned[_ignored_keys]
inexact = sorted({
    str(c) for c in [*_union_env, *_union_algo] if split_config_label(str(c))[1]
})

print(f"Scanned  cfg dirs : {len(cfg_dirs)}")
print(
    f"Loaded    runs    : {len(scanned)}"
    f"  ({len(no_cfg_dirs)} no cfg,"
    f" {len(no_memory_dirs)} no memory,"
    f" {len(unmatched)} unmatched,"
    f" {len(no_counter_dirs)} no tabular counter)"
)
if inexact:
    print("\nConfig matches carrying overrides:")
    for label in inexact:
        print(f"  {label}")


# --- Iterate over ablation plot configs ---
# Per-config `env_vmins`/`env_vmaxs` are rebuilt inside the loop so the count
# pool includes both visit-count (reference) and pseudocount (ablation) values.
for config_name, config_path, plot_cfg in plot_cfgs:
    hp_key, rows, ref_algos, abl_algos = ablation_from_entries(
        plot_cfg,
        where=str(config_path),
    )

    groups = env_groups(plot_cfg)
    _, flat_envs, _ = plot_labels(plot_cfg)

    active_algos = ref_algos | abl_algos

    def is_reference(entry):
        """Whether a scanned run is the one every row of this config is drawn
        from: its visit counts are the reference row, and the pseudocounts are
        recomputed on its memory."""

        return (
            entry["algo"] in ref_algos
            and is_null(entry["flat"].get(hp_key))
        )

    # This config's own exemptions, not every config's: exempting a key changes
    # which YAML a run matches, and one config's `ignored_cfg_keys` says nothing
    # about the runs another one draws. A run matching no YAML under them is not
    # drawn here, whatever it matched under another config's.
    cfg_ignored = ignored_cfg_keys(plot_cfg)
    cfg_env, cfg_algo, cfg_overrides = labels_of(cfg_ignored)
    config_runs: dict = {}
    unknown_here, without_maps = 0, 0
    for name, entry in scanned.items():
        if cfg_env[name] == "unknown" or cfg_algo[name] == "unknown":
            unknown_here += 1
            continue
        entry = {
            **entry,
            "env": cfg_env[name],
            "algo": cfg_algo[name],
            "overrides": cfg_overrides[name],
        }
        # A reference run without maps is dropped, and only this config can say
        # which runs those are. An ablation run is kept without them: its hp
        # value still names a row.
        if is_reference(entry) and entry["maps"] is None:
            without_maps += 1
            continue
        config_runs[name] = entry
    if unknown_here:
        vprint(
            f"[{config_path.name}] {unknown_here} scanned run(s) match no YAML "
            f"under its exemptions and are not drawn."
        )
    if without_maps:
        vprint(
            f"[{config_path.name}] {without_maps} reference run(s) have no maps "
            f"and are not drawn."
        )

    # Reindex scanned runs by (env, algo, v_key): the reference keys on None,
    # every other run on its canonical hp value.
    collected: dict = {}          # (env, algo, v_key) -> {frac: {key: array}}
    first_cfg_dir: dict = {}
    duplicate_pairs: list = []
    off_ablation: list = []       # runs overriding a key the ablation does not vary

    for cfg_dir_name, entry in config_runs.items():
        if entry["algo"] not in active_algos:
            continue
        # The ablated hyperparameter is the only key a run may differ from its
        # config files in: a run overriding anything else is another
        # configuration, and bucketing it here would draw it as this one, or
        # report it as a duplicate of it.
        other = {k: v for k, v in entry["overrides"].items() if k != hp_key}
        if other:
            off_ablation.append((cfg_dir_name, entry["env"], entry["algo"], other))
            continue
        if is_reference(entry):
            v_key = None
        else:
            v_key = _canon(entry["flat"].get(hp_key))
        key = (entry["env"], entry["algo"], v_key)
        if key in collected:
            duplicate_pairs.append(
                (cfg_dir_name, entry["env"], entry["algo"], v_key, first_cfg_dir[key])
            )
            continue
        collected[key] = entry["maps"]
        first_cfg_dir[key] = cfg_dir_name

    if not collected:
        vprint(f"[{config_path.name}] No matching runs.")
        continue

    # First reference algo's memory per env → source of `obs`, `bin_centers`,
    # `scale` for the pseudocount recomputation. Every row is drawn from this one
    # memory: the reference row is its visit count, and an ablation row is
    # `pseudocount(radius=v)` over the same data, so the figure compares the
    # radii on one set of samples rather than on whatever each run collected.
    ref_data_per_env: dict = {}
    for entry in config_runs.values():
        if (
            is_reference(entry)
            and entry["ref_data"] is not None
            and entry["env"] not in ref_data_per_env
        ):
            ref_data_per_env[entry["env"]] = entry["ref_data"]

    if not ref_data_per_env:
        # Every row is drawn from the reference memory, so without one there is
        # nothing to draw at all -- including the pseudocount rows, whose own
        # runs are never read. Said here rather than left to the pruning below,
        # which reports each row missing without saying what they have in common.
        candidates = [
            (name, e) for name, e in config_runs.items()
            if e["algo"] in ref_algos and is_null(e["flat"].get(hp_key))
        ]
        print(
            f"\n[{config_path.name}] No reference memory: "
            f"nothing can be drawn, since every row comes from it."
        )
        if not candidates:
            seen = sorted({
                str(_canon(e["flat"].get(hp_key)))
                for e in config_runs.values() if e["algo"] in ref_algos
            })
            print(
                f"  No run of {sorted(ref_algos)} has {hp_key} null. "
                f"Values of {hp_key} among those runs: {seen}."
            )
        else:
            print(
                f"  {len(candidates)} run(s) qualify, none with a memory read "
                f"at seed {args.rng_seed} (see --rng_seed):"
            )
            for name, e in candidates[:10]:
                found = (root / name / str(args.rng_seed) / "memory.npz").exists()
                print(
                    f"    {name} ({e['env']}): memory "
                    f"{'found' if found else 'MISSING'}, "
                    f"maps {'built' if e['maps'] is not None else 'none'}"
                )
        continue

    if duplicate_pairs:
        print(f"\n[{config_path.name}] Duplicate cfg dirs (kept first):")
        for d, en, al, v, primary in duplicate_pairs:
            print(f"  {d}  →  {en}/{al}/{v}  (kept: {primary})")

    if off_ablation:
        print(
            f"\n[{config_path.name}] {len(off_ablation)} run(s) override a key "
            f"the ablation does not vary and are not drawn:"
        )
        for d, en, al, other in off_ablation:
            shown = ", ".join(f"{k}={v}" for k, v in sorted(other.items()))
            print(f"  {d}  →  {en}/{al}  ({shown})")

    if not groups:
        groups = {"": sorted({en for (en, _, _) in collected})}
    ordered_envs = (
        list(flat_envs.keys()) if flat_envs
        else sorted({en for (en, _, _) in collected})
    )

    # Precompute pseudocount arrays for ablation rows.
    # Keyed by (env, v_key, frac). Non-numeric v_key or missing ref_data → skipped.
    pseudo_cache: dict = {}
    for en in ordered_envs:
        d = ref_data_per_env.get(en)
        if d is None:
            continue

        v_keys_numeric = [
            (v_key, _radius(v_key))
            for v_key in {r["v_key"] for r in rows if r["kind"] == "pseudo"}
            if _radius(v_key) is not None
        ]
        if not v_keys_numeric:
            continue

        for (vk, frac), arr in _pseudocounts(d, v_keys_numeric, FRACTIONS).items():
            pseudo_cache[(en, vk, frac)] = arr

    def _row_arr(en, ri, frac, key, source=None):
        """Dispatch: reference row → visit_count from that ref's memory;
        ablation row → pseudocount(radius=v_key) on that same memory.

        `source` is the row list `ri` indexes into, which is not always the one
        being drawn: the pruning below walks the rows as they were before it
        dropped any."""

        row = (rows if source is None else source)[ri]
        if row["kind"] == "true":
            per_frac = collected.get((en, row["algo_id"], None), {})
            return per_frac.get(frac, {}).get(key)
        return pseudo_cache.get((en, row["v_key"], frac))

    def _scale_group(ri):
        """Every row is a count, so they all share one colour range."""

        return "count"

    # Drop rows/envs with no data anywhere (checked against `_row_arr` so
    # ablation rows without a numeric hp or without ref_data are pruned).
    _probe_frac = FRACTIONS[-1]
    _probe_key  = MEMORY_KEYS[0]
    rows_orig = list(rows)
    _drawn = {
        ri: any(
            _row_arr(en, ri, _probe_frac, _probe_key, rows_orig) is not None
            for en in ordered_envs
        )
        for ri in range(len(rows_orig))
    }
    missing_row_labels = [
        r["label"] for ri, r in enumerate(rows_orig) if not _drawn[ri]
    ]
    # Against `rows_orig`, and before `rows` is rebound: an environment has no
    # data when none of the rows the config asked for has any there, and the
    # pruned list no longer holds them all.
    missing_envs_list = [
        e for e in ordered_envs
        if not any(
            _row_arr(e, ri, _probe_frac, _probe_key, rows_orig) is not None
            for ri in range(len(rows_orig))
        )
    ]
    rows = [r for ri, r in enumerate(rows_orig) if _drawn[ri]]
    all_envs = list(ordered_envs)
    ordered_envs = [e for e in ordered_envs if e not in missing_envs_list]

    vprint(f"\n{'='*60}\nAblation config: {config_path.name}\n{'='*60}")
    if missing_row_labels:
        vprint(f"  [WARNING] Missing rows (no memory data): {missing_row_labels}")
    if missing_envs_list:
        vprint(
            f"  [WARNING] Missing environments (no memory data): "
            f"{missing_envs_list}"
        )
    if not rows or not ordered_envs:
        vprint("  No data at all, skipping.")
        continue

    # Pooled per (environment, key, scale group) over every row and fraction, for
    # --shared_vmap=all_figures. Rebuilt per config, since pseudo_cache is too.
    env_vmins: dict = {}
    env_vmaxs: dict = {}
    for _key in MEMORY_KEYS:
        for en in ordered_envs:
            for ri in range(len(rows)):
                for frac in FRACTIONS:
                    arr = _row_arr(en, ri, frac, _key)
                    if arr is None or arr.size == 0:
                        continue
                    lo = float(np.nanmin(arr))
                    hi = float(np.nanmax(arr))
                    if not (np.isfinite(lo) and np.isfinite(hi)):
                        continue
                    ek = (en, scale_key(_key), _scale_group(ri))
                    if ek not in env_vmins or lo < env_vmins[ek]:
                        env_vmins[ek] = lo
                    if ek not in env_vmaxs or hi > env_vmaxs[ek]:
                        env_vmaxs[ek] = hi

    # The config names itself by where it sits under configs/plots, not by its
    # stem: a directory of configs holds more than one `default.yaml`, and two of
    # them would write their figures over each other.
    cfg_out = output_root / config_name
    cfg_out.mkdir(parents=True, exist_ok=True)

    for key in MEMORY_KEYS:
        key_out = cfg_out / key
        key_out.mkdir(exist_ok=True)

        for group_name, group_envs in groups.items():
            group_envs = [e for e in group_envs if e in ordered_envs]
            if not group_envs:
                continue
            suffix = f"{group_name}" if group_name else ""

            for frac in FRACTIONS:
                pct_suffix = f"_{int(round(frac * 100))}"

                last_arrays = {}
                for en in group_envs:
                    for ri in range(len(rows)):
                        last_arrays[(en, ri)] = _row_arr(en, ri, frac, key)

                if all(a is None for a in last_arrays.values()):
                    continue

                col_ranges = {}
                if args.shared_vmap == "same_figure":
                    shared_keys = [
                        k for k in MEMORY_KEYS if scale_key(k) == scale_key(key)
                    ]
                    for en in group_envs:
                        for g in SCALE_GROUPS:
                            col_ranges[(en, g)] = pooled_range(
                                _row_arr(en, ri, frac, k2)
                                for k2 in shared_keys for ri in range(len(rows))
                                if _scale_group(ri) == g
                            )

                nrows, ncols = (
                    (len(group_envs), len(rows)) if args.rotate
                    else (len(rows), len(group_envs))
                )
                # Which panels share a bar: an (environment, scale group) each,
                # or one panel each. Settled before the grid is built, since it
                # decides whether a row gap has to be opened for the bars.
                cell_keys = {
                    ((en_i, ri) if args.rotate else (ri, en_i)): (
                        (en, _scale_group(ri)) if args.shared_vmap
                        else ((en_i, ri) if args.rotate else (ri, en_i))
                    )
                    for ri in range(len(rows))
                    for en_i, en in enumerate(group_envs)
                }
                fig, axs = grid_for(cell_keys, nrows, ncols)

                cell_ranges = {}
                for ri, row in enumerate(rows):
                    for en_i, en in enumerate(group_envs):
                        r = en_i if args.rotate else ri
                        c = ri if args.rotate else en_i
                        ax = axs[r][c]
                        arr = last_arrays[(en, ri)]
                        g = _scale_group(ri)
                        if args.shared_vmap == "all_figures":
                            ek = (en, scale_key(key), g)
                            vmin = env_vmins.get(ek, 0.0)
                            vmax = env_vmaxs.get(ek, 1.0)
                        elif args.shared_vmap == "same_figure":
                            vmin, vmax = col_ranges[(en, g)]
                        else:
                            vmin, vmax = _arr_range(arr)
                        plot_cell(ax, arr, vmin, vmax)
                        if arr is not None and np.any(np.isfinite(arr)):
                            cell_ranges[(r, c)] = (vmin, vmax)
                        if r == 0:
                            _t = row["label"] if args.rotate else flat_envs.get(en, en)
                            ax.set_title(
                                _t,
                                fontsize=font_size * LABEL_SCALE,
                                pad=3,
                                **_tex(_t),
                            )
                        if c == 0:
                            _yl = flat_envs.get(en, en) if args.rotate else row["label"]
                            ax.set_ylabel(
                                _yl,
                                fontsize=font_size * LABEL_SCALE,
                                **_tex(_yl),
                            )

                add_colorbars(fig, axs, cell_keys, cell_ranges)

                rot_suffix = "_rotated" if args.rotate else ""
                fname = key_out / f"{suffix}{pct_suffix}{rot_suffix}.png"
                dpi = auto_dpi(
                    list(last_arrays.values()),
                    args.max_dpi,
                    args.pixels_per_cell,
                    SUBPLOT_SIZE,
                )
                fig.savefig(fname, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
                plt.close(fig)
                vprint(f"  Saved: {fname.relative_to(output_root)}")

        # --- One figure per environment: a column per memory fraction, a row per
        # (algorithm, hyperparameter value).
        if len(FRACTIONS) > 1:
            for en in ordered_envs:
                step_arrays = {}
                for ri in range(len(rows)):
                    for fi, frac in enumerate(FRACTIONS):
                        step_arrays[(ri, fi)] = _row_arr(en, ri, frac, key)

                valid = [a for a in step_arrays.values() if a is not None]
                if not valid:
                    continue

                if args.shared_vmap == "all_figures":
                    fig_ranges = {
                        g: (
                            env_vmins.get((en, scale_key(key), g), 0.0),
                            env_vmaxs.get((en, scale_key(key), g), 1.0),
                        )
                        for g in SCALE_GROUPS
                    }
                elif args.shared_vmap == "same_figure":
                    shared_keys = [
                        k for k in MEMORY_KEYS if scale_key(k) == scale_key(key)
                    ]
                    fig_ranges = {
                        g: pooled_range(
                            _row_arr(en, ri, fr, k2)
                            for k2 in shared_keys for ri in range(len(rows))
                            if _scale_group(ri) == g for fr in FRACTIONS
                        )
                        for g in SCALE_GROUPS
                    }
                else:
                    fig_ranges = None

                n_fracs = len(FRACTIONS)
                nrows, ncols = (
                    (n_fracs, len(rows)) if args.rotate else (len(rows), n_fracs)
                )
                # A range per scale group -- every fraction of every row on one --
                # or one panel each.
                cell_keys = {
                    ((fi, ri) if args.rotate else (ri, fi)): (
                        _scale_group(ri) if fig_ranges is not None
                        else ((fi, ri) if args.rotate else (ri, fi))
                    )
                    for ri in range(len(rows)) for fi in range(n_fracs)
                }
                fig, axs = grid_for(cell_keys, nrows, ncols)

                cell_ranges = {}
                for ri, row in enumerate(rows):
                    for fi in range(n_fracs):
                        r = fi if args.rotate else ri
                        c = ri if args.rotate else fi
                        ax = axs[r][c]
                        arr = step_arrays[(ri, fi)]
                        if fig_ranges is None:
                            vmin, vmax = _arr_range(arr)
                        else:
                            vmin, vmax = fig_ranges[_scale_group(ri)]
                        plot_cell(ax, arr, vmin, vmax)
                        if arr is not None and np.any(np.isfinite(arr)):
                            cell_ranges[(r, c)] = (vmin, vmax)
                        pct_label = f"{int(round(FRACTIONS[fi] * 100))}%"
                        if r == 0:
                            _t = row["label"] if args.rotate else pct_label
                            ax.set_title(
                                _t,
                                fontsize=font_size * LABEL_SCALE,
                                pad=3,
                                **_tex(_t),
                            )
                        if c == 0:
                            _yl = pct_label if args.rotate else row["label"]
                            ax.set_ylabel(
                                _yl,
                                fontsize=font_size * LABEL_SCALE,
                                **_tex(_yl),
                            )

                add_colorbars(fig, axs, cell_keys, cell_ranges)

                rot_suffix = "_rotated" if args.rotate else ""
                fname = key_out / f"{en}{rot_suffix}.png"
                dpi = auto_dpi(
                    list(step_arrays.values()),
                    args.max_dpi,
                    args.pixels_per_cell,
                    SUBPLOT_SIZE,
                )
                fig.savefig(fname, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
                plt.close(fig)
                vprint(f"  Saved: {fname.relative_to(output_root)}")

print("\nDone.")
