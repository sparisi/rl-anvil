"""Heatmaps from memory.npz, the replay-memory dump a run saves at the end.

Two kinds of figure, both of which --rotate transposes:

    all_<fraction>.png  a row per configuration and a column per environment,
                        counted over the first <fraction> of the memory. Only the
                        full memory is drawn unless --progression_step asks for
                        more checkpoints.
    <environment>.png   written only when --progression_step is below 1.0: one
                        environment per figure, a column per checkpoint, so a row
                        reads left to right as the visitation building up.

An entry of a plot config is a whole configuration: its keys are composed
through Hydra with everything configs/default.yaml and the files it selects put
under them, and a run is drawn only when its WHOLE configuration is the one that
comes out. A plot config's `ignored_cfg_keys` are exempt from the comparison
against the environment and algorithm YAMLs, for keys a launch set for every run
alike.

--shared_vmap draws several heatmaps on one (vmin, vmax), always pooled within one
environment and never across them: `same_figure` pools over the maps of one figure,
`all_figures` over every map of that environment the script draws, so the scale it
gets is the same in every figure it appears in. Either way the pool is over a
family of maps rather than one map: (`visit_count`, `pseudocount`) share the same
vmap, and so do (`goal_selected_count`, `goal_reached_count`).

--colorbar draws the colour scale under the panels it applies to: one per
environment under --shared_vmap, one per panel when it is unset. Panels are drawn
larger so the bar and its ticks stay legible.

Output goes to <data_dir>/<output>/heatmaps/<plot config>/<memory key>/.

Example

    python plot_heatmaps.py -f data_example -p example --progression_step=0.1 --shared_vmap --log_scale -v
"""

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import argparse
import re
import yaml
from pathlib import Path
from tqdm import tqdm

from src.utils.plot import (
    assign_groups, composed_mask, ensure_font, entry_config,
    env_groups, flatten_cfg, ignored_cfg_keys, load_config_group,
    load_plot_configs, memory_fractions, plot_entries, plot_labels,
    split_config_label, tex_kwargs as _tex,
)
from src.utils.heatmaps import (
    CBAR_SUBPLOT_SIZE, add_colorbar, arr_range, auto_dpi, cbar_groups, cbar_hspace,
    compute_maps_all_fractions, plot_map, pooled_range, scale_funcs, scale_key,
    viridis_black_bad, quiet_empty_maps,
)

FONT_SIZE    = 7
# Panel titles and axis labels, as a multiple of FONT_SIZE. They name the
# environment and the configuration of a panel a few centimetres across, so they
# are set well above the tick and colour-bar text.
LABEL_SCALE  = 2.2
SUBPLOT_SIZE = 1.5   # inches per panel, without a colour bar under it

MEMORY_KEYS = [
    "visit_count",
    "pseudocount",
    "goal_selected_count",
    "goal_reached_count",
]


# {family: the keys in it}, so pooling a family walks its keys without asking
# scale_key about every one of them again.
SCALE_FAMILY_KEYS = {
    family: [k for k in MEMORY_KEYS if scale_key(k) == family]
    for family in {scale_key(k) for k in MEMORY_KEYS}
}

# The only fields of memory.npz these maps are built from. Everything else it
# holds is left alone rather than decompressed on load.
#
# The goal maps are not stored and are derived: how often a bin was aimed at
# comes from `goal_obs` and `goal_valid`, and how often it was then hit needs
# `act` and `goal_act` beside them. Leaving any of the four out does not draw an
# empty map, it draws no map at all.
NEEDED_MEM_KEYS = {"obs", "count", "goal_obs", "goal_valid", "act", "goal_act"}

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--folder",
    default="data_dir",
    metavar="DIR",
    help="A data directory, or a single seed directory "
         "(<data_dir>/<config_id>/<rng_seed>) to draw one run.",
)
parser.add_argument(
    "--rng_seed",
    type=int,
    default=0,
    metavar="N",
    help="Which seed of each configuration to draw. Ignored when -f is already a "
         "seed directory. Default: 0.",
)
parser.add_argument(
    "-o", "--output",
    default="plots",
    help="Output root, under the data directory the runs came from. Figures go in "
         "<root>/heatmaps/<plot config>, so one data directory holds the output of "
         "every plotting script without one overwriting another.",
)
parser.add_argument(
    "-a", "--algorithms",
    default="configs/algorithm",
    help="Directory of algorithm YAMLs.",
)
parser.add_argument(
    "-e", "--environments",
    default="configs/environment",
    help="Directory of environment YAMLs.",
)
parser.add_argument(
    "-p", "--plot_config",
    required=True,
    metavar="NAME",
    help="Plot config: which configurations become the rows, over which "
         "environments, and what to call them. A name without .yaml is looked for "
         "in configs/plots, or give a path; a directory is a set of configs and "
         "every one of them is drawn.",
)
parser.add_argument(
    "--progression_step",
    type=float,
    default=1.0,
    help="Step (in (0, 1]) between memory-fraction checkpoints. Below 1 it also "
         "writes a figure per environment whose columns are those checkpoints, so "
         "a row reads left to right as the visitation building up. Default 1.0 = "
         "the full memory only.",
)
parser.add_argument(
    "--shared_vmap",
    choices=["all_figures", "same_figure"],
    default=None,
    help="Share vmin/vmax across heatmaps, always within one environment. "
         "'same_figure': pool over the maps of one figure. 'all_figures': pool "
         "over every map of that environment the script draws, so its scale is "
         "the same in every figure. Unset: each heatmap uses its own min/max, "
         "which hides how a progression fills up.",
)
parser.add_argument(
    "--colorbar",
    action="store_true",
    help="Draw the colour scale under the panels it applies to: one per "
         "environment under --shared_vmap, one per panel when it is unset. "
         "Panels are drawn larger.",
)
parser.add_argument("--max_dpi", type=int, default=1000)
parser.add_argument("--pixels_per_cell", type=int, default=15)
parser.add_argument(
    "--log_scale",
    action="store_true",
    help="Apply np.log1p to the maps and their bounds before plotting.",
)
parser.add_argument(
    "--rotate",
    action="store_true",
    help="Transpose every figure: what a row holds goes in a column instead.",
)
parser.add_argument(
    "-v", "--verbose",
    action="store_true",
    help="Print progress messages. The recap is always printed.",
)
args = parser.parse_args()

# What a panel is actually drawn at. A colour bar and its tick labels need more
# room than a panel of the default size leaves under it. auto_dpi divides by the
# same number, so a cell keeps its pixel budget and only the figure grows.
#
# A constant of its own rather than SUBPLOT_SIZE reassigned: make_grid took the
# size as a default argument, which binds once when the function is defined, so
# the two were only ever equal because this line happened to come first.
PANEL_SIZE = CBAR_SUBPLOT_SIZE if args.colorbar else SUBPLOT_SIZE

ensure_font()
plt.rcParams["font.size"] = FONT_SIZE

_CMAP = viridis_black_bad()
_FORWARD, _INVERSE = scale_funcs(log_scale=args.log_scale)


def vprint(*a, **kw):
    """Print only under --verbose."""

    if args.verbose:
        print(*a, **kw)


# --- Naming ---

def display_name(name, labels):
    """A config label under its display name. An inexactly matched run is labelled
    `<stem> (key=value)`, and the labels file names the stem, so only that part is
    substituted and the overrides are kept."""

    stem, _ = split_config_label(str(name))
    return str(name).replace(stem, labels.get(stem, stem), 1)


def safe_stem(name):
    """A config label as a filename: an override suffix carries spaces, parens and
    `=`, none of which belong in one."""

    return re.sub(r"[^\w.=+-]+", "_", str(name)).strip("_") or "unnamed"


# --- Drawing ---

def add_colorbars(fig, axs, cell_keys, cell_ranges):
    """One bar per set of panels drawn on a shared range (see cbar_groups).

    `cell_ranges` holds the bounds as the maps hold them; plot_cell compresses
    them on its way to the colours, so a bar is normalised on the compressed
    bounds -- its gradient is then the one on the panels -- and its ticks are
    labelled with the values they came from."""

    if not args.colorbar:
        return
    groups = cbar_groups(cell_keys, cell_ranges)
    for (vmin, vmax), cells in groups:
        add_colorbar(
            fig,
            [axs[r][c] for r, c in cells],
            _FORWARD(vmin),
            _FORWARD(vmax),
            cmap=_CMAP,
            inverse=_INVERSE,
            fontsize=FONT_SIZE,
        )


def make_grid(nrows, ncols, subplot_size=None, hspace=0.05):
    """A Figure of `nrows` x `ncols` touching panels, as `axs[row][col]`.

    The panels fill the canvas edge to edge. A gridspec's default margins are a
    tenth of the figure on every side, and the saved crop keeps whatever an
    artist sits in -- so those margins come out as a border of white around a grid
    that is meant to touch. What is left outside the panels is the labels.

    An `hspace` above the default opens a gap between the rows for a colour bar
    that ends above the bottom row, and the figure grows by exactly that gap: the
    panels keep their size rather than shrinking to make room.

    `subplot_size` defaults to PANEL_SIZE, read here rather than bound as a
    default argument so that it is the one this run is drawing at."""

    if subplot_size is None:
        subplot_size = PANEL_SIZE
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


def plot_cell(ax, arr, vmin, vmax):
    """One map in one panel, at the scale this run was asked for."""

    with quiet_empty_maps():
        plot_map(ax, arr, vmin, vmax, cmap=_CMAP, log_scale=args.log_scale)


def resolve_runs(path):
    """Walk what `-f` was given and return (seed directories to read,
    configurations that had none).

    A seed directory is one holding a memory.npz, and `path` is taken to be one
    when it holds it -- that is how a single run is drawn. Anything else is read
    as a data directory, and each configuration contributes its --rng_seed."""

    if (path / "memory.npz").is_file():
        return [path], []

    # A seed directory that saved no memory: say so, rather than walk it as a data
    # directory and report having found no configurations inside it.
    if (path.parent / "cfg.yaml").is_file():
        contents = ", ".join(sorted(p.name for p in path.iterdir())) or "nothing"
        raise SystemExit(
            f"{path} is a run directory but holds no memory.npz, so there is "
            f"nothing to draw. It holds: {contents}. memory.npz is only written "
            f"when results.save_memory=True."
        )

    found, without = [], []
    for cfg_dir in sorted(p for p in path.iterdir() if p.is_dir()):
        seed_dir = cfg_dir / str(args.rng_seed)
        if (seed_dir / "memory.npz").is_file():
            found.append(seed_dir)
        elif (cfg_dir / "cfg.yaml").is_file():
            without.append(cfg_dir.name)
    return found, without


# --- Resolve which runs to draw ---
root = Path(args.folder)
if not root.is_dir():
    raise SystemExit(f"Not a directory: {root}")
run_dirs, no_memory_dirs = resolve_runs(root)

if not run_dirs:
    seed_note = "" if (root / "memory.npz").is_file() else f" at seed {args.rng_seed}"
    raise SystemExit(
        f"No memory.npz under {root}{seed_note} ({len(no_memory_dirs)} "
        f"configuration(s) had none). memory.npz is only written when "
        f"results.save_memory=True."
    )

# A seed directory sits at <data_dir>/<config_id>/<rng_seed>, so the data
# directory is two levels up. Under it, <root>/heatmaps/<sweep>: the root is
# shared with every other plotting script, and the sweep's own directory keeps
# two sweeps of the same data apart -- "all" when there is no sweep to name it.
output_root = run_dirs[0].parent.parent / args.output / "heatmaps"

# Memory-fraction checkpoints, one column each.
try:
    FRACTIONS = memory_fractions(args.progression_step)
except ValueError as err:
    parser.error(f"--progression_step {err}")

# --- Read each run's configuration ---
flat_cfgs:   dict = {}   # "<config_id>/<seed>" -> flat cfg
run_cfgs:    dict = {}   # "<config_id>/<seed>" -> nested cfg
mem_files:   dict = {}   # "<config_id>/<seed>" -> path to memory.npz
no_cfg_dirs: list = []

for seed_dir in tqdm(
    run_dirs, desc="Reading configs", unit="cfg", disable=not args.verbose
):
    # cfg.yaml lives one level up, shared by every seed of the configuration.
    cfg_file = seed_dir.parent / "cfg.yaml"
    if not cfg_file.is_file():
        no_cfg_dirs.append(seed_dir.parent.name)
        continue

    run_cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    name = f"{seed_dir.parent.name}/{seed_dir.name}"
    # Interpolations kept, as src.utils.id.composed_config keeps them: the two
    # are compared key by key below, and a key dropped from one side and not the
    # other is a key the comparison reads as missing from the run.
    flat_cfgs[name] = flatten_cfg(run_cfg, drop_interpolations=False)
    run_cfgs[name]  = run_cfg
    mem_files[name] = seed_dir / "memory.npz"

if not flat_cfgs:
    raise SystemExit(
        f"Found {len(run_dirs)} memory.npz, but no cfg.yaml in the directory above any "
        f"of them. -f takes a data directory or a <data_dir>/<config_id>/<rng_seed>."
    )

df = pd.DataFrame.from_dict(flat_cfgs, orient="index")
df.columns = pd.MultiIndex.from_tuples([("hyperparameter", c) for c in df.columns])

plot_cfgs = load_plot_configs(args.plot_config)
vprint(f"\n{len(plot_cfgs)} plot config(s): {[n for n, _, _ in plot_cfgs]}")

# --- Identify environment and algorithm ---
# Neither is recorded by a run: both are matched back from their config YAMLs,
# the file every key a run recorded agrees with. A run agreeing with no
# environment YAML is "unknown" there and is not drawn. The algorithm is named
# for the label alone: which run fills a cell is decided by the whole
# configuration an entry composes, which already holds every key an algorithm
# YAML declares.
env_configs = load_config_group(args.environments)
algo_configs = load_config_group(args.algorithms)

# The answers `assign_groups` has already given, by the exemptions that produced
# them: configs declaring the same ones -- usually every one of them -- get the
# same answer, and the frame is compared against every YAML to arrive at it.
_assigned: dict = {}


def run_of(env, selector, label, ignored, matched):
    """The one run of `env` whose whole recorded configuration is the one the
    entry composes to there, or None.

    The comparison covers every key, the ones the entry leaves out included: a
    run launched with another value for one of those is a different
    configuration, and belongs to whichever entry composes to it.

    `ignored` and `matched` are the pair `assign_groups` returned for this plot
    config, and have to be that pair: a mask built under one set of exemptions
    says nothing about a comparison made under another.

    A configuration matching several runs of one environment draws the first,
    since a cell holds one map, and says which ones it passed over: two runs of
    one configuration and one environment means a key the plot config does not
    name is varying under it."""

    flat = entry_config(env, selector, label)
    if flat is None:
        return None
    names = list(df.index[composed_mask(df, flat, ignore=ignored) & matched])
    if not names:
        return None
    if len(names) > 1:
        print(
            f"WARNING: '{label}' matches {len(names)} runs of {env} and a cell "
            f"holds one map, so {names[0]} is drawn and {names[1:]} are not."
        )
    return names[0]

# --- Which runs the plot configs ask for ---
# Decided before any memory is read, and read once for all of them. A cell holds
# one run, so a data directory of hundreds is a handful of maps -- and a
# memory.npz costs a decompression whether it ends up in a figure or not.
#
# config name -> (entries, {group: envs}, {(entry, env): run}, environments).
# The environments are its own: which YAML a run matches depends on the
# exemptions, and those are the config's.
layout: dict = {}
for config_name, _cfg_path, plot_cfg in plot_cfgs:
    # Only the label and the selector: a heatmap has no use for the styling
    # plot_results.py puts on top of the same block.
    entries = [(label, sel) for _group, label, sel in plot_entries(plot_cfg)]
    if not entries:
        print(f"WARNING: {config_name} declares no configurations to plot, skipping.")
        continue

    # This config's own exemptions, not every config's: one config's
    # `ignored_cfg_keys` says nothing about the runs another one draws.
    ignored_keys = ignored_cfg_keys(plot_cfg)
    if ignored_keys:
        vprint(
            f"{config_name} exempts from config matching: {sorted(ignored_keys)}"
        )
    # A cell sits in a column of one environment, so the environment label is
    # what places a run, and it comes from the environment YAMLs alone. The
    # algorithm label is assigned for the figures to read. `run_of` selects by
    # the whole configuration an entry composes, a comparison that already holds
    # every key an algorithm YAML declares, at the value the configuration
    # composes to. A YAML declaring a default a sweep overrode agrees with the
    # runs it launched on every key but that one, which leaves them "unknown"
    # there.
    _env, _algo, env_stem, matched = assign_groups(
        df,
        env_configs,
        algo_configs,
        ignored_keys,
        env_id_fallback=False,
        algo_optional=False,
        algo_decides_usable=False,
        cache=_assigned,
    )
    # Without the "unknown" stem, so a config that declares no environment group
    # of its own does not get one holding every run that matched no YAML.
    envs = sorted(set(env_stem[matched]), key=str)
    groups = env_groups(plot_cfg) or {"": envs}
    selected = {
        (li, env): run_of(env, sel, label, ignored_keys, matched)
        for li, (label, sel) in enumerate(entries)
        for group_envs in groups.values() for env in group_envs if env in envs
    }
    layout[config_name] = (
        entries,
        groups,
        {k: n for k, n in selected.items() if n is not None},
        envs,
    )

needed = sorted({n for _e, _g, sel, _v in layout.values() for n in sel.values()})
if not needed:
    raise SystemExit("No run matches any configuration of the plot config(s).")
vprint(f"{len(needed)} run(s) to read, of {len(df)} found")

# --- Load memory, compute maps ---
collected:       dict = {}   # name -> {fraction: {key: 2D array}}
no_counter_dirs: list = []
corrupt_dirs:    list = []

with tqdm(needed, desc="Loading memory", unit="run", disable=not args.verbose) as pbar:
    for name in pbar:
        # The run's own name, not the environment and algorithm it was assigned:
        # those are decided per plot config, and this is one bar for all of them.
        pbar.set_postfix_str(name)

        try:
            with np.load(mem_files[name]) as data:
                mem = {k: data[k] for k in data.files if k in NEEDED_MEM_KEYS}
        except Exception as e:
            vprint(f"  WARNING: could not read {mem_files[name]}: {e}")
            corrupt_dirs.append(name)
            continue

        try:
            with quiet_empty_maps():
                maps_per_frac = compute_maps_all_fractions(
                    run_cfgs[name], mem, FRACTIONS
                )
        except Exception as e:
            vprint(f"  WARNING: compute_maps failed for {name}: {e}")
            continue
        if maps_per_frac is None:
            no_counter_dirs.append(name)
            continue

        collected[name] = maps_per_frac

# --- Recap ---
print(f"Runs found  : {len(run_dirs)}  (seed {args.rng_seed})")
print(
    f"Runs plotted: {len(collected)}"
    f"  ({len(no_cfg_dirs)} no cfg,"
    f" {len(no_memory_dirs)} configs with no memory.npz,"
    f" {len(no_counter_dirs)} no tabular counter,"
    f" {len(corrupt_dirs)} unreadable)"
)

if not collected:
    raise SystemExit("Nothing to plot.")

# Which environments each run belongs to, over every config drawn. `all_figures`
# pools a range over every figure the script writes, and it pools it per
# environment, so it needs the environments of runs the config being drawn does
# not itself ask for.
#
# A set per run rather than one environment: which YAML a run matches depends on
# the keys a config exempts, so two configs can select the same run under
# different environments, and the range drawn under either has to hold it.
envs_of_run: dict = {}
for _entries, _groups, _sel, _envs in layout.values():
    for (_li, _env), _name in _sel.items():
        envs_of_run.setdefault(_name, set()).add(_env)


def env_pooled_range(env, family):
    """Return the (vmin, vmax) of one environment's maps of one scale family,
    over every run of it and every checkpoint.

    Per environment and never across them: two environments hold counts on
    scales that have nothing to do with each other, and one range over both
    flattens whichever is visited less into a single colour.

    Over every config drawn, not only the one being drawn: the scale is what
    makes two of its figures readable against each other."""

    with quiet_empty_maps():
        return pooled_range(
            collected[n][frac].get(k)
            for n in collected if env in envs_of_run.get(n, ())
            for frac in FRACTIONS for k in family
        )


# --- Figures ------------------------------------------------------------------
# A `-p` naming a directory is a set of configs that belong together -- one
# sweep's -- and every one of them is drawn, into its own directory. Which run
# fills which cell was settled before the memory was read (see `layout`).
for config_name, _cfg_path, plot_cfg in plot_cfgs:
    if config_name not in layout:
        continue
    vprint(f"\n{'#' * 60}\nPlot config: {config_name}\n{'#' * 60}")
    _, env_labels, _ = plot_labels(plot_cfg)
    entries, groups, selected, envs = layout[config_name]

    config_root = output_root / config_name
    config_root.mkdir(parents=True, exist_ok=True)

    for key in MEMORY_KEYS:
        key_out = config_root / key
        key_out.mkdir(parents=True, exist_ok=True)

        family = SCALE_FAMILY_KEYS[scale_key(key)]
        all_ranges = {}
        if args.shared_vmap == "all_figures":
            all_ranges = {
                env: env_pooled_range(env, family)
                for env in {e for _li, e in selected}
            }

        for group_name, group_envs in groups.items():
            present = [e for e in group_envs if e in envs]
            if not present:
                print(
                    f"WARNING: environments in {config_name} with no runs: "
                    f"{list(group_envs)}"
                )
                continue

            run_at = {
                (li, env): n for (li, env), n in selected.items()
                if env in present and n in collected
            }
            if not run_at:
                vprint(f"  Nothing selected for {group_name or '(unnamed)'}, skipping.")
                continue
            labels = [label for label, _ in entries]

            # --- A row per configuration, a column per environment, at one
            # checkpoint of the memory. A column is one environment under every
            # configuration, a row one configuration everywhere -- the comparison
            # a sweep is run to make. --rotate transposes it.
            for frac in FRACTIONS:
                arrays = {
                    (li, env): collected[n][frac].get(key)
                    for (li, env), n in run_at.items()
                }
                if not any(a is not None for a in arrays.values()):
                    continue

                # Per column, never over the whole figure: a column is an
                # environment, and see env_pooled_range for why a range never
                # crosses one.
                if args.shared_vmap == "all_figures":
                    ranges = {env: all_ranges[env] for env in present}
                elif args.shared_vmap == "same_figure":
                    # Over the family, not over the panels on screen: the figure
                    # of one key is read against the figure of its sibling, so
                    # both are pooled whichever of them is being drawn.
                    with quiet_empty_maps():
                        ranges = {
                            env: pooled_range([
                                collected[n][frac].get(k)
                                for (li, e), n in run_at.items() if e == env
                                for k in family
                            ])
                            for env in present
                        }
                else:
                    ranges = None

                nrows, ncols = (
                    (len(present), len(labels)) if args.rotate
                    else (len(labels), len(present))
                )
                # Which panels share a bar: an environment's, or one each.
                # Settled before the grid is built, since it decides whether a
                # row gap has to be opened for the bars.
                cell_keys = {
                    ((ei, li) if args.rotate else (li, ei)): (
                        env if ranges is not None
                        else ((ei, li) if args.rotate else (li, ei))
                    )
                    for li in range(len(labels)) for ei, env in enumerate(present)
                }
                fig, axs = make_grid(
                    nrows,
                    ncols,
                    hspace=cbar_hspace(cell_keys, nrows) if args.colorbar else 0.05,
                )

                cell_ranges = {}
                for li, label in enumerate(labels):
                    for ei, env in enumerate(present):
                        r, c = (ei, li) if args.rotate else (li, ei)
                        ax = axs[r][c]
                        arr = arrays.get((li, env))
                        with quiet_empty_maps():
                            vmin, vmax = (
                                ranges[env] if ranges is not None
                                else arr_range(arr)
                            )
                        plot_cell(ax, arr, vmin, vmax)
                        if arr is not None and np.any(np.isfinite(arr)):
                            cell_ranges[(r, c)] = (vmin, vmax)

                        env_text = display_name(env, env_labels)
                        if r == 0:
                            title = label if args.rotate else env_text
                            ax.set_title(
                                title,
                                fontsize=FONT_SIZE * LABEL_SCALE,
                                pad=3,
                                **_tex(title),
                            )
                        if c == 0:
                            ylabel = env_text if args.rotate else label
                            ax.set_ylabel(
                                ylabel,
                                fontsize=FONT_SIZE * LABEL_SCALE,
                                **_tex(ylabel),
                            )

                add_colorbars(fig, axs, cell_keys, cell_ranges)

                pct = int(round(frac * 100))
                suffix = f"_{group_name}" if group_name else ""
                rotated = "_rotated" if args.rotate else ""
                fname = key_out / f"all{suffix}_{pct}{rotated}.png"
                dpi = auto_dpi(
                    arrays.values(),
                    args.max_dpi,
                    args.pixels_per_cell,
                    PANEL_SIZE,
                )
                fig.savefig(fname, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
                plt.close(fig)
                vprint(
                    f"  Saved: {fname.relative_to(output_root)}  "
                    f"({len(labels)} config(s) x {len(present)} environment(s))"
                )

            # --- One figure per environment, a column per checkpoint, so a row
            # reads left to right as the visitation building up. Only worth
            # drawing when --progression_step asked for more than one checkpoint.
            if len(FRACTIONS) < 2:
                continue
            for env in present:
                names = [
                    (li, run_at[(li, env)])
                    for li in range(len(labels)) if (li, env) in run_at
                ]
                arrays = {
                    (li, fi): collected[n][frac].get(key)
                    for li, n in names for fi, frac in enumerate(FRACTIONS)
                }
                if all(a is None for a in arrays.values()):
                    continue

                if args.shared_vmap == "all_figures":
                    fig_range = all_ranges[env]
                elif args.shared_vmap == "same_figure":
                    # The family's maps, not just this key's, as above.
                    with quiet_empty_maps():
                        fig_range = pooled_range([
                            collected[n][frac].get(k)
                            for _li, n in names for frac in FRACTIONS for k in family
                        ])
                else:
                    fig_range = None

                n_frac = len(FRACTIONS)
                nrows, ncols = (
                    (n_frac, len(names)) if args.rotate else (len(names), n_frac)
                )
                # One figure is one environment here, so its range covers every
                # panel, or there is one per panel.
                cell_keys = {
                    ((fi, ni) if args.rotate else (ni, fi)): (
                        "" if fig_range is not None
                        else ((fi, ni) if args.rotate else (ni, fi))
                    )
                    for ni in range(len(names)) for fi in range(n_frac)
                }
                fig, axs = make_grid(
                    nrows,
                    ncols,
                    hspace=cbar_hspace(cell_keys, nrows) if args.colorbar else 0.05,
                )

                cell_ranges = {}
                for ni, (li, _name) in enumerate(names):
                    for fi in range(n_frac):
                        r, c = (fi, ni) if args.rotate else (ni, fi)
                        ax = axs[r][c]
                        arr = arrays[(li, fi)]
                        with quiet_empty_maps():
                            vmin, vmax = (
                                fig_range if fig_range is not None
                                else arr_range(arr)
                            )
                        plot_cell(ax, arr, vmin, vmax)
                        if arr is not None and np.any(np.isfinite(arr)):
                            cell_ranges[(r, c)] = (vmin, vmax)

                        pct_text = f"{int(round(FRACTIONS[fi] * 100))}%"
                        run_text = labels[li]
                        if r == 0:
                            title = run_text if args.rotate else pct_text
                            ax.set_title(
                                title,
                                fontsize=FONT_SIZE * LABEL_SCALE,
                                pad=3,
                                **_tex(title),
                            )
                        if c == 0:
                            ylabel = pct_text if args.rotate else run_text
                            ax.set_ylabel(
                                ylabel,
                                fontsize=FONT_SIZE * LABEL_SCALE,
                                **_tex(ylabel),
                            )

                add_colorbars(fig, axs, cell_keys, cell_ranges)

                rotated = "_rotated" if args.rotate else ""
                fname = key_out / f"{safe_stem(env)}{rotated}.png"
                dpi = auto_dpi(
                    arrays.values(),
                    args.max_dpi,
                    args.pixels_per_cell,
                    PANEL_SIZE,
                )
                fig.savefig(fname, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
                plt.close(fig)
                vprint(
                    f"  Saved: {fname.relative_to(output_root)}  "
                    f"({len(names)} config(s) x {n_frac})"
                )

print("\nDone.")
