"""Coverage and entropy of a run's whole memory.npz, as one number per run.

The memory is read in full and reduced to a single value per statistic, so the
figures are bars rather than curves: a subplot per environment, a bar per
configuration, and a 95% confidence interval over the seeds.

    sa_%      fraction of (bin, action) pairs visited at least once
    sa_h      entropy of the visits over those pairs, normalized to [0, 1]
    sa_h_knn  joint entropy of the observations and the actions, in nats

The first two are what `exploration_stats` in src/experiment.py computes, read
off a fresh binning of the observations. The binning comes from
src/pseudocount.py, so an observation lands in the bin the run would have put it
in, and --bins writes a figure per bin size. An environment whose counter bins
nothing gets the kNN entropy alone.

The third is estimated from the observations themselves (does not depend on bin
size).

--discrete_cols can be used to filter out discrete observations, making them
part of the discrete action space.

Output goes to <data_dir>/<output>/knn_entropy/<plot config>/.

Example

    python plot_knn_entropy.py -f gcrl_lunar -p gcrl/lunar_full --discrete_cols 6 7 -v
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import re
import sys
import yaml
from pathlib import Path
from tqdm import tqdm

from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

import src.pseudocount as pseudocount
from src.pseudocount import BinnedCount

from src.utils.plot import (
    ENV,
    assign_groups,
    ci_bounds,
    composed_mask,
    detect_precision,
    draw_bars,
    ensure_font,
    entry_config,
    env_groups,
    flatten_cfg,
    ignored_cfg_keys,
    load_config_group,
    load_plot_configs,
    plot_labels,
    save_figure,
    set_3_ticks,
    style_entries,
    tex_kwargs,
)

# A plot config names its configurations whatever reads best -- "ε-greedy (1 → 0)"
# -- and a console encoding that has no ε raises rather than prints. The figures
# carry the label either way; only the recap has to give ground.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

FONT_SIZE = 12
SUBPLOT_W = 3.0  # inches per subplot
SUBPLOT_H = 2.4

# Where a plot config named without a path is looked for.
PLOT_CONFIG_DIR = "configs/plots"

# The statistics read off a binning of the memory, each with the label used when
# the plot config names none. One figure per (statistic, bin size).
BINNED_STATISTICS = {
    "sa_%": "Coverage",
    "sa_h": "Entropy",
}

# The statistic estimated from the observations themselves, which has no bin size
# to vary and so is drawn once.
STATISTICS = {
    "sa_h_knn": "Entropy (kNN)",
}

# A flat bin index is an int64, so a table larger than this cannot be indexed at
# all. Left well below the type's maximum, since the index is built by
# multiplying the per-dimension indices out.
MAX_TABLE_SIZE = 2.0 ** 62

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--folder",
    default="data_dir",
    metavar="DIR",
    help="A data directory, or a single seed directory "
         "(<data_dir>/<config_id>/<rng_seed>) to read one run.",
)
parser.add_argument(
    "-p", "--plot_config",
    required=True,
    metavar="NAME",
    help=f"Plot config: which configurations become the bars, over which "
         f"environments, and what to call them. A name without .yaml is looked for "
         f"in {PLOT_CONFIG_DIR}, or give a path; a directory is a set of configs "
         f"and every one of them is drawn.",
)
parser.add_argument(
    "-o", "--output",
    default="plots",
    help="Output root, under the data directory the runs came from. Figures go in "
         "<root>/knn_entropy/<plot config>, so one data directory holds the output "
         "of every plotting script without one overwriting another.",
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
    "--rng_seeds",
    type=int,
    nargs="+",
    default=None,
    metavar="N",
    help="Which seeds of each configuration to read. Default: every seed that "
         "saved a memory.",
)
parser.add_argument(
    "--bins",
    type=int,
    nargs="+",
    default=[40],
    metavar="N",
    help="Bins per observation dimension the coverage and the binned entropy are "
         "read at, one FIGURE per value: the memory is binned again at each of "
         "them, so a run's coverage can be read at a resolution it was never "
         "launched with. Default: 40.",
)
parser.add_argument(
    "--knn_k",
    type=int,
    default=10,
    metavar="K",
    help="Neighbours the entropy is estimated from. Small k is low bias and high "
         "variance. Default: 10.",
)
parser.add_argument(
    "--discrete_cols",
    type=int,
    nargs="+",
    default=[],
    metavar="J",
    help="Indices of the observation columns to condition on, for a column that "
         "takes a handful of values and so carries a discrete entropy of its "
         "own. The entropy is then H(b) + sum_b p(b) H(s_cont | b) over their "
         "joint values and the action's, estimated over the columns left. "
         "Default: every column goes to the estimator, which reports -inf where "
         "enough observations coincide exactly.",
)
parser.add_argument(
    "--bar_width",
    type=float,
    default=0.85,
    metavar="W",
    help="Bar width as a fraction of the slot it sits in.",
)
parser.add_argument(
    "--no_legend",
    action="store_true",
    help="Leave the legend off the figures.",
)
parser.add_argument(
    "-v", "--verbose",
    action="store_true",
    help="Print progress messages. The recap is always printed.",
)
args = parser.parse_args()

# Taken as written, in ascending order and deduplicated: each distinct bin size
# is one figure.
BINS = sorted({int(b) for b in args.bins})
if not BINS or BINS[0] < 1:
    raise SystemExit(f"--bins takes positive bin counts, got {args.bins}")

ensure_font()
sns.set_context("paper")
sns.set_style("darkgrid", {"legend.frameon": True})
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.axisbelow"] = False
plt.rcParams["grid.linestyle"] = "--"


def vprint(*a, **kw):
    """Print only under --verbose."""

    if args.verbose:
        print(*a, **kw)


# --- Binning ------------------------------------------------------------------
# A counter owns the binning, and it is where the knowledge of how an environment
# encodes its states lives: it converts an observation into the frame it bins in,
# holds the bounds and the bins per dimension, and turns an observation into a
# flat index.

class _NoTable:
    """Mixin that gives a counter its binning with an empty table.

    A counter's table holds one cell per (bin, action) pair, which at the bin
    counts swept here is orders of magnitude larger than the memory being binned.
    The bin edges and the flat index are all this script reads of a counter, so
    the allocation is skipped and the bin count is free to grow past what fits.
    """

    def reset(self, seed=None):
        self._table = None


def env_family(env):
    """The environment's name without its version, e.g. `LunarLander-v3` ->
    `LunarLander`, which is what a counter is named after."""

    spec = getattr(getattr(env, "unwrapped", env), "spec", None)
    env_id = str(getattr(spec, "id", "") or "")
    return re.sub(r"-v\d+$", "", env_id.split("/")[-1])


def counter_class(env):
    """Return the counter class to bin `env` with, or None when it has no binning.

    The class is the one `tabular_count` in src/pseudocount.py builds for the
    environment, and it is returned when that counter is a `BinnedCount`. Any
    other counter reads a discrete observation, which has one binning available
    to it, so its bin size is left out of the sweep.
    """

    # One bin per dimension: only the class of the counter is read here, and it
    # is the same class at any bin count, while `tabular_count` allocates a table
    # that at the bin counts swept below is orders of magnitude larger than the
    # memory being binned (see _NoTable).
    counter = pseudocount.tabular_count(env, n_bins=1)
    if not isinstance(counter, BinnedCount):
        return None
    return type(counter)


def make_counter(env, cls, n_bins):
    """An instance of `cls` binning `env` at `n_bins` bins per dimension.

    A counter decides for itself how many bins a dimension gets: `n_bins` is what
    a continuous dimension is given, and a dimension that takes two values gets
    two bins however large `n_bins` is.
    """

    lean = type(f"Lean{cls.__name__}", (_NoTable, cls), {})
    return lean(
        env.observation_space.low,
        env.observation_space.high,
        n_bins,
        int(env.action_space.n),
    )


def stats_from_nonzero(counts, size):
    """Coverage and normalized entropy of a histogram, from its occupied bins
    alone, as (coverage, entropy).

    `counts` holds the visits of every bin visited at least once and `size` is how
    many bins there are in all. This is what `exploration_stats` in
    src/experiment.py computes, written for a table too large to hold: an empty
    bin contributes zero to both numbers.
    """

    total = counts.sum()
    if total == 0 or size <= 0:
        return 0.0, 0.0
    p = counts.astype(np.float64) / total
    entropy = float(-(p * np.log(p)).sum() / np.log(size)) if size > 1 else 0.0
    return float(counts.size / size), entropy


def binned_stats(env, cls, obs, act):
    """The binned statistics of a whole memory, as {n_bins: {statistic: value}}.

    One binning of the memory per bin size, and the (bin, action) pairs it
    visited counted once: `np.unique` gives the counts of the occupied pairs
    directly, which is what lets a bincount stand in for a table too large to
    hold.

    The bin sizes kept are the ones whose flat index fits an int64, which is what
    a figure is written for.
    """

    n_actions = int(env.action_space.n)
    per_bins = {}
    for n_bins in BINS:
        counter = make_counter(env, cls, n_bins)
        size_s = float(np.prod(counter.n_bins, dtype=np.float64))
        if size_s * n_actions > MAX_TABLE_SIZE:
            continue
        state = counter.bin_index(obs).astype(np.int64, copy=False)
        _, counts = np.unique(state * n_actions + act, return_counts=True)
        coverage, entropy = stats_from_nonzero(counts, size_s * n_actions)
        per_bins[n_bins] = {"sa_%": coverage, "sa_h": entropy}
    return per_bins


_env_cache: dict = {}


def env_of_cfg(env_cfg: dict):
    """The environment a saved run was launched with, and the counter class it is
    binned with, as (env, counter class).

    `env_cfg` is the `environment` section of that run's config, as written to
    cfg.yaml. Creation is often 0.5-3s and dominates when many runs share one
    environment, so identical configs are only built once.
    """

    from src.wrappers import gym_wrappers

    key = yaml.safe_dump(env_cfg, sort_keys=True, default_flow_style=True)
    if key not in _env_cache:
        env = gym_wrappers.make_gym_env(**env_cfg)
        _env_cache[key] = (env, counter_class(env))
    return _env_cache[key]


# --- kNN entropy --------------------------------------------------------------

def knn_entropy(x, k=10):
    """Estimate the differential entropy of samples x with the Kozachenko-Leonenko k-NN estimator and return it in nats."""

    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n, d = x.shape
    dist, _ = cKDTree(x).query(x, k=k + 1)
    eps = dist[:, -1]
    log_vd = (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1)
    return digamma(n) - digamma(k) + log_vd + d * np.mean(np.log(eps))


def memory_entropy(obs, act, k, discrete_cols):
    """The joint kNN entropy H(s, a) of a whole memory, as (nats, dropped mass).

    It is H(b) + sum_b p(b) H(s_cont | b): b ranges over the joint values of the
    action and of the columns `discrete_cols` names, and s_cont over the
    remaining columns, which `knn_entropy` estimates over. The sum runs over the
    conditioning values holding more than k observations, the count a k-th
    neighbour needs.

    The dropped mass is the share of the memory in the values passed over. They
    keep their p(b) log p(b) in H(b), so the entropy is short by
    sum p(b) H(s_cont | b) over them, a quantity of either sign.
    """

    obs = np.asarray(obs, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs[:, None]
    n, d = obs.shape

    discrete_idx = np.asarray(discrete_cols, dtype=np.int64)
    if discrete_idx.size and (discrete_idx.min() < 0 or discrete_idx.max() >= d):
        raise SystemExit(
            f"--discrete_cols {list(discrete_cols)} names a column this "
            f"observation does not have: it has {d}, so the indices run 0 to "
            f"{d - 1}."
        )
    cont_idx = np.setdiff1d(np.arange(d), discrete_idx)
    disc = obs[:, discrete_idx]
    cont = obs[:, cont_idx]

    def labels(*cols):
        cols = [np.asarray(c).reshape(n, -1) for c in cols if c is not None]
        cols = [c for c in cols if c.shape[1] > 0]
        if not cols:
            return np.zeros(n, dtype=np.int64)
        _, inverse = np.unique(np.column_stack(cols), axis=0, return_inverse=True)
        return inverse.ravel()

    def entropy(labels):
        values, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        h = float(-np.sum(p * np.log(p)))
        if cont.shape[1] == 0:
            return h, 0.0
        dropped = 0.0
        for value, pv, count in zip(values, p, counts):
            if count > k:
                h += pv * float(knn_entropy(cont[labels == value], k=k))
            else:
                dropped += float(pv)
        return h, dropped

    return entropy(labels(disc, np.asarray(act).ravel()))


# --- Resolve which runs to read -----------------------------------------------

def resolve_runs(path):
    """Walk what `-f` was given and return (seed directories to read,
    configurations that had none).

    A seed directory is one holding a memory.npz, and `path` is taken to be one
    when it holds it -- that is how a single run is read. Anything else is read as
    a data directory, and each configuration contributes the seeds --rng_seeds
    asks for, or every seed that saved a memory.
    """

    if (path / "memory.npz").is_file():
        return [path], []

    # A cfg.yaml one level up makes this a run directory that saved no memory,
    # which the message below says outright.
    if (path.parent / "cfg.yaml").is_file():
        contents = ", ".join(sorted(p.name for p in path.iterdir())) or "nothing"
        raise SystemExit(
            f"{path} is a run directory but holds no memory.npz, so there is "
            f"nothing to read. It holds: {contents}. memory.npz is only written "
            f"when results.save_memory=True."
        )

    found, without = [], []
    for cfg_dir in sorted(p for p in path.iterdir() if p.is_dir()):
        seed_dirs = sorted(
            (p for p in cfg_dir.iterdir() if (p / "memory.npz").is_file()),
            key=lambda p: (len(p.name), p.name),
        )
        if args.rng_seeds is not None:
            wanted = {str(s) for s in args.rng_seeds}
            seed_dirs = [p for p in seed_dirs if p.name in wanted]
        if seed_dirs:
            found.extend(seed_dirs)
        elif (cfg_dir / "cfg.yaml").is_file():
            without.append(cfg_dir.name)
    return found, without


root = Path(args.folder)
if not root.is_dir():
    raise SystemExit(f"Not a directory: {root}")
run_dirs, no_memory_dirs = resolve_runs(root)

if not run_dirs:
    raise SystemExit(
        f"No memory.npz under {root} ({len(no_memory_dirs)} configuration(s) had "
        f"none). memory.npz is only written when results.save_memory=True."
    )

# A seed directory sits at <data_dir>/<config_id>/<rng_seed>, so the data
# directory is two levels up.
output_root = run_dirs[0].parent.parent / args.output / "knn_entropy"

# --- Read each run's configuration ---
flat_cfgs:   dict = {}   # "<config_id>/<seed>" -> flat cfg
run_cfgs:    dict = {}   # "<config_id>/<seed>" -> nested cfg
mem_files:   dict = {}   # "<config_id>/<seed>" -> path to memory.npz
no_cfg_dirs: list = []

for seed_dir in run_dirs:
    # cfg.yaml lives one level up, shared by every seed of the configuration.
    cfg_file = seed_dir.parent / "cfg.yaml"
    if not cfg_file.is_file():
        no_cfg_dirs.append(seed_dir.parent.name)
        continue

    run_cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    name = f"{seed_dir.parent.name}/{seed_dir.name}"
    # Interpolations kept, as `composed_config` keeps them: the two configs are
    # compared key by key below, and a key dropped from one side and not the
    # other is a key the comparison reads as missing from the run.
    flat_cfgs[name] = flatten_cfg(run_cfg, drop_interpolations=False)
    run_cfgs[name] = run_cfg
    mem_files[name] = seed_dir / "memory.npz"

if not flat_cfgs:
    raise SystemExit(
        f"Found {len(run_dirs)} memory.npz, but no cfg.yaml in the directory above "
        f"any of them. -f takes a data directory or a "
        f"<data_dir>/<config_id>/<rng_seed>."
    )

df = pd.DataFrame.from_dict(flat_cfgs, orient="index")
df.columns = pd.MultiIndex.from_tuples([("hyperparameter", c) for c in df.columns])

plot_cfgs = load_plot_configs(args.plot_config)
vprint(f"\n{len(plot_cfgs)} plot config(s): {[n for n, _, _ in plot_cfgs]}")

# --- Identify environment and algorithm ---
# A run records the keys, and its environment and algorithm are matched back from
# them: each is the config YAML every key the run recorded agrees with. A run
# agreeing with none is "unknown" there.
env_configs = load_config_group(args.environments)
algo_configs = load_config_group(args.algorithms)

# The answers `assign_groups` has already given, by the exemptions that produced
# them: configs declaring the same ones -- usually every one of them -- get the
# same answer, and the frame is compared against every YAML to arrive at it.
_assigned: dict = {}


def runs_of(env, entry, ignored, matched):
    """The runs of one environment that an entry selects: the ones whose whole
    recorded configuration is the one the entry composes to there.

    The comparison covers every key, the ones the entry leaves out included: a
    run launched with another value for one of those is a different
    configuration, and belongs to whichever entry composes to it.

    `ignored` and `matched` are the pair `assign_groups` returned for this plot
    config, and have to be that pair: a mask built under one set of exemptions
    says nothing about a comparison made under another.
    """

    flat = entry_config(env, entry["selector"], entry["label"])
    if flat is None:
        return []
    return list(df.index[composed_mask(df, flat, ignore=ignored) & matched])


# --- Which runs the plot configs ask for ---
# Decided before any memory is read, and read once for all of them: a memory.npz
# costs a decompression whether it ends up in a figure or not.
# config name -> (entries, {group: envs}, {(entry, env): runs}, environments).
# The environments are its own: which YAML a run matches depends on the
# exemptions, and those are the config's.
layout: dict = {}
for config_name, _cfg_path, plot_cfg in plot_cfgs:
    entries = style_entries(plot_cfg)
    if not entries:
        print(f"WARNING: {config_name} declares no configurations to plot, skipping.")
        continue

    # This config's own exemptions, not every config's: exempting a key changes
    # which YAML a run matches, and one config's `ignored_cfg_keys` says nothing
    # about the runs another one draws.
    ignored_keys = ignored_cfg_keys(plot_cfg)
    if ignored_keys:
        vprint(f"{config_name} exempts from config matching: {sorted(ignored_keys)}")
    # Both groups are required: a bar is one configuration, which an entry names by
    # algorithm as well as environment, and each label comes from that group's
    # YAMLs. A group with no YAMLs leaves every run "unknown" there.
    #
    # The environment label alone decides which runs are drawable; the algorithm
    # label is assigned for the figures to read. An entry here is a whole composed
    # configuration, so the comparison `runs_of` makes already holds every key an
    # algorithm YAML declares, at the value the configuration composes to. A YAML
    # declaring a default a sweep overrode agrees with the runs it launched on
    # every key but that one, which leaves them "unknown" there.
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
    # The runs no environment YAML names, reported here rather than in the recap:
    # which runs those are depends on the exemptions, and those are known here.
    # The environment alone is reported, being the label that decides this.
    blocked = [n for n in df.index if df.loc[n, ENV] == "unknown"]
    if blocked:
        print(
            f"\n{len(blocked)} run(s) match no environment YAML under "
            f"{config_name} and are not drawn:"
        )
        for name in blocked:
            print(f"  {name}")
    # The stems of the matched runs alone, so a config that declares no
    # environment group of its own gets one group per environment a YAML named.
    envs = sorted(set(env_stem[matched]), key=str)
    groups = env_groups(plot_cfg) or {"": envs}
    selected = {
        (li, env): runs_of(env, entry, ignored_keys, matched)
        for li, entry in enumerate(entries)
        for group_envs in groups.values() for env in group_envs if env in envs
    }
    layout[config_name] = (
        entries,
        groups,
        {k: n for k, n in selected.items() if n},
        envs,
    )

needed = sorted({
    n for _e, _g, sel, _v in layout.values()
    for names in sel.values() for n in names
})
if not needed:
    raise SystemExit("No run matches any configuration of the plot config(s).")
vprint(f"{len(needed)} run(s) to read, of {len(df)} found")

# Which environment each run is read under, for the progress bar below. Taken
# from the layout, so a run is named here by a config that selected it; the
# frame's own labels are whichever config `assign_groups` was last called for.
env_of_run = {
    name: env
    for _entries, _groups, sel, _envs in layout.values()
    for (_li, env), names in sel.items() for name in names
}

# --- Load each memory, bin it and estimate its entropy ------------------------
collected:       dict = {}   # name -> {n_bins: {statistic: value}}
collected_knn:   dict = {}   # name -> {statistic: value}
no_counter_dirs: list = []
non_finite_dirs: list = []
corrupt_dirs:    list = []
# name -> the share of that run's memory the kNN sum passed over, where it is
# above zero. What it costs the estimate is in `memory_entropy`.
dropped_of:      dict = {}

with tqdm(needed, desc="Reading memory", unit="run", disable=not args.verbose) as pbar:
    for name in pbar:
        pbar.set_postfix_str(str(env_of_run.get(name, "")))

        try:
            env, cls = env_of_cfg(run_cfgs[name].get("environment") or {})
        except Exception as e:
            vprint(f"  WARNING: could not build the environment of {name}: {e}")
            corrupt_dirs.append(name)
            continue

        try:
            with np.load(mem_files[name]) as data:
                obs = data["obs"]
                # Flat, whatever shape the buffer stored: an (n, 1) column would
                # broadcast the flat bin index in `binned_stats` into an (n, n)
                # array.
                act = data["act"].astype(np.int64, copy=False).ravel()
        except Exception as e:
            vprint(f"  WARNING: could not read {mem_files[name]}: {e}")
            corrupt_dirs.append(name)
            continue

        # The kNN entropy is estimated for every environment, reading the
        # observations themselves; the binned statistics need a counter.
        if cls is None:
            no_counter_dirs.append((name, env_of_run.get(name, "")))
        else:
            collected[name] = binned_stats(env, cls, obs, act)
        estimate, dropped = memory_entropy(
            obs,
            act,
            k=args.knn_k,
            discrete_cols=args.discrete_cols,
        )
        if dropped > 0.0:
            dropped_of[name] = dropped
        # -inf comes of observations that coincide exactly in the estimated
        # columns, which puts a zero k-th neighbour distance into log(eps). The
        # figures leave such a run out, so it is named here.
        if np.isfinite(estimate):
            collected_knn[name] = {"sa_h_knn": estimate}
        else:
            non_finite_dirs.append((name, estimate))
        del obs, act

# --- Recap --------------------------------------------------------------------
# The runs matching no YAML are reported per plot config, above, where the
# exemptions that decide the matching are known.
print(f"Runs found : {len(run_dirs)}")
print(
    f"Runs read  : {len(collected_knn)}"
    f"  ({len(no_cfg_dirs)} no cfg,"
    f" {len(no_memory_dirs)} configs with no memory.npz,"
    f" {len(no_counter_dirs)} no binned counter,"
    f" {len(corrupt_dirs)} unreadable)"
)
print(
    f"Estimator  : k={args.knn_k}, whole memory, no subsampling, "
    f"conditioned on {list(args.discrete_cols) or 'no column'} and the action"
)
# How much of each memory the kNN sum passed over. What it costs the estimate is
# in `memory_entropy`.
if dropped_of:
    worst = max(dropped_of, key=dropped_of.get)
    print(
        f"Dropped    : {len(dropped_of)} of {len(collected_knn)} run(s) hold "
        f"conditioning values of {args.knn_k} observations or fewer, up to "
        f"{dropped_of[worst]:.3g} of a memory ({worst})"
    )
    for name in sorted(dropped_of, key=dropped_of.get, reverse=True):
        vprint(f"  {name}: {dropped_of[name]:.3g}")
else:
    print(
        f"Dropped    : none, every conditioning value holds more than "
        f"{args.knn_k} observations"
    )
for _key, (env, cls) in _env_cache.items():
    if cls is None:
        how = "no binned counter"
    else:
        n_bins = [int(b) for b in make_counter(env, cls, BINS[0]).n_bins]
        how = f"{cls.__name__}, {n_bins} at {BINS[0]} bins per dimension"
    print(f"Binning    : {env_family(env)}  ->  {how}")
if no_counter_dirs:
    print("\nNo binned statistics (the environment has no binned counter, so no "
          "bin size to vary):")
    for d, en in no_counter_dirs:
        print(f"  {d}  →  {en}")
if non_finite_dirs:
    print("\nNo kNN entropy (observations coincide exactly in the estimated "
          "columns, so the k-th neighbour sits at distance zero). "
          "--discrete_cols may be naming too few columns:")
    for d, value in non_finite_dirs:
        print(f"  {d}: {value}")
if corrupt_dirs:
    print("\nSkipped (unreadable):")
    for d in corrupt_dirs:
        print(f"  {d}")

if not collected_knn:
    raise SystemExit("Nothing to plot.")

# --- Figures ------------------------------------------------------------------
# A `-p` naming a directory is a set of configs that belong together -- one
# sweep's -- and every one of them is drawn, into its own directory. Which runs
# fill which bar was settled before the memory was read (see `layout`).
written = 0
for config_name, _cfg_path, plot_cfg in plot_cfgs:
    if config_name not in layout:
        continue
    vprint(f"\n{'#' * 60}\nPlot config: {config_name}\n{'#' * 60}")
    stat_labels, env_labels, ylims = plot_labels(plot_cfg)
    entries, groups, selected, envs = layout[config_name]

    # A plot config's `statistics` names what to draw and what to call it; one
    # naming none draws them all. A name that is none of them belongs to another
    # script and is left to it.
    binned_stat_labels = {
        k: v for k, v in stat_labels.items() if k in BINNED_STATISTICS
    } or BINNED_STATISTICS
    stats = {k: v for k, v in stat_labels.items() if k in STATISTICS} or STATISTICS

    output_dir = output_root / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    def bars_of(env, stat, n_bins):
        """The value of one environment and statistic per entry, as
        [(entry, (mean, half-width) or None)] and the seeds each was read from.

        `n_bins` of None reads the kNN entropy, which stands on its own for every
        binning. Every entry keeps its slot, with None where its runs hold no
        finite value, so the same configuration is the same bar in every subplot.
        """

        held = collected if n_bins is not None else collected_knn
        items, alive_of = [], {}
        for li, entry in enumerate(entries):
            values = []
            for name in selected.get((li, env), []):
                run = held.get(name)
                if run is not None and n_bins is not None:
                    run = run.get(n_bins)
                if run is not None and stat in run:
                    values.append(run[stat])
            values = [v for v in values if np.isfinite(v)]
            if not values:
                items.append((entry, None))
                continue
            # ci_bounds reduces seeds along axis 0, so one column of seeds gives
            # the mean and the half-width of this one bar.
            mean, half, alive = ci_bounds(np.array(values, dtype=float)[:, None])
            items.append((entry, (float(mean[0]), float(half[0]))))
            alive_of[li] = int(alive[0])
        return items, alive_of

    def draw_entropy(stat, ylabel, group_name, drawn_envs, n_bins=None):
        """One figure: a subplot per environment, a bar per configuration, and a
        95% confidence interval over the seeds."""

        fig, axs = plt.subplots(
            1,
            len(drawn_envs),
            figsize=(SUBPLOT_W * len(drawn_envs), SUBPLOT_H),
            squeeze=False,
        )
        fig.subplots_adjust(wspace=0.3)
        handles = []
        drew = False

        for col, env in enumerate(drawn_envs):
            ax = axs[0][col]
            title = env_labels.get(env, env)
            ax.set_title(title, fontsize=FONT_SIZE, **tex_kwargs(title))

            items, _alive = bars_of(env, stat, n_bins)
            if not any(v is not None for _e, v in items):
                continue
            # draw_bars names the configurations by colour and hatch, which the
            # legend spells out once for the figure, and stands every bar from
            # one floor below the lowest interval.
            draw_bars(
                ax,
                items,
                ylabel=ylabel if col == 0 else "",
                bar_width=args.bar_width,
                font_size=FONT_SIZE,
            )
            drew = True
            # From the first subplot that drew anything, not from the first
            # subplot: an environment holding none of the configurations would
            # otherwise leave the figure without a legend.
            if not handles:
                handles = [
                    mpatches.Patch(
                        facecolor=entry["color"],
                        hatch=entry["hatch"],
                        edgecolor="black",
                        linewidth=0.6,
                        label=entry["label"],
                    )
                    for entry, value in items if value is not None
                ]

            if (env, stat) in ylims:
                ax.set_ylim(*ylims[(env, stat)])
            set_3_ticks(ax, "y")

        if not drew:
            plt.close(fig)
            return 0

        if handles and not args.no_legend:
            legend = fig.legend(
                handles,
                [h.get_label() for h in handles],
                fontsize=FONT_SIZE - 2,
                frameon=True,
                loc="lower center",
                ncol=len(handles),
                bbox_to_anchor=(0.5, -0.22),
            )
            for text in legend.get_texts():
                text.set(**tex_kwargs(text.get_text()))

        # The bin size is in the name, so each binning keeps its own file.
        parts = [p for p in (group_name, stat) if p]
        if n_bins is not None:
            parts.append(f"{n_bins}bins")
        vprint(f"  Saved: {save_figure(fig, output_dir, '_'.join(parts))}")
        plt.close(fig)
        return 1

    for group_name, group_envs in groups.items():
        drawn_envs = [e for e in group_envs if e in envs]
        if not drawn_envs:
            continue

        # The kNN entropy once, being one estimate over the observations, and the
        # binned statistics once per bin size.
        drawn_stats = [(stat, label, None) for stat, label in stats.items()]
        drawn_stats += [
            (stat, label, n_bins)
            for n_bins in BINS
            for stat, label in binned_stat_labels.items()
        ]
        for stat, ylabel, n_bins in drawn_stats:
            written += draw_entropy(stat, ylabel, group_name, drawn_envs, n_bins)

        # The console carries the numbers to the precision the seeds justify,
        # which a bar standing from a floor reads off its axis only roughly.
        for stat, _label, n_bins in drawn_stats:
            rows = []
            for env in drawn_envs:
                items, alive_of = bars_of(env, stat, n_bins)
                rows.extend(
                    (env, entry["label"], value[0], value[1], alive_of[li])
                    for li, (entry, value) in enumerate(items) if value is not None
                )
            if not rows:
                continue
            prec = detect_precision(
                [mean for _env, _label, mean, _half, _alive in rows],
                min_prec=2,
                max_prec=6,
            )
            at_bins = f" at {n_bins} bins" if n_bins is not None else ""
            print(
                f"\n{stat}{at_bins}"
                f"{f'  [{group_name}]' if group_name else ''}:"
            )
            for env, label, mean, half, alive in rows:
                print(
                    f"  {env:<20} {label:<28} "
                    f"{mean:.{prec}f} ± {half:.{prec}f}  ({alive} seed(s))"
                )

print(f"\n{written} figure(s) written to {output_root}")
