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
in, and --bins writes a figure per bin size. Which counter an environment gets is
the one `tabular_count` builds for it, or the one a `counter` key names beside
that environment in the plot config. An environment whose counter bins nothing
gets the kNN entropy alone.

The third is estimated from the observations themselves (does not depend on bin
size).

--discrete_cols can be used to filter out discrete observations, making them
part of the discrete action space.

Output goes to <data_dir>/<output>/knn_entropy/<plot config>/.

Example

    python plot_knn_entropy.py -f data_gcrl_lunar_full -p gcrl_lunar --discrete_cols 6 7 --bins 10 20 30 40 50 60 70 80 90 100 101 -v
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
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
    GROUP_GAP_BAR,
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
    rows_cols,
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
# Height one configuration takes under --horizontal_bars, in inches. The bar
# takes --bar_width of it and the rest is the gap to the next, so this sets how
# thick a bar is and --bar_width how much air stands around it.
HBAR_UNIT_INCHES = 0.18
# Room above the bars for the title and below them for the value axis, in
# inches. The figure is these plus the bars, so a row is HBAR_UNIT_INCHES tall
# whatever the configuration count; a floor on the whole figure would instead
# stretch the rows to fill it.
HBAR_TOP_INCHES = 0.30
HBAR_BOTTOM_INCHES = 0.55

# A value axis whose ticks all sit below this is labelled in scientific
# notation. `set_3_ticks` writes at most four decimals on x, so a tick whose
# first significant digit falls past the fourth reads as 0.0000.
SCIENTIFIC_BELOW = 1e-4
# Widths of the lines a bar is drawn with, in points: its outline, which the
# legend's patches carry too, and its error bar.
BAR_EDGE_WIDTH = 0.25
ERROR_BAR_WIDTH = 0.4

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

# Where --save writes and --load reads, under the output directory.
STATS_FILE = "statistics.gzip"

# The bin size a row of that file carries for a statistic read off no binning.
NO_BINS = -1

# The columns of a row, in order. The settings come with the value: a statistic
# computed at another counter, --knn_k or --discrete_cols is another quantity,
# and --load compares them before it takes a row.
STATS_COLUMNS = (
    "config_id",
    "rng_seed",
    "statistic",
    "n_bins",
    "value",
    "counter",
    "knn_k",
    "discrete_cols",
    "dropped",
)

# The settings a row must agree with for --load to take it. The binned
# statistics are decided by the counter and the bin size, which is a column of
# its own; the kNN entropy reads no counter.
BINNED_SETTINGS = ("counter",)
KNN_SETTINGS = ("knn_k", "discrete_cols")

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--folder",
    default="data_dir",
    metavar="DIR",
    help=(
        "A data directory, or a single seed directory "
        "(<data_dir>/<config_id>/<rng_seed>) to read one run."
    ),
)
parser.add_argument(
    "-p",
    "--plot_config",
    required=True,
    metavar="NAME",
    help=(
        f"Plot config: which configurations become the bars, over which "
        f"environments, and what to call them. A name without .yaml is looked for "
        f"in {PLOT_CONFIG_DIR}, or give a path; a directory is a set of configs "
        f"and every one of them is drawn."
    ),
)
parser.add_argument(
    "-o",
    "--output",
    default="plots",
    help=(
        "Output root, under the data directory the runs came from. Figures go in "
        "<root>/knn_entropy/<plot config>, so one data directory holds the output "
        "of every plotting script without one overwriting another."
    ),
)
parser.add_argument(
    "-a",
    "--algorithms",
    default="configs/algorithm",
    help="Directory of algorithm YAMLs.",
)
parser.add_argument(
    "-e",
    "--environments",
    default="configs/environment",
    help="Directory of environment YAMLs.",
)
parser.add_argument(
    "--rng_seeds",
    type=int,
    nargs="+",
    default=None,
    metavar="N",
    help=(
        "Which seeds of each configuration to read. Default: every seed that "
        "saved a memory."
    ),
)
parser.add_argument(
    "--bins",
    type=int,
    nargs="+",
    default=[40],
    metavar="N",
    help=(
        "Bins per observation dimension the coverage and the binned entropy are "
        "read at, one FIGURE per value: the memory is binned again at each of "
        "them, so a run's coverage can be read at a resolution it was never "
        "launched with. Default: 40."
    ),
)
parser.add_argument(
    "--knn_k",
    type=int,
    default=10,
    metavar="K",
    help=(
        "Neighbours the entropy is estimated from. Small k is low bias and high "
        "variance. Default: 10."
    ),
)
parser.add_argument(
    "--discrete_cols",
    type=int,
    nargs="+",
    default=[],
    metavar="J",
    help=(
        "Indices of the observation columns to condition on, for a column that "
        "takes a handful of values and so carries a discrete entropy of its "
        "own. The entropy is then H(b) + sum_b p(b) H(s_cont | b) over their "
        "joint values and the action's, estimated over the columns left. "
        "Default: every column goes to the estimator, which reports -inf where "
        "enough observations coincide exactly."
    ),
)
parser.add_argument(
    "--bar_width",
    type=float,
    default=0.95,
    metavar="W",
    help="Bar width as a fraction of the slot it sits in.",
)
parser.add_argument(
    "--horizontal_bars",
    action="store_true",
    help=(
        "Lay the bars along the x axis, one configuration per row, named on the "
        "y axis. The subplot grows with the number of configurations."
    ),
)
parser.add_argument(
    "--save",
    action="store_true",
    help=(
        f"Write the statistics to {STATS_FILE} under the output directory, one "
        f"row per (run, statistic, bin size) with the settings that produced "
        f"it. A row already in the file is replaced only by the same statistic "
        f"of the same run at the same settings, so one file holds several "
        f"sweeps."
    ),
)
parser.add_argument(
    "--load",
    action="store_true",
    help=(
        f"Read the statistics from {STATS_FILE} under the output directory and "
        f"compute only what it does not hold. A row counts only when it was "
        f"written at the same counter, --knn_k and --discrete_cols; a run "
        f"whose every statistic is there is not read from memory.npz at all."
    ),
)
parser.add_argument(
    "--no_legend",
    action="store_true",
    help="Leave the legend off the figures. The standalone legend is still "
         "written.",
)
parser.add_argument(
    "-v",
    "--verbose",
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

    A counter's table holds one cell per (bin, action) pair, which at large bin
    counts can be orders of magnitude larger than the observations being binned.
    The allocation is skipped, so only the bin edges and the flat index are
    usable, and the bin count is free to grow past what a table would fit in.
    """

    def reset(self, seed=None):
        self._table = None


def env_family(env):
    """Return the environment's name without its version, for example
    `LunarLander-v3` -> `LunarLander`."""

    spec = getattr(getattr(env, "unwrapped", env), "spec", None)
    env_id = str(getattr(spec, "id", "") or "")
    return re.sub(r"-v\d+$", "", env_id.split("/")[-1])


def draw_hbars(ax, items, label, bar_width, font_size=FONT_SIZE, names=False):
    """Draw one horizontal bar per (entry, summary) of `items` on `ax`, stacked
    top to bottom in the order given.

    The same floor, group gaps, colours, hatches and error bars as `draw_bars`,
    with the value on the x axis. A summary of None keeps its slot empty, so the
    same entry is in the same row wherever the same `items` order is drawn.

    `names` writes each entry's label beside its row.
    """

    drawn = [entry for entry, _ in items]
    # A slot of one, so `bar_width` is the share of it the bar takes and what is
    # left is the gap. A slot that grew with the bar would leave it filling the
    # same share of its row whatever `bar_width` said.
    slot = 1.0
    ys = [0.0]
    for prev, cur in zip(drawn, drawn[1:]):
        gap = GROUP_GAP_BAR if cur["group"] != prev["group"] else 0.0
        ys.append(ys[-1] + slot + gap)

    lows = [m - e for _, v in items if v is not None for m, e in [v] if np.isfinite(m)]
    floor = min(lows) - 0.1 * abs(min(lows)) if lows else 0.0

    for y, (entry, value) in zip(ys, items):
        if value is None:
            continue
        mean, err = value
        ax.barh(
            y,
            mean - floor,
            left=floor,
            height=bar_width,
            xerr=err,
            color=entry["color"],
            hatch=entry["hatch"],
            edgecolor="black",
            linewidth=BAR_EDGE_WIDTH,
            capsize=2,
            error_kw={"linewidth": ERROR_BAR_WIDTH, "ecolor": "black"},
        )

    if ys:
        # Inverted, so the first configuration is on top.
        ax.set_ylim(ys[-1] + slot / 2, ys[0] - slot / 2)
        ax.set_xlim(left=floor)
    if names:
        ax.set_yticks(ys)
        ax.set_yticklabels([e["label"] for e in drawn], fontsize=font_size - 2)
        for text in ax.get_yticklabels():
            text.set(**tex_kwargs(text.get_text()))
        ax.tick_params(axis="y", length=0, pad=2)
    else:
        ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=font_size - 2, pad=1)
    ax.set_xlabel(label, fontsize=font_size, **tex_kwargs(label))
    # darkgrid would otherwise draw a line through every bar at its tick.
    ax.yaxis.grid(False)


def binned_counters():
    """Collect every binned counter src/pseudocount.py defines and return them as
    {name: class}.

    The generic `BinnedCount` is included under its own name, for an environment
    binned on the bounds of its observation space.
    """

    return {
        name: obj
        for name, obj in vars(pseudocount).items()
        if isinstance(obj, type) and issubclass(obj, BinnedCount)
    }


def counter_class(env, name):
    """Return the counter class to bin `env` with, or None when it has no binning.

    The class is the binned counter `name` names in src/pseudocount.py. When
    `name` is None it is the one `tabular_count` builds for the environment,
    returned when that counter is a `BinnedCount`: any other counter reads a
    discrete observation, which has a single binning, and gives None.
    """

    counters = binned_counters()
    if name is not None:
        if name not in counters:
            raise SystemExit(
                f"Counter {name} is not a binned counter in "
                f"src/pseudocount.py. It defines: {', '.join(sorted(counters))}."
            )
        return counters[name]

    # One bin per dimension: only the class of the counter is read here, and it
    # is the same class at any bin count, while `tabular_count` allocates a table
    # that at the bin counts swept below is orders of magnitude larger than the
    # memory being binned (see _NoTable).
    counter = pseudocount.tabular_count(env, n_bins=1)
    if not isinstance(counter, BinnedCount):
        return None
    return type(counter)


def make_counter(env, cls, n_bins):
    """Build an instance of `cls` binning `env` at `n_bins` bins per dimension
    and return it.

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
    """Compute the coverage and normalized entropy of a histogram from its
    occupied bins alone and return them as (coverage, entropy).

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


def binned_stats(env, cls, obs, act, bins):
    """Bin a whole memory at every size in `bins` and return its binned
    statistics as {n_bins: {statistic: value}}.

    One binning of the memory per bin size, and the (bin, action) pairs it
    visited counted once: `np.unique` gives the counts of the occupied pairs
    directly, which is what lets a bincount stand in for a table too large to
    hold.

    A bin size whose flat index does not fit an int64 gets NaN for every
    statistic.
    """

    n_actions = int(env.action_space.n)
    per_bins = {}
    for n_bins in bins:
        counter = make_counter(env, cls, n_bins)
        size_s = float(np.prod(counter.n_bins, dtype=np.float64))
        if size_s * n_actions > MAX_TABLE_SIZE:
            per_bins[n_bins] = {
                statistic: float("nan") for statistic in BINNED_STATISTICS
            }
            continue
        state = counter.bin_index(obs).astype(np.int64, copy=False)
        _, counts = np.unique(state * n_actions + act, return_counts=True)
        coverage, entropy = stats_from_nonzero(counts, size_s * n_actions)
        per_bins[n_bins] = {"sa_%": coverage, "sa_h": entropy}
    return per_bins


_env_cache: dict = {}
_class_cache: dict = {}


def cfg_key(env_cfg: dict):
    """Return a string identifying `env_cfg`, equal for equal configs."""

    return yaml.safe_dump(env_cfg, sort_keys=True, default_flow_style=True)


def env_of_cfg(env_cfg: dict):
    """Build the environment a saved run was launched with and return it.

    `env_cfg` is the `environment` section of that run's config, as written to
    cfg.yaml. Creation is often 0.5-3s and dominates when many runs share one
    environment, so identical configs are only built once.
    """

    from src.wrappers import gym_wrappers

    key = cfg_key(env_cfg)
    if key not in _env_cache:
        _env_cache[key] = gym_wrappers.make_gym_env(**env_cfg)
    return _env_cache[key]


def class_of_cfg(env_cfg: dict, name):
    """Return the counter class `counter_class` gives the environment of
    `env_cfg` under the counter name `name`, or None.

    Resolved once per (environment config, name).
    """

    key = (cfg_key(env_cfg), name)
    if key not in _class_cache:
        _class_cache[key] = counter_class(env_of_cfg(env_cfg), name)
    return _class_cache[key]


def value_ticks(ax, axis):
    """Put three ticks on the value `axis` of `ax`, in scientific notation where
    a fixed number of decimals would read as zero.

    `set_3_ticks` formats an axis with a fixed number of decimal places, four of
    them at most on x, so a tick whose first significant digit falls past the
    fourth formats as 0.0000. Those axes get a mantissa per tick and one exponent
    for the axis instead. Coverage over a fine binning of a high-dimensional
    space is the case this is here for.
    """

    set_3_ticks(ax, axis)
    which = ax.xaxis if axis == "x" else ax.yaxis
    ticks = [abs(t) for t in which.get_ticklocs() if np.isfinite(t) and t != 0.0]
    if ticks and max(ticks) < SCIENTIFIC_BELOW:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        which.set_major_formatter(formatter)


def write_legend(entries, plot_cfg, output_dir):
    """Write the legend on its own, for a figure that has to carry one
    elsewhere, and return how many were written.

    Both shapes go out every time, since which one fits depends on the document
    the figure goes into rather than on the data. The horizontal is laid out by
    `legend_rows_cols`; the vertical is one entry per row. --no_legend takes the
    legend off the figures and leaves these alone.
    """

    handles = [
        mpatches.Patch(
            facecolor=entry["color"],
            hatch=entry["hatch"],
            edgecolor="black",
            linewidth=BAR_EDGE_WIDTH,
            label=entry["label"],
        )
        for entry in entries
    ]
    if not handles:
        return 0
    _n_rows, n_cols = rows_cols(plot_cfg.get("legend_rows_cols"), len(handles))

    written = 0
    for name, cols in (("legend_horizontal", n_cols), ("legend_vertical", 1)):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis("off")
        legend = ax.legend(
            handles,
            [h.get_label() for h in handles],
            fontsize=FONT_SIZE,
            frameon=True,
            loc="center",
            ncol=cols,
            # In font units. The defaults (2.0 and 0.7) draw a block half the
            # height of the text beside it.
            handlelength=2.6,
            handleheight=1.3,
        )
        for text in legend.get_texts():
            text.set(**tex_kwargs(text.get_text()))
        # The canvas is cut to the legend rather than guessed at. A figure sized
        # by a rule of thumb is a floor the crop cannot go below, and one entry
        # per row in a canvas wide enough for four is all margin.
        fig.canvas.draw()
        box = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.set_size_inches(box.width, box.height)
        vprint(f"  Saved: {save_figure(fig, output_dir, name)}")
        plt.close(fig)
        written += 1
    return written


def wanted_statistics(plot_cfg):
    """Read the statistics a plot config draws and return them as (binned, kNN),
    each {name: label}.

    A plot config's `statistics` names what to draw and what to call it; one
    naming none of this script's statistics draws them all. A name that is none
    of them belongs to another script and is left to it.
    """

    stat_labels = plot_labels(plot_cfg)[0]
    binned = {k: v for k, v in stat_labels.items() if k in BINNED_STATISTICS}
    knn = {k: v for k, v in stat_labels.items() if k in STATISTICS}
    if binned or knn:
        return binned, knn
    return dict(BINNED_STATISTICS), dict(STATISTICS)


def plot_counters(config_name, plot_cfg):
    """Read the counters the environments of a plot config name and return them
    as {environment: counter name}.

    An environment names one with a `counter` key anywhere under `environments`:
    in the mapping that gives its own label and y-axis limits, which is one
    counter for that environment, or beside the `environment: label` entries of a
    block, which is one counter for every environment of it. Which of the two a
    `counter` is comes from what stands beside it: `label` and `ylim` are an
    environment's own keys, and anything else is an environment.

    The key is taken out of the mapping, so the helpers that read the
    environments of a plot config see only environments. `plot_cfg` is therefore
    changed in place and a second call on it finds nothing. A name that is not a
    binned counter in src/pseudocount.py is an error.
    """

    found = {}

    def walk(node, env=None):
        """Walk `node`, which sits under the key `env` when it has one above it.

        A mapping is walked under the key that holds it, so an environment's own
        mapping is always reached under that environment's name however deeply
        the groups above it nest.
        """

        if isinstance(node, list):
            for value in node:
                walk(value, env)
            return
        if not isinstance(node, dict):
            return
        # Taken out of the mapping, so that `env_groups` and `plot_labels` read
        # the environments beside it and nothing else.
        counter = node.pop("counter", None)
        if counter is not None:
            if not isinstance(counter, str):
                raise SystemExit(
                    f"{config_name}: a `counter` names a counter of "
                    f"src/pseudocount.py as a string, and this one is "
                    f"{counter!r}."
                )
            beside = set(node) - {"label", "ylim"}
            if not beside and env is not None:
                found[env] = counter
            else:
                for key in beside:
                    found[key] = counter
        for key, value in node.items():
            walk(value, key)

    walk(plot_cfg.get("environments"))

    counters = binned_counters()
    for env, name in found.items():
        if name not in counters:
            raise SystemExit(
                f"{config_name}: {env} names {name}, which is not a binned "
                f"counter in src/pseudocount.py. It defines: "
                f"{', '.join(sorted(counters))}."
            )
    return found


# --- kNN entropy --------------------------------------------------------------

def knn_entropy(x, k=10):
    """Estimate the differential entropy of samples x with the
    Kozachenko-Leonenko k-NN estimator and return it in nats."""

    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n, d = x.shape
    dist, _ = cKDTree(x).query(x, k=k + 1)
    eps = dist[:, -1]
    log_vd = (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1)
    return digamma(n) - digamma(k) + log_vd + d * np.mean(np.log(eps))


def memory_entropy(obs, act, k, discrete_cols):
    """Estimate the joint kNN entropy H(s, a) of a whole memory and return it as
    (nats, dropped mass).

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

    def entropy(codes):
        values, counts = np.unique(codes, return_counts=True)
        p = counts / counts.sum()
        h = float(-np.sum(p * np.log(p)))
        if cont.shape[1] == 0:
            return h, 0.0
        dropped = 0.0
        for value, pv, count in zip(values, p, counts):
            if count > k:
                h += pv * float(knn_entropy(cont[codes == value], k=k))
            else:
                dropped += float(pv)
        return h, dropped

    return entropy(labels(disc, np.asarray(act).ravel()))


# --- Saved statistics ---------------------------------------------------------
# One row per (run, statistic, bin size, settings), written as the parquet
# process_data.py writes its own frame to. --load takes a row when its settings
# are the ones asked for now, and --save under --load writes back every row it
# read, so a sweep can be extended a bin size at a time.

def settings_now(counter_name):
    """Return the settings a row computed now carries, as {column: value}."""

    return {
        "counter": counter_name,
        "knn_k": int(args.knn_k),
        "discrete_cols": ",".join(str(int(c)) for c in args.discrete_cols),
    }


def settings_of(statistic):
    """Return the columns of `settings_now` a row of `statistic` is compared on."""

    return KNN_SETTINGS if statistic in STATISTICS else BINNED_SETTINGS


def row_key(row):
    """Return the key a row is held under, as (run, statistic, bin size,
    settings).

    The settings are the values of the columns `settings_of` names for the row's
    statistic, as strings, so a value read back from the file and the same value
    computed now give the same key.
    """

    return (
        f"{row['config_id']}/{row['rng_seed']}",
        row["statistic"],
        int(row["n_bins"]),
        tuple(str(row[column]) for column in settings_of(row["statistic"])),
    )


def load_statistics(path):
    """Read the rows of `path` and return them as {row key: row}, keyed by
    `row_key`.

    Every row is returned, whatever its settings. Which of them a run may take
    depends on the counter its environment resolves to, so the settings are
    matched at the lookup, where that is known.
    """

    if not path.is_file():
        print(f"No statistics at {path}.")
        return {}
    frame = pd.read_parquet(path)
    missing = [c for c in STATS_COLUMNS if c not in frame.columns]
    if missing:
        raise SystemExit(
            f"{path} is missing the column(s) {', '.join(missing)}, so it was "
            f"written by another version of this script. Delete it or point "
            f"--output elsewhere."
        )
    rows = {}
    for row in frame.to_dict(orient="records"):
        rows[row_key(row)] = row
    # Read, not taken: --save reads the file to keep the rows it does not
    # recompute, and reuses none of them. What --load took is in the recap.
    print(f"Read {len(rows)} statistic(s) from {path}")
    return rows


def cached_value(held, name, statistic, n_bins, settings):
    """Return the row `held` holds for one (run, statistic, bin size) under
    `settings`, or None.

    A row written at other settings is a different quantity under the same name,
    so it is not returned and the statistic is computed again.
    """

    key = (
        name,
        statistic,
        n_bins,
        tuple(str(settings[column]) for column in settings_of(statistic)),
    )
    return held.get(key)


def statistics_frame(binned, knn, dropped):
    """Build the rows to write and return them as a DataFrame with STATS_COLUMNS
    in order."""

    rows = []
    for name, per_counter in binned.items():
        config_id, _, rng_seed = name.rpartition("/")
        for counter_name, per_bins in per_counter.items():
            for n_bins, stats in per_bins.items():
                for statistic, value in stats.items():
                    rows.append(
                        {
                            "config_id": config_id,
                            "rng_seed": rng_seed,
                            "statistic": statistic,
                            "n_bins": int(n_bins),
                            "value": float(value),
                            "counter": counter_name,
                            "knn_k": int(args.knn_k),
                            "discrete_cols": "",
                            "dropped": float("nan"),
                        }
                    )
    for name, stats in knn.items():
        config_id, _, rng_seed = name.rpartition("/")
        for statistic, value in stats.items():
            rows.append(
                {
                    "config_id": config_id,
                    "rng_seed": rng_seed,
                    "statistic": statistic,
                    "n_bins": NO_BINS,
                    "value": float(value),
                    "counter": "",
                    "knn_k": int(args.knn_k),
                    "discrete_cols": ",".join(str(int(c)) for c in args.discrete_cols),
                    "dropped": float(dropped.get(name, 0.0)),
                }
            )
    return pd.DataFrame(rows, columns=list(STATS_COLUMNS))


# --- Resolve which runs to read -----------------------------------------------

def resolve_runs(path):
    """Walk what `-f` was given and return (seed directories to read,
    configurations that had none, configurations whose memories are all outside
    --rng_seeds).

    A seed directory is one holding a memory.npz, and `path` is taken to be one
    when it holds it -- that is how a single run is read. Anything else is read as
    a data directory, and each configuration contributes the seeds --rng_seeds
    asks for, or every seed that saved a memory.
    """

    if (path / "memory.npz").is_file():
        return [path], [], []

    # A cfg.yaml one level up makes this a run directory that saved no memory,
    # which the message below says outright.
    if (path.parent / "cfg.yaml").is_file():
        contents = ", ".join(sorted(p.name for p in path.iterdir())) or "nothing"
        raise SystemExit(
            f"{path} is a run directory but holds no memory.npz, so there is "
            f"nothing to read. It holds: {contents}. memory.npz is only written "
            f"when results.save_memory=True."
        )

    found, without, filtered = [], [], []
    for cfg_dir in sorted(p for p in path.iterdir() if p.is_dir()):
        seed_dirs = sorted(
            (p for p in cfg_dir.iterdir() if (p / "memory.npz").is_file()),
            key=lambda p: (len(p.name), p.name),
        )
        if args.rng_seeds is not None:
            wanted = {str(s) for s in args.rng_seeds}
            kept = [p for p in seed_dirs if p.name in wanted]
            if seed_dirs and not kept:
                filtered.append(cfg_dir.name)
                continue
            seed_dirs = kept
        if seed_dirs:
            found.extend(seed_dirs)
        elif (cfg_dir / "cfg.yaml").is_file():
            without.append(cfg_dir.name)
    return found, without, filtered


root = Path(args.folder)
if not root.is_dir():
    raise SystemExit(f"Not a directory: {root}")
run_dirs, no_memory_dirs, filtered_dirs = resolve_runs(root)

if not run_dirs:
    raise SystemExit(
        f"No memory.npz under {root} ({len(no_memory_dirs)} configuration(s) had "
        f"none, {len(filtered_dirs)} had none of --rng_seeds). memory.npz is only "
        f"written when results.save_memory=True."
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
    """Return the runs of one environment that an entry selects: the ones whose
    whole recorded configuration is the one the entry composes to there.

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
# config name -> (entries, {group: envs}, {(entry, env): runs}, environments,
# {environment: counter name}).
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
    named_counters = plot_counters(config_name, plot_cfg)
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
    # A counter is named for an environment, and a name that is none of the ones
    # drawn -- a group's, a misspelling, or an environment no run here matches --
    # would otherwise be read by nothing.
    unplaced = sorted(set(named_counters) - set(envs), key=str)
    if unplaced:
        print(
            f"WARNING: {config_name} names a counter for "
            f"{', '.join(unplaced)}, which no run here matches. A `counter` "
            f"belongs beside an environment, not a group."
        )
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
        {env: named_counters.get(env) for env in envs},
    )

needed = sorted(
    {
        n for _e, _g, sel, _v, _c in layout.values()
        for names in sel.values() for n in names
    }
)
if not needed:
    raise SystemExit("No run matches any configuration of the plot config(s).")
vprint(f"{len(needed)} run(s) to read, of {len(df)} found")

# Which environment each run is read under, for the progress bar below. Taken
# from the layout, so a run is named here by a config that selected it; the
# frame's own labels are whichever config `assign_groups` was last called for.
env_of_run = {
    name: env
    for _entries, _groups, sel, _envs, _counter in layout.values()
    for (_li, env), names in sel.items() for name in names
}

counters_of_run: dict = {}
for _entries, _groups, sel, _envs, counter_of_env in layout.values():
    for (_li, env), names in sel.items():
        for name in names:
            counters_of_run.setdefault(name, set()).add(counter_of_env[env])

# --- Load each memory, bin it and estimate its entropy ------------------------
collected:       dict = {}   # name -> {counter class: {n_bins: {statistic: value}}}
collected_knn:   dict = {}   # name -> {statistic: value}
no_counter_dirs: list = []
non_finite_dirs: list = []
corrupt_dirs:    list = []
# name -> the share of that run's memory the kNN sum passed over, where it is
# above zero. What it costs the estimate is in `memory_entropy`.
dropped_of:      dict = {}
# (name, counter name) -> the name of the counter class it resolved to, or None,
# for the figures to find the binned statistics of a run.
resolved:        dict = {}

# What the plot configs draw decides what is computed: a statistic no figure
# asks for is an estimate nothing reads, and the kNN entropy is the expensive
# half of a run.
drawn_binned, drawn_knn = set(), set()
for config_name, _cfg_path, plot_cfg in plot_cfgs:
    if config_name in layout:
        binned, knn = wanted_statistics(plot_cfg)
        drawn_binned.update(binned)
        drawn_knn.update(knn)

# Read under --save as well as --load, so that a save replaces the rows it
# recomputed and no others. A run computes only the statistics its plot configs
# draw, so the rows of a statistic none of them draws are on file and nowhere
# else. Read here, before any memory is, so a file that cannot be read stops the
# script before the work rather than after it.
on_file = (
    load_statistics(output_root / STATS_FILE)
    if args.load or args.save
    else {}
)
held = on_file if args.load else {}
reused = 0
read_runs = 0
filled_runs = 0
idle_runs = 0

with tqdm(needed, desc="Reading memory", unit="run", disable=not args.verbose) as pbar:
    for name in pbar:
        pbar.set_postfix_str(str(env_of_run.get(name, "")))

        env_cfg = run_cfgs[name].get("environment") or {}
        try:
            env = env_of_cfg(env_cfg)
            classes = {
                counter_name: class_of_cfg(env_cfg, counter_name)
                for counter_name in sorted(counters_of_run[name], key=str)
            }
        except Exception as e:
            vprint(f"  WARNING: could not build the environment of {name}: {e}")
            corrupt_dirs.append(name)
            continue

        # The kNN entropy reads the observations themselves, so every
        # environment has one where a plot config draws it; the binned
        # statistics need a counter as well.
        for counter_name, cls in classes.items():
            resolved[(name, counter_name)] = None if cls is None else cls.__name__
        if any(cls is None for cls in classes.values()):
            no_counter_dirs.append((name, env_of_run.get(name, "")))
        to_count = (
            {cls.__name__: cls for cls in classes.values() if cls is not None}
            if drawn_binned
            else {}
        )

        # What the saved rows already hold, so that the memory is read only for
        # what they do not. The counter is known by here and the observations
        # are not, which is what lets a run be filled without decompressing one.
        to_bin = {}
        took_from_file = 0
        for cls_name in to_count:
            settings = settings_now(cls_name)
            per_bins = collected.setdefault(name, {}).setdefault(cls_name, {})
            for n_bins in BINS:
                from_file = {
                    statistic: cached_value(held, name, statistic, n_bins, settings)
                    for statistic in BINNED_STATISTICS
                }
                if all(row is not None for row in from_file.values()):
                    per_bins[n_bins] = {
                        statistic: float(row["value"])
                        for statistic, row in from_file.items()
                    }
                    reused += len(from_file)
                    took_from_file += len(from_file)
            missing = [b for b in BINS if b not in per_bins]
            if missing:
                to_bin[cls_name] = missing
        knn_row = (
            cached_value(held, name, "sa_h_knn", NO_BINS, settings_now(""))
            if drawn_knn
            else None
        )
        if knn_row is not None:
            collected_knn[name] = {"sa_h_knn": float(knn_row["value"])}
            if float(knn_row["dropped"]) > 0.0:
                dropped_of[name] = float(knn_row["dropped"])
            reused += 1
            took_from_file += 1

        needs_knn = bool(drawn_knn) and knn_row is None
        if not to_bin and not needs_knn:
            # Counted only where a row stood in for the memory: a run whose
            # statistics are none of the ones drawn was not filled, it was
            # never wanted.
            if took_from_file:
                filled_runs += 1
            else:
                idle_runs += 1
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
        read_runs += 1

        for cls_name, bins in to_bin.items():
            collected[name][cls_name].update(
                binned_stats(env, to_count[cls_name], obs, act, bins)
            )
        if not needs_knn:
            del obs, act
            continue

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

if args.save:
    path = output_root / STATS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    new_rows = statistics_frame(collected, collected_knn, dropped_of).to_dict(
        orient="records"
    )
    new_keys = {row_key(row) for row in new_rows}
    kept = [row for key, row in on_file.items() if key not in new_keys]
    if kept:
        print(f"Keeping {len(kept)} statistic(s) already in {path}")
    frame = pd.DataFrame(kept + new_rows, columns=list(STATS_COLUMNS))
    try:
        frame.to_parquet(path, index=False)
        print(f"\n{len(frame)} statistic(s) saved to {path}")
    except Exception as e:
        print(f"\nError saving the statistics to {path}: {e}")

# --- Recap --------------------------------------------------------------------
# The runs matching no YAML are reported per plot config, above, where the
# exemptions that decide the matching are known.
print(
    f"Runs found : {len(run_dirs)}"
    f"  ({len(no_cfg_dirs)} without a cfg.yaml;"
    f" {len(no_memory_dirs)} config(s) with no memory.npz,"
    f" {len(filtered_dirs)} with none of --rng_seeds)"
)
print(
    f"Runs needed: {len(needed)}"
    f"  ({read_runs} read from memory.npz,"
    f" {filled_runs} filled from {STATS_FILE},"
    f" {idle_runs} with nothing drawn to compute,"
    f" {len(corrupt_dirs)} unreadable)"
)
print(
    f"kNN entropy: {len(collected_knn)} run(s)"
    f"  ({len(non_finite_dirs)} not finite)"
    if drawn_knn
    else "kNN entropy: no plot config draws it, so none was estimated"
)
# The runs holding a statistic, not the ones given an entry: `collected` is given
# a run's entry while its counters are resolved, which is before the memory it
# needs is read.
binned_runs = sum(
    1
    for per_counter in collected.values()
    if any(per_bins for per_bins in per_counter.values())
)
print(
    f"Binned     : {binned_runs} run(s)"
    f"  ({len(no_counter_dirs)} with no binned counter)"
    if drawn_binned
    else "Binned     : no plot config draws them, so no memory was binned"
)
# Both lines are about the kNN estimate, so they are printed where one was made.
if drawn_knn:
    print(
        f"Estimator  : k={args.knn_k}, whole memory, no subsampling, "
        f"conditioned on {list(args.discrete_cols) or 'no column'} and the action"
    )
if args.load:
    print(f"Loaded     : {reused} statistic(s) taken from {STATS_FILE}")
# How much of each memory the kNN sum passed over. What it costs the estimate is
# in `memory_entropy`.
if drawn_knn and dropped_of:
    worst = max(dropped_of, key=dropped_of.get)
    print(
        f"Dropped    : {len(dropped_of)} of {len(collected_knn)} run(s) hold "
        f"conditioning values of {args.knn_k} observations or fewer, up to "
        f"{dropped_of[worst]:.3g} of a memory ({worst})"
    )
    for name in sorted(dropped_of, key=dropped_of.get, reverse=True):
        vprint(f"  {name}: {dropped_of[name]:.3g}")
elif drawn_knn:
    print(
        f"Dropped    : none, every conditioning value holds more than "
        f"{args.knn_k} observations"
    )
binning_lines = []
for (key, _counter_name), cls in _class_cache.items():
    env = _env_cache[key]
    if cls is None:
        how = "no binned counter"
    else:
        n_bins = [int(b) for b in make_counter(env, cls, BINS[0]).n_bins]
        how = f"{cls.__name__}, {n_bins} at {BINS[0]} bins per dimension"
    line = f"Binning    : {env_family(env)}  ->  {how}"
    if line not in binning_lines:
        binning_lines.append(line)
for line in binning_lines:
    print(line)
if no_counter_dirs:
    print(
        "\nNo binned statistics (the environment has no binned counter, so no "
        "bin size to vary):"
    )
    for d, en in no_counter_dirs:
        print(f"  {d}  →  {en}")
oversized: dict = {}
for name, per_counter in collected.items():
    for per_bins in per_counter.values():
        for n_bins, per_stat in per_bins.items():
            if not all(np.isfinite(v) for v in per_stat.values()):
                oversized.setdefault(env_of_run.get(name, ""), set()).add(n_bins)
if oversized:
    print(
        "\nNo binned statistics at these bin sizes (the (bin, action) table is "
        "too large for an int64 flat index):"
    )
    for en, sizes in sorted(oversized.items(), key=lambda kv: str(kv[0])):
        print(f"  {en}: {', '.join(str(b) for b in sorted(sizes))} bins")
if non_finite_dirs:
    print(
        "\nNo kNN entropy (observations coincide exactly in the estimated "
        "columns, so the k-th neighbour sits at distance zero). "
        "--discrete_cols may be naming too few columns:"
    )
    for d, value in non_finite_dirs:
        print(f"  {d}: {value}")
if corrupt_dirs:
    print("\nSkipped (unreadable):")
    for d in corrupt_dirs:
        print(f"  {d}")

if not collected_knn and not binned_runs:
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
    _stat_labels, env_labels, ylims = plot_labels(plot_cfg)
    entries, groups, selected, envs, counter_of_env = layout[config_name]

    binned_stat_labels, stats = wanted_statistics(plot_cfg)

    output_dir = output_root / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    def bars_of(env, stat, n_bins):
        """Collect the value of one environment and statistic per entry and
        return it as [(entry, (mean, half-width) or None)], with the seeds each
        was read from.

        `n_bins` of None reads the kNN entropy, which stands on its own for every
        binning. Every entry keeps its slot, with None where its runs hold no
        finite value, so the same configuration is the same bar in every subplot.
        """

        known = collected if n_bins is not None else collected_knn
        items, alive_of = [], {}
        for li, entry in enumerate(entries):
            values = []
            for name in selected.get((li, env), []):
                run = known.get(name)
                if run is not None and n_bins is not None:
                    cls_name = resolved.get((name, counter_of_env.get(env)))
                    run = run.get(cls_name, {}).get(n_bins)
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
        """Draw one figure, a subplot per environment, a bar per configuration
        and a 95% confidence interval over the seeds, and return how many
        figures were written (0 or 1)."""

        # A horizontal subplot holds its configurations down the side, so its
        # height is theirs: a row each, the wider gaps between colour groups, and
        # a fixed allowance above and below for the title and the value axis. The
        # axes are then pinned to that allowance, so a row is HBAR_UNIT_INCHES
        # tall whatever the configuration count.
        slots = len(entries) + GROUP_GAP_BAR * sum(
            1 for a, b in zip(entries, entries[1:]) if a["group"] != b["group"]
        )
        bars_inches = HBAR_UNIT_INCHES * slots
        height = (
            bars_inches + HBAR_TOP_INCHES + HBAR_BOTTOM_INCHES
            if args.horizontal_bars
            else SUBPLOT_H
        )
        fig, axs = plt.subplots(
            1,
            len(drawn_envs),
            figsize=(SUBPLOT_W * len(drawn_envs), height),
            squeeze=False,
        )
        if args.horizontal_bars:
            fig.subplots_adjust(
                wspace=0.3,
                top=1.0 - HBAR_TOP_INCHES / height,
                bottom=HBAR_BOTTOM_INCHES / height,
            )
        else:
            fig.subplots_adjust(wspace=0.3)
        shown = set()
        drew = False

        for col, env in enumerate(drawn_envs):
            ax = axs[0][col]
            title = env_labels.get(env, env)
            ax.set_title(title, fontsize=FONT_SIZE, **tex_kwargs(title))

            items, _alive = bars_of(env, stat, n_bins)
            if not any(v is not None for _e, v in items):
                if col == 0 and args.horizontal_bars:
                    draw_hbars(
                        ax,
                        items,
                        label=ylabel,
                        bar_width=args.bar_width,
                        font_size=FONT_SIZE,
                        names=True,
                    )
                elif col == 0:
                    ax.set_ylabel(ylabel, fontsize=FONT_SIZE, **tex_kwargs(ylabel))
                continue
            # The bars name the configurations by colour and hatch, which the
            # legend spells out once for the figure, and every bar stands from
            # one floor below the lowest interval.
            if args.horizontal_bars:
                draw_hbars(
                    ax,
                    items,
                    label=ylabel,
                    bar_width=args.bar_width,
                    font_size=FONT_SIZE,
                    names=col == 0,
                )
            else:
                draw_bars(
                    ax,
                    items,
                    ylabel=ylabel if col == 0 else "",
                    bar_width=args.bar_width,
                    font_size=FONT_SIZE,
                )
            drew = True
            shown.update(li for li, (_e, v) in enumerate(items) if v is not None)

            # The plot config's limits are on the value axis, whichever one that
            # is.
            axis = "x" if args.horizontal_bars else "y"
            if (env, stat) in ylims:
                (ax.set_xlim if args.horizontal_bars else ax.set_ylim)(
                    *ylims[(env, stat)]
                )
            value_ticks(ax, axis)

        if not drew:
            plt.close(fig)
            return 0

        # Every configuration drawn in any subplot, in the plot config's order.
        handles = [
            mpatches.Patch(
                facecolor=entry["color"],
                hatch=entry["hatch"],
                edgecolor="black",
                linewidth=BAR_EDGE_WIDTH,
                label=entry["label"],
            )
            for li, entry in enumerate(entries) if li in shown
        ]
        if handles and not args.no_legend:
            legend = fig.legend(
                handles,
                [h.get_label() for h in handles],
                fontsize=FONT_SIZE - 2,
                frameon=True,
                loc="lower center",
                ncol=rows_cols(plot_cfg.get("legend_rows_cols"), len(handles))[1],
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

    written += write_legend(entries, plot_cfg, output_dir)

print(f"\n{written} figure(s) written to {output_root}")
