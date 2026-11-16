"""Causes of the goal reselections of goal-conditioned runs, from a pre-built
results.parquet and the runs' replay-memory dumps, laid out by a plot config.

The plot config names the configurations and the environments to draw, as for
plot_results.py (see configs/plots/gcrl_selections/reselections.yaml). Runs are selected
as plot_results.py selects them: a run is drawn only when its whole
configuration is the one its entry composes to. For every environment group,
three files are written in <root>/<plot config>, the path of the plot config
under configs/plots without the extension:

  reselection_causes_<group>.png
                    a 100% stacked horizontal bar per environment and entry
                    with reselections: the fraction of the reselections, pooled
                    over seeds, whose goal was unreachable where it was
                    selected, became unreachable, lost true value to a random
                    action, to a suboptimal action or to the environment's
                    noise, or kept its true value while the value check fired
                    (approximation error). Each reselection has one cause (see
                    src.utils.reachability.reselection_causes).
  reselection_unreachable_<group>.png
                    a 100% stacked horizontal bar per environment and entry
                    with goals unreachable from the state of their last
                    selection: the fraction of them, pooled over seeds, that a
                    reselection replaced and that were kept until the end of
                    their episode, with their number and the median steps from
                    their first selection to the reselection at the end of the
                    bar (see src.utils.reachability.unreachable_selections).
  reselection_causes_steps_<group>.png
                    the same reselections over training, a row per entry and a
                    column per environment with reselections: a histogram in
                    --cause_bins equal-width bins of steps, each bar the mean
                    number of reselections per seed per 1,000 training steps
                    in the bin, stacked by cause in the same colours. Its axes
                    carry round ticks (see round_ticks). The statistics of
                    OVERLAY_STATS the runs logged are drawn over the bars on a
                    log-scale right axis, as the mean over seeds.

The `histograms` block of the plot config maps a group name to a list of
environments, and each group gets one more reselection_causes_steps_<group>.png
over those environments alone, without the curves of OVERLAY_STATS or the bar
figures, with the total reselections as a percentage of the training
steps in the top right corner of every subplot and the legend of the causes in
one column inside the first subplot, and its legend of the
causes alone in one column as
reselection_causes_steps_<group>_legend_vertical.png and in one row as
reselection_causes_steps_<group>_legend_horizontal.png. The environments are
labelled as
in `environments`.

When the plot config has a `margin` block, one more file is written there:

  reselection_margin.png
                    a row per entry: the histogram of reselection_causes_steps
                    of every environment listed under `margin.histograms`, and
                    next to it the reselections attributed to approximation
                    error by the steps of true progress toward the goal
                    replaced since its selection
                    (src.utils.reachability.critic_error_progress), per seed per
                    1,000 training steps on a log scale, a bar per environment
                    of `margin.environments` (environment: label) at every k.

The legend of the causes and of the curves of OVERLAY_STATS is also written
alone, once per plot config, as reselection_causes_legend.png in the same
directory, whatever --no_legend says. The decision tree that assigns the causes
is written once per plot config too, as reselection_causes_tree.png.

The reselections are read from each run's <config_id>/<seed>/memory.npz and
classified with the true visit Q-function of its environment
(src.oracle.TrueVisitQ), built from the run's cfg.yaml. Runs whose environment
TrueVisitQ does not support are left out.

Run with --help for the rest of the options.

Example

    python plot_reselections.py -f data_gcrl_oracle -v
"""

import argparse
import json
import os
import sys
from ast import literal_eval
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

from src.oracle import TrueVisitQ
from src.pseudocount import tabular_count
from src.utils.heatmaps import env_context
from src.utils.id import CONFIG_ID, SEED, read_run_cfg
from src.utils.matching import ignored_cfg_keys, load_config_group
from src.utils.misc import make_vprint
from src.utils.plot import (
    PLOT_CONFIG_DIR,
    add_config_group_args,
    assign_groups,
    ensure_font,
    entry_rows,
    env_groups,
    hyperparameter_strings,
    load_plot_configs,
    plot_labels,
    save_and_close,
    series_of,
    style_entries,
    tex_kwargs,
)
from src.utils.reachability import (
    ORACLE_TOL,
    RESELECTION_CAUSES,
    critic_error_progress,
    reselection_causes,
    unreachable_selections,
)
from src.wrappers import gym_wrappers

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

FONT_SIZE = 12

# The memory keys the causes of the reselections are read from.
MEMORY_KEYS = (
    "obs",
    "act",
    "term",
    "trunc",
    "goal_obs",
    "goal_act",
    "goal_selection",
    "has_goal",
    "act_source",
)

# Reselection causes of each memory dump read, by its path (see memory_causes),
# the true visit Q-functions built (see true_visit_of), and read_run_cfg of the
# cfg.yaml of each run directory, read once for all of its seeds.
_memory_causes: dict = {}
_true_visit: dict = {}
_run_cfgs: dict = {}

# Label, colour and label colour of each cause of RESELECTION_CAUSES, in its
# order: reds for the goals that were or became unreachable, blues for the true
# value lost while acting, grey for the value checks that fired without a drop.
CAUSE_STYLE = {
    "unreachable_at_selection": ("Unreachable at selection", "#b2182b", "white"),
    "became_unreachable": ("Became unreachable", "#ef8a62", "black"),
    "random_action": ("Random action", "#d1e5f0", "black"),
    "suboptimal_action": ("Suboptimal action", "#4393c3", "white"),
    "env_noise": ("Environment noise", "#2166ac", "white"),
    "critic_error": ("Approximation error", "#bababa", "black"),
}
# Label, colour and label colour of the goals unreachable from the state of
# their last selection that a reselection replaced, in the red of
# unreachable_at_selection, and of those kept until the end of their episode.
UNREACHABLE_STYLE = {
    "replaced": ("Replaced by the check", "#b2182b", "white"),
    "kept": ("Kept until the end of the episode", "#fddbc7", "black"),
}
# Statistic, label and colour of each curve drawn over the bars of
# reselection_causes_steps: the absolute error of the difference
# V(s, g, a_g) - V(s_sel, g, a_g), with V(s, g, a_g) = max_a Q(s, a | g, a_g),
# that the value check compares against 0, logged at every check with
# results.true_visit_check, and the visit critic's training loss.
OVERLAY_STATS = (
    ("v_err_check_diff", "Approx. error", "black"),
    ("loss_visit", "Training TD error", "white"),
)
# Line width of the curves of OVERLAY_STATS, and the outline drawn around each
# curve, keyed by the curve's color.
OVERLAY_WIDTH = 2.0
OUTLINE = {
    "black": [pe.Stroke(linewidth=OVERLAY_WIDTH + 1.0, foreground="white"), pe.Normal()],
    "white": [pe.Stroke(linewidth=OVERLAY_WIDTH + 1.0, foreground="black"), pe.Normal()],
}
# A segment of a bar narrower than this fraction is drawn without its
# percentage.
CAUSE_LABEL_MIN = 0.04
# Inches per bar of reselection_causes, and the width of the figure.
CAUSE_BAR_H = 0.55
CAUSE_FIG_W = 8.0
# Inches of margin below the entries of the standalone legends of the causes
# of write_cause_steps_legends.
CAUSE_LEGEND_BOTTOM = 0.25
# Height, as a fraction of the axes, of the top of the legend of the causes
# drawn inside a subplot of draw_cause_steps_figure.
INNER_LEGEND_TOP = 0.69
# Inches spanned by the axis limits of reselection_causes_tree, in width and in
# height, before the figure is cropped to its boxes and notes.
TREE_FIG_W = 8.9
TREE_FIG_H = 5.5
# Inches per subplot of reselection_causes_steps, in width and in height.
CAUSE_HIST_W = 3.9
CAUSE_HIST_H = 2.0
# Inches per subplot of reselection_margin in width (its height is
# CAUSE_HIST_H), the steps of true progress from which its reselections share
# one bar, the grays of its environments, from dark to light, and the font size
# of its titles, axis labels and legend (two points smaller).
MARGIN_W = 5.0
MARGIN_TAIL = 6
MARGIN_GRAYS = (0.15, 0.85)
MARGIN_FONT_SIZE = FONT_SIZE - 1
# Most intervals between the round ticks (see round_ticks) of a step axis and of
# a value axis.
STEP_TICK_BINS = 2
VALUE_TICK_BINS = 4

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--folder",
    required=True,
    metavar="DIR",
    help="Data directory holding the parquet file and the runs.",
)
parser.add_argument("--parquet", default="results.parquet")
parser.add_argument(
    "-p", "--plot",
    default="gcrl_selections/reselections",
    metavar="NAME",
    help=f"Plot config: which configurations to draw, and over which environments. "
         f"A name without .yaml is looked for in {PLOT_CONFIG_DIR}, or give a path; "
         f"a directory is a set of configs and every one of them is drawn, bar "
         f"those marked `# skip`.",
)
parser.add_argument(
    "-o", "--output",
    default="plots",
    help="Output root, under the data directory. Files go in <root>/<the "
         "path of the plot config under configs/plots, without .yaml>.",
)
add_config_group_args(parser)
parser.add_argument(
    "--cause_bins",
    type=int,
    default=20,
    help="Bins of training steps of the reselection_causes_steps histograms.",
)
parser.add_argument(
    "--no_legend",
    action="store_true",
    help="Leave the legend off the figures. The standalone legend is still "
         "written.",
)
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

vprint = make_vprint(args.verbose)

plot_cfgs = load_plot_configs(args.plot)
vprint(f"\n{len(plot_cfgs)} plot config(s): {[n for n, _, _ in plot_cfgs]}")

parquet_path = os.path.join(args.folder, args.parquet)
df = pd.read_parquet(parquet_path)
vprint(f"\nLoaded parquet file from: {parquet_path}")
if all(isinstance(c, str) and c.startswith("(") for c in df.columns):
    df.columns = pd.MultiIndex.from_tuples([literal_eval(c) for c in df.columns])

# Every hyperparameter column as value_str renders it, once: entry_rows compares
# against them for every (environment, entry).
hp_strings = hyperparameter_strings(df)

env_configs = load_config_group(args.environments) if args.environments else {}
algo_configs = load_config_group(args.algorithms) if args.algorithms else {}
_assigned: dict = {}

ensure_font()
sns.set_context("paper")
sns.set_style("darkgrid", {"legend.frameon": True})
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.axisbelow"] = False
plt.rcParams["grid.linestyle"] = "--"


@dataclass
class Drawing:
    """Everything one plot config decides, for the functions that draw it.

    `rows` is the cache entry_rows keeps, mapping (environment, entry index) to
    the runs the entry selects there."""

    name: str
    cfg: dict
    entries: list
    env_labels: dict
    ignored_keys: frozenset
    matched: pd.Series
    env_stem: pd.Series
    output_dir: str
    rows: dict = field(default_factory=dict)


def rows_of(drawing, env, index):
    """Return the runs entry `index` selects in `env` (see entry_rows)."""

    return entry_rows(
        df,
        hp_strings,
        drawing.rows,
        env,
        index,
        drawing.entries[index],
        drawing.ignored_keys,
        drawing.matched,
    )


def true_visit_of(env_cfg, critic_cfg):
    """Return the true visit Q-function (src.oracle.TrueVisitQ) of the
    environment `env_cfg` under the discount and the rewards of the visit critic
    `critic_cfg`, or None when the critic has no visit discount or TrueVisitQ
    does not support the environment. Built once per process for each
    combination."""

    gamma = critic_cfg.get("gamma_visit")
    if gamma is None:
        return None
    negative = bool(critic_cfg.get("negative_visit_reward", False))
    key = json.dumps([env_cfg, gamma, negative], sort_keys=True, default=str)
    if key not in _true_visit:
        env = gym_wrappers.make_gym_env(**env_cfg)
        try:
            _true_visit[key] = TrueVisitQ(
                env,
                tabular_count(env),
                gamma=float(gamma),
                negative_reward=negative,
            )
        except ValueError as e:
            vprint(f"  WARNING: no true visit Q-function for {env_cfg.get('id')}: {e}")
            _true_visit[key] = None
        finally:
            env.close()
    return _true_visit[key]


def read_memory_causes(run_dir, mem_file):
    """Return `(causes, progress, fates, held)` for the memory dump `mem_file` of
    a run in `run_dir`: the cause of every reselection, as reselection_causes
    returns them, the true progress of the reselections attributed to
    critic_error, as critic_error_progress returns it, and whether a
    reselection replaced every goal unreachable from the state of its last
    selection, with the steps it was held before, as unreachable_selections
    returns them. Return None when the
    dump or the run's cfg.yaml cannot be read, the dump lacks a key of
    MEMORY_KEYS, the environment has no tabular counter, or true_visit_of gives
    no true visit Q-function."""

    if not os.path.isfile(mem_file):
        return None
    if run_dir not in _run_cfgs:
        _run_cfgs[run_dir] = read_run_cfg(os.path.join(run_dir, "cfg.yaml"))
    _, run_cfg, reason = _run_cfgs[run_dir]
    if reason is not None:
        vprint(f"  WARNING: {run_dir}/cfg.yaml {reason}.")
        return None
    env_cfg = run_cfg.get("environment") or {}
    counter = env_context(env_cfg)[0]
    true_visit = true_visit_of(env_cfg, (run_cfg.get("agent") or {}).get("critic") or {})
    if counter is None or true_visit is None:
        return None
    try:
        with np.load(mem_file) as data:
            mem = {k: data[k] for k in data.files if k in MEMORY_KEYS}
    except Exception as e:
        vprint(f"  WARNING: skipping unreadable {mem_file}: {e}")
        return None
    if any(k not in mem for k in MEMORY_KEYS):
        vprint(f"  WARNING: {mem_file} lacks {[k for k in MEMORY_KEYS if k not in mem]}, skipping.")
        return None
    size = len(mem["obs"])
    flat = {k: np.asarray(mem[k]).reshape(size) for k in MEMORY_KEYS if k not in ("obs", "goal_obs")}
    state = np.asarray(counter.bin_index(mem["obs"])).reshape(size)
    goal = np.asarray(counter.bin_index(mem["goal_obs"])).reshape(size)
    goal_act = flat["goal_act"].astype(np.int64)
    selected = flat["goal_selection"].astype(bool)
    causes = reselection_causes(
        true_visit,
        state,
        goal,
        flat["act"].astype(np.int64),
        goal_act,
        flat["has_goal"].astype(bool),
        flat["act_source"],
        selected,
        flat["term"].astype(bool) | flat["trunc"].astype(bool),
        ORACLE_TOL,
    )
    progress = critic_error_progress(true_visit, state, goal, goal_act, selected, causes)
    fates, held = unreachable_selections(
        true_visit,
        state,
        goal,
        flat["act"].astype(np.int64),
        goal_act,
        flat["has_goal"].astype(bool),
        selected,
        flat["term"].astype(bool) | flat["trunc"].astype(bool),
    )
    return causes, progress, fates, held


def memory_causes(config_id, seed):
    """Return read_memory_causes for the run <config_id>/<seed> of the data
    directory, reading its dump once per process."""

    run_dir = os.path.join(args.folder, str(config_id))
    mem_file = os.path.join(run_dir, str(int(seed)), "memory.npz")
    if mem_file not in _memory_causes:
        _memory_causes[mem_file] = read_memory_causes(run_dir, mem_file)
    return _memory_causes[mem_file]


def memory_runs(drawing, env, index):
    """Return read_memory_causes of every run entry `index` selects in `env`,
    leaving out the runs without them."""

    rows = rows_of(drawing, env, index)
    if any(c not in rows.columns for c in (CONFIG_ID, SEED)):
        return []
    runs = []
    for config_id, seed in rows[[CONFIG_ID, SEED]].drop_duplicates().itertuples(index=False):
        read = memory_causes(config_id, seed)
        if read is not None:
            runs.append(read)
    return runs


def cause_runs(drawing, env, index):
    """Return the reselection causes of every run entry `index` selects in
    `env`, one int array over the steps of its memory dump per run (see
    reselection_causes), leaving out the runs without them (see
    read_memory_causes)."""

    return [causes for causes, _progress, _fates, _held in memory_runs(drawing, env, index)]


def progress_rates(drawing, env, index):
    """Return the reselections attributed to critic_error in the runs entry
    `index` selects in `env`, by their steps of true progress (see
    critic_error_progress) rounded to the nearest integer: one count per k from
    0 to MARGIN_TAIL - 1 and one for every k from MARGIN_TAIL on, each as the
    mean number per seed per 1,000 training steps over the steps of their memory
    dumps. Return None when no run has them (see read_memory_causes)."""

    runs = [progress for _causes, progress, _fates, _held in memory_runs(drawing, env, index)]
    if not runs:
        return None
    k = np.rint(np.concatenate([p[~np.isnan(p)] for p in runs])).astype(np.int64)
    counts = np.bincount(np.clip(k, 0, MARGIN_TAIL), minlength=MARGIN_TAIL + 1)
    return 1000.0 * counts / sum(len(p) for p in runs)


def cause_counts(drawing, env, index):
    """Return `(counts, n_runs, n_steps)` for the runs entry `index` selects in
    `env`: how many reselections of each cause of RESELECTION_CAUSES they made,
    summed over their seeds, how many runs have the causes of their
    reselections, and the steps of the longest of their memory dumps, which is
    the training length. Return None when no run has them."""

    runs = cause_runs(drawing, env, index)
    if not runs:
        return None
    counts = sum(np.bincount(c[c >= 0], minlength=len(RESELECTION_CAUSES)) for c in runs)
    return counts, len(runs), max(len(c) for c in runs)


def add_cause_legend(fig, overlays=(), n_cols=3, fontsize=FONT_SIZE - 2, **kwargs):
    """Draw the legend of the causes of CAUSE_STYLE on `fig`, a figure or an
    axes, without a frame, in `fontsize` points, placed by the legend keyword
    arguments `kwargs`. Without curves, the
    causes fill `n_cols` columns. With the curves of OVERLAY_STATS named in
    `overlays`, the causes keep the columns they take when the curves follow
    them in three columns, and the curves fill one more row below them, one per
    column from the left."""

    causes = [
        (mpatches.Patch(facecolor=color, edgecolor="white"), label)
        for label, color, _text in CAUSE_STYLE.values()
    ]
    curves = [
        (Line2D([], [], color=color, linewidth=OVERLAY_WIDTH, path_effects=OUTLINE[color]), label)
        for stat, label, color in OVERLAY_STATS
        if stat in overlays
    ]
    entries = causes
    if curves:
        blank = (Line2D([], [], linestyle="none"), "")
        # fig.legend fills its columns top to bottom, so the entries are listed
        # column by column, each column padded to the same number of rows.
        n_rows = -(-(len(causes) + len(curves)) // 3)
        columns = [causes[i:i + n_rows] for i in range(0, len(causes), n_rows)]
        entries, n_cols = [], len(columns)
        for j, column in enumerate(columns):
            entries += column + [blank] * (n_rows - len(column))
            entries.append(curves[j] if j < len(curves) else blank)
    fig.legend(
        [handle for handle, _label in entries],
        [label for _handle, label in entries],
        ncol=n_cols,
        frameon=False,
        fontsize=fontsize,
        **kwargs,
    )


def legend_top(fig, axes):
    """Return the height, as a fraction of `fig`, just under where `axes` and
    their tick labels end, measured."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottom = min(ax.get_tightbbox(renderer).y0 for ax in axes)
    return fig.transFigure.inverted().transform((0, bottom))[1] - 0.02


def cause_legend(fig, axes, overlays=()):
    """Draw the legend of the causes of CAUSE_STYLE and of the curves of
    OVERLAY_STATS named in `overlays` under where `axes` and their tick labels
    end (see legend_top), unless --no_legend."""

    if args.no_legend:
        return
    add_cause_legend(fig, overlays, loc="upper center", bbox_to_anchor=(0.5, legend_top(fig, axes)))


def write_cause_legend(output_dir):
    """Write the legend of the causes of CAUSE_STYLE and of every curve of
    OVERLAY_STATS alone, as the figures draw it, to
    reselection_causes_legend.png in `output_dir`, and return the number of
    figures written."""

    fig = plt.figure(figsize=(CAUSE_FIG_W, 0.8))
    add_cause_legend(fig, [stat for stat, _label, _color in OVERLAY_STATS], loc="center")
    return save_and_close(fig, output_dir, "reselection_causes_legend", vprint)


def write_cause_steps_legends(output_dir, name):
    """Write the legend of the causes of CAUSE_STYLE alone, in one column to
    <name>_vertical.png and in one row to <name>_horizontal.png in
    `output_dir`, and return the number of figures written. Each image is
    cropped to the entries on the top, left and right, and keeps
    CAUSE_LEGEND_BOTTOM inches of margin below them."""

    n = len(CAUSE_STYLE)
    written = 0
    for shape, n_cols, size in (
        ("vertical", 1, (CAUSE_FIG_W / 3, 0.3 * n)),
        ("horizontal", n, (CAUSE_FIG_W * 2, 0.5)),
    ):
        fig = plt.figure(figsize=size)
        add_cause_legend(fig, n_cols=n_cols, loc="center", borderpad=0.0)
        fig.canvas.draw()
        box = fig.legends[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        crop = Bbox.from_extents(box.x0, box.y0 - CAUSE_LEGEND_BOTTOM, box.x1, box.y1)
        written += save_and_close(fig, output_dir, f"{name}_{shape}", vprint, bbox_inches=crop)
    return written


def write_cause_tree(output_dir):
    """Write the decision tree reselection_causes follows to assign the cause
    of a reselection to reselection_causes_tree.png in `output_dir`, and return
    the number of figures written.

    Every test is a white box and every cause a box in its colours of
    CAUSE_STYLE, with a note under it on the decision it reflects. Every branch
    is labelled with the outcome of its test. The figure is cropped to the
    boxes and the notes, keeping the inches per axis unit of TREE_FIG_W and
    TREE_FIG_H, and saved without padding."""

    tests = {
        "sel": (5.0, 5.4, "Goal $g$ dropped at $s_{\\mathrm{now}}$:\nis $g$ reachable from $s_{\\mathrm{sel}}$?"),
        "now": (6.2, 4.1, "Is $g$ reachable from $s_{\\mathrm{now}}$?"),
        "drop": (7.6, 2.8, "$V^*(s_{\\mathrm{now}}) < V^*(s_{\\mathrm{sel}})$?"),
        "step": (5.2, 1.5, "First step losing $V^*$:\nwhat action caused it?"),
    }
    causes = {
        "unreachable_at_selection": (3.2, 4.1, "wrong selection of $s_{\\mathrm{sel}}$\ncorrect reselection"),
        "became_unreachable": (3.6, 2.8, "correct reselection"),
        "critic_error": (8.1, 1.5, "critic error in the check"),
        "random_action": (2.8, 0.2, "$\\varepsilon$-greedy exploratory action"),
        "suboptimal_action": (5.2, 0.2, "critic error while acting"),
        "env_noise": (8.2, 0.2, "transition stochasticity"),
    }
    edges = (
        ("sel", "unreachable_at_selection", "no"),
        ("sel", "now", "yes"),
        ("now", "became_unreachable", "no"),
        ("now", "drop", "yes"),
        ("drop", "critic_error", "no"),
        ("drop", "step", "yes (correct reselection)"),
        ("step", "random_action", "random"),
        ("step", "suboptimal_action", "not optimal under $Q^*$"),
        ("step", "env_noise", "optimal"),
    )
    position = {name: (x, y) for name, (x, y, _text) in {**tests, **causes}.items()}

    fig, ax = plt.subplots(figsize=(TREE_FIG_W, TREE_FIG_H))
    ax.set_axis_off()
    ax.set_xlim(1.1, 10.0)
    ax.set_ylim(-0.6, 6.0)
    for parent, child, label in edges:
        (x0, y0), (x1, y1) = position[parent], position[child]
        ax.plot([x0, x1], [y0, y1], color="0.4", linewidth=1.0, zorder=1)
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            label,
            ha="center",
            va="center",
            fontsize=FONT_SIZE - 3,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"),
            zorder=2,
        )
    for x, y, text in tests.values():
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=FONT_SIZE - 1,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.4"),
            zorder=3,
        )
    for cause, (x, y, note) in causes.items():
        label, color, text_color = CAUSE_STYLE[cause]
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            color=text_color,
            fontsize=FONT_SIZE - 1,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor="white"),
            zorder=3,
        )
        ax.text(x, y - 0.28, note, ha="center", va="top", fontsize=FONT_SIZE - 3, style="italic", zorder=3)

    # The axes fill the figure, then the limits and the figure size shrink to
    # the union of the texts and their boxes at the same inches per unit.
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.canvas.draw()
    extents = [
        t.get_window_extent() if t.get_bbox_patch() is None else t.get_bbox_patch().get_window_extent()
        for t in ax.texts
    ]
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(Bbox.union(extents).get_points())
    (lx0, lx1), (ly0, ly1) = ax.get_xlim(), ax.get_ylim()
    fig.set_size_inches(TREE_FIG_W * (x1 - x0) / (lx1 - lx0), TREE_FIG_H * (y1 - y0) / (ly1 - ly0))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    return save_and_close(fig, output_dir, "reselection_causes_tree", vprint, pad_inches=0.0)


def round_ticks(ax, which, n_bins):
    """Put round ticks on the `which` axis ("x" or "y") of `ax`: multiples of
    1, 2, 2.5 or 5 times a power of ten, at most `n_bins` intervals apart over
    its limits, and grow the limits to the round ticks just outside them, so
    the axis starts and ends on a tick. A value axis is labelled in the shortest
    form of each tick ("%g")."""

    if which == "x":
        get_lim, set_lim, set_ticks = ax.get_xlim, ax.set_xlim, ax.set_xticks
    else:
        get_lim, set_lim, set_ticks = ax.get_ylim, ax.set_ylim, ax.set_yticks
    lo, hi = get_lim()
    ticks = MaxNLocator(nbins=n_bins, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    eps = 1e-9 * max(abs(hi - lo), 1e-12)
    first = ticks[ticks <= lo + eps].max(initial=ticks[0])
    last = ticks[ticks >= hi - eps].min(initial=ticks[-1])
    ticks = ticks[(ticks >= first) & (ticks <= last)]
    set_ticks(ticks)
    set_lim(ticks[0], ticks[-1])
    if which == "y":
        ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:g}")


def unreachable_stats(drawing, env, index):
    """Return `(replaced, kept, steps)` for the goals unreachable from the state
    of their last selection (see unreachable_selections) in the runs entry
    `index` selects in `env`, pooled over their seeds: how many of them a
    reselection replaced, how many were kept until the end of their episode,
    and the median steps from their first selection to the reselection, NaN
    when no reselection replaced one."""

    runs = memory_runs(drawing, env, index)
    fates = np.concatenate([f[f >= 0] for _causes, _progress, f, _held in runs] or [np.zeros(0)])
    held = np.concatenate([h[~np.isnan(h)] for _causes, _progress, _fates, h in runs] or [np.zeros(0)])
    steps = np.median(held) if held.size else np.nan
    return int((fates == 1).sum()), int((fates == 0).sum()), steps


def kilo(n):
    """Return the count `n` as text, in thousands with three significant
    digits and a "k" from 1,000 on."""

    return f"{float(f'{n / 1000:.3g}'):g}k" if n >= 1000 else f"{n}"


def draw_share_figure(drawing, title, bars, styles, notes):
    """Draw a 100% stacked horizontal bar per (environment, entry index,
    counts) of `bars`, titled `title`, and return the figure and its axes.

    The segments of a bar are the fractions of its counts, in the order of
    `styles`, a (label, colour, label colour) per count. A segment at least
    CAUSE_LABEL_MIN wide is labelled with its percentage. The text of `notes`,
    one per bar, is written at the end of the bar. A bar is named by its
    environment, and by its entry too when several entries have bars."""

    several = len({i for _, i, _ in bars}) > 1
    fig, ax = plt.subplots(figsize=(CAUSE_FIG_W, CAUSE_BAR_H * len(bars) + 0.6))
    labels = []
    for y, ((env, i, counts), note) in enumerate(zip(bars, notes)):
        label = drawing.env_labels.get(str(env), str(env))
        if several:
            label = f"{label}\n{drawing.entries[i]['label']}"
        labels.append(label)
        total = counts.sum()
        left = 0.0
        for (_label, color, text_color), n in zip(styles, counts):
            frac = n / total
            ax.barh(y, frac, left=left, height=0.7, color=color, edgecolor="white", linewidth=0.8)
            if frac >= CAUSE_LABEL_MIN:
                ax.text(
                    left + frac / 2,
                    y,
                    f"{100 * frac:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZE - 3,
                    color=text_color,
                )
            left += frac
        ax.text(
            1.01,
            y,
            note,
            ha="left",
            va="center",
            fontsize=FONT_SIZE - 3,
            clip_on=False,
        )
        vprint(
            f"  {label.replace(chr(10), ' / ')}: {int(total)}, "
            + ", ".join(f"{s[0]} {100 * n / total:.1f}%" for s, n in zip(styles, counts))
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(len(bars) - 0.5, -0.5)
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels(labels, fontsize=FONT_SIZE - 1)
    for text in ax.get_yticklabels():
        text.set(**tex_kwargs(text.get_text()))
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(lambda v, _pos: f"{100 * v:.0f}%")
    ax.tick_params(axis="x", labelsize=FONT_SIZE - 2)
    ax.tick_params(axis="y", length=0)
    ax.yaxis.grid(False)
    ax.set_title(title, fontsize=FONT_SIZE)
    return fig, ax


def draw_cause_figure(drawing, name, envs):
    """Draw the causes of the reselections as a 100% stacked horizontal bar per
    (environment, entry) of `envs` with at least one reselection (see
    draw_share_figure), and return the number of figures written.

    Every reselection has one cause (see reselection_causes), and the segments
    of a bar are the fractions of its reselections of each cause, pooled over
    the seeds, in the order and the colours of CAUSE_STYLE. At the end of the
    bar, "total = <reselections>" is written over "(<x>% of <training
    steps>)", x being the mean number of reselections per seed as a percentage
    of the training steps, both counts written by kilo."""

    bars, notes = [], []
    for env in envs:
        for i in range(len(drawing.entries)):
            counted = cause_counts(drawing, env, i)
            if counted is not None and counted[0].sum() > 0:
                counts, n_runs, n_steps = counted
                total = int(counts.sum())
                bars.append((env, i, counts))
                notes.append(f"total = {kilo(total)}\n({100 * total / (n_runs * n_steps):.1f}% of {kilo(n_steps)})")
    if not bars:
        vprint(f"  No reselection for {name}, skipping.")
        return 0

    fig, ax = draw_share_figure(
        drawing,
        "Causes of Goal Reselections",
        bars,
        list(CAUSE_STYLE.values()),
        notes,
    )
    cause_legend(fig, [ax])
    return save_and_close(fig, drawing.output_dir, name, vprint)


def draw_unreachable_figure(drawing, name, envs):
    """Draw the goals unreachable from the state of their last selection (see
    unreachable_stats) as a 100% stacked horizontal bar per (environment,
    entry) of `envs` with at least one (see draw_share_figure), and return the
    number of figures written.

    The segments of a bar are the fractions of the goals, pooled over the
    seeds, that a reselection replaced and that were kept until the end of
    their episode, in the order and the colours of UNREACHABLE_STYLE, with its
    legend below. At the end of the bar, "total = <goals>" is written over
    "(median <s> steps)", s being the median steps from their first selection
    to the reselection, the count written by kilo."""

    bars, notes = [], []
    for env in envs:
        for i in range(len(drawing.entries)):
            replaced, kept, steps = unreachable_stats(drawing, env, i)
            if replaced + kept > 0:
                bars.append((env, i, np.array([replaced, kept])))
                median = "no replacement" if np.isnan(steps) else f"median {steps:g} steps"
                notes.append(f"total = {kilo(replaced + kept)}\n({median})")
    if not bars:
        vprint(f"  No goal unreachable at selection for {name}, skipping.")
        return 0

    fig, ax = draw_share_figure(
        drawing,
        "Goals Unreachable at Selection",
        bars,
        list(UNREACHABLE_STYLE.values()),
        notes,
    )
    if not args.no_legend:
        fig.legend(
            [mpatches.Patch(facecolor=color, edgecolor="white") for _label, color, _text in UNREACHABLE_STYLE.values()],
            [label for label, _color, _text in UNREACHABLE_STYLE.values()],
            ncol=len(UNREACHABLE_STYLE),
            frameon=False,
            fontsize=FONT_SIZE - 2,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_top(fig, [ax])),
        )
    return save_and_close(fig, drawing.output_dir, name, vprint)


def cause_hist(drawing, env, index):
    """Return `(edges, heights)` of the reselections of the runs entry `index`
    selects in `env` over training: --cause_bins equal-width bins of the steps
    of their memory dumps, and per cause of RESELECTION_CAUSES and bin the mean
    number of reselections per seed per 1,000 training steps. Return None when
    no run made a reselection."""

    runs = cause_runs(drawing, env, index)
    if not runs or not any((c >= 0).any() for c in runs):
        return None
    n_steps = max(len(c) for c in runs)
    edges = np.linspace(0.0, n_steps, args.cause_bins + 1)
    heights = np.zeros((len(RESELECTION_CAUSES), args.cause_bins))
    for causes in runs:
        steps = np.flatnonzero(causes >= 0)
        for c in range(len(RESELECTION_CAUSES)):
            heights[c] += np.histogram(steps[causes[steps] == c], bins=edges)[0]
    return edges, 1000.0 * heights / len(runs) / np.diff(edges)


def draw_cause_hist(ax, drawing, env, index, edges, heights, errors=True):
    """Draw the histogram `edges`, `heights` of cause_hist on `ax`, stacked by
    cause in the order and the colours of CAUSE_STYLE, with round ticks on both
    axes (see round_ticks). With `errors`, draw over the bars every statistic
    of OVERLAY_STATS the runs entry `index` selects in `env` logged, on a
    log-scale right axis against its logged steps, as the mean over seeds.
    Return the right axis, None when no statistic was drawn, and the set of
    statistics drawn."""

    bottom = np.zeros(heights.shape[1])
    for cause, height in zip(RESELECTION_CAUSES, heights):
        ax.bar(
            edges[:-1],
            height,
            width=edges[1] - edges[0],
            bottom=bottom,
            align="edge",
            color=CAUSE_STYLE[cause][1],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += height
    twin, drawn = None, set()
    rows = rows_of(drawing, env, index)
    for stat, _label, color in OVERLAY_STATS if errors else ():
        series = series_of(rows, stat)
        if series is None:
            continue
        if twin is None:
            twin = ax.twinx()
            twin.set_yscale("log")
            twin.grid(False)
        values, x, _steps, _seeds = series
        twin.plot(
            x,
            values.mean(axis=1).to_numpy(),
            color=color,
            linewidth=OVERLAY_WIDTH,
            path_effects=OUTLINE[color],
        )
        drawn.add(stat)
    if twin is not None:
        twin.tick_params(axis="y", labelsize=FONT_SIZE - 2, pad=1)
    ax.set_xlim(edges[0], edges[-1])
    round_ticks(ax, "x", STEP_TICK_BINS)
    ax.xaxis.set_major_formatter(lambda v, _pos: f"{v / 1000:g}k")
    ax.tick_params(axis="x", labelsize=FONT_SIZE - 2, pad=-2)
    ax.tick_params(axis="y", labelsize=FONT_SIZE - 2, pad=1)
    ax.set_ylim(bottom=0.0)
    round_ticks(ax, "y", VALUE_TICK_BINS)
    return twin, drawn


def draw_cause_steps_figure(
    drawing,
    name,
    envs,
    errors=True,
    legend=True,
    total=False,
    inner_legend=False,
):
    """Draw when the reselections happened, a row per entry and a column per
    environment of `envs` with at least one reselection, and return the number
    of figures written.

    A subplot is the histogram of cause_hist, drawn by draw_cause_hist with the
    statistics of OVERLAY_STATS over it when `errors` is set, which are added
    to the legend. A subplot with no reselection is left blank. Without
    `legend`, the figure is drawn without the legend under the subplots. With
    `inner_legend`, the legend of the causes is drawn in one column inside the
    top left subplot, in its right half below INNER_LEGEND_TOP. With `total`, the top
    right corner of every subplot reads "Total reselections:" over "<x>% of
    training steps", x being the mean number of reselections per seed as a percentage
    of the training steps (see cause_counts)."""

    hists = {}
    for env in envs:
        for i in range(len(drawing.entries)):
            hist = cause_hist(drawing, env, i)
            if hist is not None:
                hists[(env, i)] = hist
    if not hists:
        vprint(f"  No reselection for {name}, skipping.")
        return 0
    envs = [env for env in envs if any((env, i) in hists for i in range(len(drawing.entries)))]
    drawn = sorted({i for _, i in hists})
    several = len(drawn) > 1

    n_rows, n_cols = len(drawn), len(envs)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(CAUSE_HIST_W * n_cols, CAUSE_HIST_H * n_rows),
        squeeze=False,
    )
    overlays = set()
    for r, i in enumerate(drawn):
        ylabel = "Reselections / 1k Steps"
        if several:
            ylabel = f"{drawing.entries[i]['label']}\n{ylabel}"
        for c, env in enumerate(envs):
            ax = axs[r][c]
            if r == 0:
                title = drawing.env_labels.get(str(env), str(env)).replace("\n", " ")
                ax.set_title(title, fontsize=FONT_SIZE, **tex_kwargs(title))
            if (env, i) not in hists:
                ax.set_axis_off()
                continue
            twin, stats = draw_cause_hist(ax, drawing, env, i, *hists[(env, i)], errors=errors)
            overlays |= stats
            if total:
                counts, n_runs, n_steps = cause_counts(drawing, env, i)
                ax.text(
                    0.98,
                    0.95,
                    f"Total reselections:\n{100 * counts.sum() / (n_runs * n_steps):.1f}% of training steps",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=FONT_SIZE - 1,
                )
            if inner_legend and r == 0 and c == 0:
                add_cause_legend(
                    ax,
                    n_cols=1,
                    fontsize=FONT_SIZE - 4,
                    loc="upper right",
                    bbox_to_anchor=(0.99, INNER_LEGEND_TOP),
                    borderpad=0.0,
                    labelspacing=0.3,
                    handlelength=1.2,
                )
            if twin is not None and c == n_cols - 1:
                twin.set_ylabel("Errors", fontsize=FONT_SIZE)
            ax.set_ylabel(
                ylabel if c == 0 else "",
                fontsize=FONT_SIZE,
                **tex_kwargs(ylabel),
            )

    fig.subplots_adjust(hspace=0.35, wspace=0.5 if overlays else 0.15)
    if legend:
        cause_legend(fig, [ax for row in axs for ax in row if ax.axison], overlays)
    return save_and_close(fig, drawing.output_dir, name, vprint)


def draw_margin_figure(drawing, name, margin):
    """Draw, a row per entry, the histogram of cause_hist of every environment
    in the `histograms` of the plot config's `margin` block, drawn by
    draw_cause_hist, and next to them the reselections attributed to
    critic_error by steps of true progress (see progress_rates) of every
    environment in its `environments`, which maps an environment to its label.
    The progress plot has a bar per environment at every k, in the grays of
    MARGIN_GRAYS from dark to light in the order of `environments`, on a log
    scale that ends a decade above the decade of the tallest bar, and a legend
    of the environments in its upper right corner. Return the number of figures
    written."""

    hist_envs = list(margin.get("histograms") or [])
    margin_labels = dict(margin.get("environments") or {})
    rows = []
    for i in range(len(drawing.entries)):
        hists = {env: cause_hist(drawing, env, i) for env in hist_envs}
        rates = {env: progress_rates(drawing, env, i) for env in margin_labels}
        hists = {env: hist for env, hist in hists.items() if hist is not None}
        rates = {env: rate for env, rate in rates.items() if rate is not None}
        if hists or rates:
            rows.append((i, hists, rates))
    if not rows:
        vprint(f"  No reselection for {name}, skipping.")
        return 0
    several = len(rows) > 1

    n_cols = len(hist_envs) + 1
    fig, axs = plt.subplots(
        len(rows),
        n_cols,
        figsize=(MARGIN_W * n_cols, CAUSE_HIST_H * len(rows)),
        squeeze=False,
    )
    grays = [str(g) for g in np.linspace(*MARGIN_GRAYS, len(margin_labels))]
    width = 0.8 / max(len(margin_labels), 1)
    x = np.arange(MARGIN_TAIL + 1)
    overlays = set()
    for r, (i, hists, rates) in enumerate(rows):
        hist_ylabel = "Reselections / 1k Steps"
        margin_ylabel = "Approx. Error\nReselections / 1k Steps"
        if several:
            hist_ylabel = f"{drawing.entries[i]['label']}\n{hist_ylabel}"
            margin_ylabel = f"{drawing.entries[i]['label']}\n{margin_ylabel}"
        for c, env in enumerate(hist_envs):
            ax = axs[r][c]
            if r == 0:
                title = margin_labels.get(env, drawing.env_labels.get(str(env), str(env))).replace("\n", " ")
                ax.set_title(title, fontsize=MARGIN_FONT_SIZE, **tex_kwargs(title))
            if env not in hists:
                ax.set_axis_off()
                continue
            twin, stats = draw_cause_hist(ax, drawing, env, i, *hists[env])
            overlays |= stats
            if twin is not None:
                twin.set_ylabel("Errors", fontsize=MARGIN_FONT_SIZE)
            ax.set_ylabel(hist_ylabel if c == 0 else "", fontsize=MARGIN_FONT_SIZE, **tex_kwargs(hist_ylabel))
        ax = axs[r][-1]
        for j, (env, label) in enumerate(margin_labels.items()):
            if env not in rates:
                continue
            ax.bar(
                x + (j - (len(margin_labels) - 1) / 2) * width,
                rates[env],
                width=width,
                color=grays[j],
                edgecolor="black",
                linewidth=0.8,
                label=label,
            )
        ax.set_yscale("log")
        peak = max((rate.max() for rate in rates.values()), default=0.0)
        if peak > 0:
            ax.set_ylim(top=10.0 ** (np.ceil(np.log10(peak)) + 1))
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in range(MARGIN_TAIL)] + [f"$\\geq${MARGIN_TAIL}"])
        ax.xaxis.grid(False)
        ax.tick_params(axis="both", labelsize=FONT_SIZE - 2, pad=1)
        ax.set_xlabel("Net steps toward the goal ($k$)", fontsize=MARGIN_FONT_SIZE)
        ax.set_ylabel(margin_ylabel, fontsize=MARGIN_FONT_SIZE, **tex_kwargs(margin_ylabel))
        ax.legend(
            loc="upper right",
            borderaxespad=0.1,
            frameon=False,
            fontsize=MARGIN_FONT_SIZE - 2,
        )

    fig.subplots_adjust(hspace=0.35, wspace=0.65)
    cause_legend(fig, [ax for row in axs for ax in row if ax.axison], overlays)
    return save_and_close(fig, drawing.output_dir, name, vprint)


def draw_config(config_name, cfg):
    """Draw the figures of one plot config into <root>/<plot config> and return
    how many were written."""

    entries = style_entries(cfg)
    if not entries:
        print(f"WARNING: {config_name} declares no configurations to plot, skipping.")
        return 0
    groups = env_groups(cfg)
    if not groups:
        print(f"WARNING: {config_name} declares no environments to plot, skipping.")
        return 0
    _stat_labels, env_labels, _stat_ylims, _score_bounds = plot_labels(cfg)

    ignored_keys = ignored_cfg_keys(cfg)
    if ignored_keys:
        vprint(f"Keys exempt from config matching: {sorted(ignored_keys)}")
    _env, _algo, env_stem, matched = assign_groups(
        df,
        env_configs,
        algo_configs,
        ignored_keys,
        algo_decides_usable=False,
        cache=_assigned,
    )

    output_dir = os.path.join(args.folder, args.output, config_name)
    os.makedirs(output_dir, exist_ok=True)

    drawing = Drawing(
        name=config_name,
        cfg=cfg,
        entries=entries,
        env_labels=env_labels,
        ignored_keys=ignored_keys,
        matched=matched,
        env_stem=env_stem,
        output_dir=output_dir,
    )
    written = 0
    for group_name, envs in groups.items():
        vprint(
            f"\n{'=' * 60}\n"
            f"Environment group: {group_name or '(unnamed)'}\n"
            f"{'=' * 60}"
        )
        present = [e for e in envs if (drawing.env_stem == e).any()]
        absent = [e for e in envs if e not in present]
        if absent:
            print(f"WARNING: environments in {drawing.name} with no runs: {absent}")
        if not present:
            continue
        suffix = f"_{group_name}" if group_name else ""
        written += draw_cause_figure(drawing, f"reselection_causes{suffix}", present)
        written += draw_unreachable_figure(drawing, f"reselection_unreachable{suffix}", present)
        written += draw_cause_steps_figure(drawing, f"reselection_causes_steps{suffix}", present)
    for group_name, envs in (cfg.get("histograms") or {}).items():
        present = [e for e in envs if (drawing.env_stem == e).any()]
        if present:
            written += draw_cause_steps_figure(
                drawing,
                f"reselection_causes_steps_{group_name}",
                present,
                errors=False,
                legend=False,
                total=True,
                inner_legend=True,
            )
            written += write_cause_steps_legends(output_dir, f"reselection_causes_steps_{group_name}_legend")
    if cfg.get("margin"):
        written += draw_margin_figure(drawing, "reselection_margin", cfg["margin"])
    written += write_cause_legend(output_dir)
    written += write_cause_tree(output_dir)

    print(
        f"\n{drawing.name}: {len(drawing.entries)} configuration(s), "
        f"{len(groups)} environment group(s) -> {drawing.output_dir}"
    )
    return written


written = 0
for config_name, _plot_cfg_path, config in plot_cfgs:
    vprint(f"\n{'#' * 60}\nPlot config: {config_name}\n{'#' * 60}")
    written += draw_config(config_name, config)

print(
    f"\n{len(plot_cfgs)} plot config(s), {written} figure(s) -> "
    f"{os.path.join(args.folder, args.output)}"
)
