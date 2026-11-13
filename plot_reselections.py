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
                    (critic error). Each reselection has one cause (see
                    src.utils.reachability.reselection_causes).
  reselection_causes_<group>.tex
                    the same numbers as a booktabs LaTeX tabular: the number
                    of reselections and the percentage of each cause per bar.
  reselection_causes_steps_<group>.png
                    the same reselections over training, a row per entry and a
                    column per environment with reselections: a histogram in
                    --cause_bins equal-width bins of steps, each bar the mean
                    number of reselections per seed per 1,000 training steps
                    in the bin, stacked by cause in the same colours. Its axes
                    carry round ticks (see round_ticks).

The legend of the causes is also written alone, once per plot config, as
reselection_causes_legend.png in the same directory, whatever --no_legend says.

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
import re
import sys
from ast import literal_eval
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

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
    style_entries,
    tex_kwargs,
)
from src.utils.reachability import ORACLE_TOL, RESELECTION_CAUSES, reselection_causes
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
# and the true visit Q-functions built (see true_visit_of).
_memory_causes: dict = {}
_true_visit: dict = {}

# Label, colour and label colour of each cause of RESELECTION_CAUSES, in its
# order: reds for the goals that were or became unreachable, blues for the true
# value lost while acting, grey for the value checks that fired without a drop.
CAUSE_STYLE = {
    "unreachable_at_selection": ("Unreachable at selection", "#b2182b", "white"),
    "became_unreachable": ("Became unreachable", "#ef8a62", "black"),
    "random_action": ("Random action", "#d1e5f0", "black"),
    "suboptimal_action": ("Suboptimal action", "#4393c3", "white"),
    "env_noise": ("Env. noise", "#2166ac", "white"),
    "critic_error": ("Critic error", "#bababa", "black"),
}
# A segment of a bar narrower than this fraction is drawn without its
# percentage.
CAUSE_LABEL_MIN = 0.04
# Inches per bar of reselection_causes, and the width of the figure.
CAUSE_BAR_H = 0.55
CAUSE_FIG_W = 8.0
# Inches per subplot of reselection_causes_steps, in width and in height.
CAUSE_HIST_W = 3.9
CAUSE_HIST_H = 2.0
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
    """Return the cause of every reselection in the memory dump `mem_file` of a
    run in `run_dir`, as reselection_causes returns them, or None when the dump
    or the run's cfg.yaml cannot be read, the dump lacks a key of MEMORY_KEYS,
    the environment has no tabular counter, or true_visit_of gives no true
    visit Q-function."""

    if not os.path.isfile(mem_file):
        return None
    _, run_cfg, reason = read_run_cfg(os.path.join(run_dir, "cfg.yaml"))
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
    return reselection_causes(
        true_visit,
        np.asarray(counter.bin_index(mem["obs"])).reshape(size),
        np.asarray(counter.bin_index(mem["goal_obs"])).reshape(size),
        flat["act"].astype(np.int64),
        flat["goal_act"].astype(np.int64),
        flat["has_goal"].astype(bool),
        flat["act_source"],
        flat["goal_selection"].astype(bool),
        flat["term"].astype(bool) | flat["trunc"].astype(bool),
        ORACLE_TOL,
    )


def memory_causes(config_id, seed):
    """Return read_memory_causes for the run <config_id>/<seed> of the data
    directory, reading its dump once per process."""

    run_dir = os.path.join(args.folder, str(config_id))
    mem_file = os.path.join(run_dir, str(int(seed)), "memory.npz")
    if mem_file not in _memory_causes:
        _memory_causes[mem_file] = read_memory_causes(run_dir, mem_file)
    return _memory_causes[mem_file]


def cause_runs(drawing, env, index):
    """Return the reselection causes of every run entry `index` selects in
    `env`, one int array over the steps of its memory dump per run (see
    reselection_causes), leaving out the runs without them (see
    read_memory_causes)."""

    rows = rows_of(drawing, env, index)
    if any(c not in rows.columns for c in (CONFIG_ID, SEED)):
        return []
    runs = []
    for config_id, seed in rows[[CONFIG_ID, SEED]].drop_duplicates().itertuples(index=False):
        causes = memory_causes(config_id, seed)
        if causes is not None:
            runs.append(causes)
    return runs


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


def add_cause_legend(fig, **kwargs):
    """Draw the legend of the causes of CAUSE_STYLE on `fig`, three columns
    without a frame, placed by the fig.legend keyword arguments `kwargs`."""

    fig.legend(
        [
            mpatches.Patch(facecolor=color, edgecolor="white")
            for _label, color, _text in CAUSE_STYLE.values()
        ],
        [label for label, _color, _text in CAUSE_STYLE.values()],
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZE - 2,
        **kwargs,
    )


def cause_legend(fig, axes):
    """Draw the legend of the causes of CAUSE_STYLE under where `axes` and their
    tick labels end, measured, unless --no_legend."""

    if args.no_legend:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottom = min(ax.get_tightbbox(renderer).y0 for ax in axes)
    top = fig.transFigure.inverted().transform((0, bottom))[1] - 0.02
    add_cause_legend(fig, loc="upper center", bbox_to_anchor=(0.5, top))


def write_cause_legend(output_dir):
    """Write the legend of the causes of CAUSE_STYLE alone, as the figures draw
    it, to reselection_causes_legend.png in `output_dir`, and return the number
    of figures written."""

    fig = plt.figure(figsize=(CAUSE_FIG_W, 0.8))
    add_cause_legend(fig, loc="center")
    return save_and_close(fig, output_dir, "reselection_causes_legend", vprint)


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


def tex_text(text):
    """Return `text` as LaTeX, with `_`, `&`, `%` and `#` outside `$...$`
    escaped unless a backslash already precedes them. Text between two `$` is
    left as it is."""

    parts = text.split("$")
    for k in range(0, len(parts), 2):
        parts[k] = re.sub(r"(?<!\\)([_&%#])", r"\\\1", parts[k])
    return "$".join(parts)


def write_cause_table(drawing, name, bars, several):
    """Write the reselection causes of `bars`, as (environment, entry index,
    counts per cause of RESELECTION_CAUSES), to <name>.tex in the output
    directory: a booktabs tabular with a row per bar, its environment (and its
    entry when `several`), its number of reselections and the percentage of
    each cause, one decimal each, under the labels of CAUSE_STYLE."""

    header = ["Environment"] + (["Method"] if several else []) + ["$n$"]
    header += [tex_text(CAUSE_STYLE[c][0]) for c in RESELECTION_CAUSES]
    lines = [
        r"\begin{tabular}{l" + ("l" if several else "") + "r" * (len(RESELECTION_CAUSES) + 1) + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for env, i, counts in bars:
        total = int(counts.sum())
        cells = [tex_text(drawing.env_labels.get(str(env), str(env)))]
        if several:
            cells.append(tex_text(drawing.entries[i]["label"]))
        cells.append(f"{total:,}".replace(",", "{,}"))
        cells += [f"{100 * n / total:.1f}" for n in counts]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = os.path.join(drawing.output_dir, f"{name}.tex")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    vprint(f"  Saved: {path}")


def draw_cause_figure(drawing, name, envs):
    """Draw the causes of the reselections as a 100% stacked horizontal bar per
    (environment, entry) of `envs` with at least one reselection, write the same
    numbers to <name>.tex (see write_cause_table), and return the number of
    figures written.

    Every reselection has one cause (see reselection_causes), and the segments
    of a bar are the fractions of its reselections of each cause, pooled over
    the seeds, in the order and the colours of CAUSE_STYLE. A segment at least
    CAUSE_LABEL_MIN wide is labelled with its percentage. At the end of the bar,
    "total = <reselections>" is written over "(<x>% of <training steps>)", x
    being the mean number of reselections per seed as a percentage of the
    training steps. A bar is named by its environment, and by its entry too
    when several entries have bars."""

    bars, runs_steps = [], []
    for env in envs:
        for i in range(len(drawing.entries)):
            counted = cause_counts(drawing, env, i)
            if counted is not None and counted[0].sum() > 0:
                bars.append((env, i, counted[0]))
                runs_steps.append(counted[1:])
    if not bars:
        vprint(f"  No reselection for {name}, skipping.")
        return 0
    several = len({i for _, i, _ in bars}) > 1
    write_cause_table(drawing, name, bars, several)

    fig, ax = plt.subplots(figsize=(CAUSE_FIG_W, CAUSE_BAR_H * len(bars) + 0.6))
    labels = []
    for y, ((env, i, counts), (n_runs, n_steps)) in enumerate(zip(bars, runs_steps)):
        label = drawing.env_labels.get(str(env), str(env))
        if several:
            label = f"{label}\n{drawing.entries[i]['label']}"
        labels.append(label)
        total = int(counts.sum())
        left = 0.0
        for cause, n in zip(RESELECTION_CAUSES, counts):
            frac = n / total
            _label, color, text_color = CAUSE_STYLE[cause]
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
            f"total = {total:,}\n({100 * total / (n_runs * n_steps):.1f}% of {n_steps:,})",
            ha="left",
            va="center",
            fontsize=FONT_SIZE - 3,
            clip_on=False,
        )
        vprint(
            f"  {label.replace(chr(10), ' / ')}: {total} reselections, "
            + ", ".join(f"{c} {100 * n / total:.1f}%" for c, n in zip(RESELECTION_CAUSES, counts))
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
    ax.set_title("Causes of Goal Reselections", fontsize=FONT_SIZE)

    cause_legend(fig, [ax])
    return save_and_close(fig, drawing.output_dir, name, vprint)


def draw_cause_steps_figure(drawing, name, envs):
    """Draw when the reselections happened, a row per entry and a column per
    environment of `envs` with at least one reselection, and return the number
    of figures written.

    A subplot is a histogram over the training steps in --cause_bins
    equal-width bins of the steps of the memory dumps: the height of a bar is
    the mean number of reselections per seed per 1,000 training steps in the
    bin, stacked by cause in the order and the colours of CAUSE_STYLE. Both axes
    carry round ticks (see round_ticks). A subplot with no reselection is left
    blank."""

    hists = {}
    for env in envs:
        for i in range(len(drawing.entries)):
            runs = cause_runs(drawing, env, i)
            if not runs or not any((c >= 0).any() for c in runs):
                continue
            n_steps = max(len(c) for c in runs)
            edges = np.linspace(0.0, n_steps, args.cause_bins + 1)
            heights = np.zeros((len(RESELECTION_CAUSES), args.cause_bins))
            for causes in runs:
                steps = np.flatnonzero(causes >= 0)
                for c in range(len(RESELECTION_CAUSES)):
                    heights[c] += np.histogram(steps[causes[steps] == c], bins=edges)[0]
            hists[(env, i)] = (edges, 1000.0 * heights / len(runs) / np.diff(edges))
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
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    for r, i in enumerate(drawn):
        ylabel = "Reselections / 1k Steps"
        if several:
            ylabel = f"{drawing.entries[i]['label']}\n{ylabel}"
        for c, env in enumerate(envs):
            ax = axs[r][c]
            if r == 0:
                title = drawing.env_labels.get(str(env), str(env))
                ax.set_title(title, fontsize=FONT_SIZE, **tex_kwargs(title))
            if (env, i) not in hists:
                ax.set_axis_off()
                continue
            edges, heights = hists[(env, i)]
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
            ax.set_xlim(edges[0], edges[-1])
            round_ticks(ax, "x", STEP_TICK_BINS)
            ax.xaxis.set_major_formatter(lambda v, _pos: f"{v / 1000:g}k")
            ax.tick_params(axis="x", labelsize=FONT_SIZE - 2, pad=-2)
            ax.tick_params(axis="y", labelsize=FONT_SIZE - 2, pad=1)
            ax.set_ylim(bottom=0.0)
            round_ticks(ax, "y", VALUE_TICK_BINS)
            ax.set_ylabel(
                ylabel if c == 0 else "",
                fontsize=FONT_SIZE,
                **tex_kwargs(ylabel),
            )

    cause_legend(fig, [ax for row in axs for ax in row if ax.axison])
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
        written += draw_cause_steps_figure(drawing, f"reselection_causes_steps{suffix}", present)
    written += write_cause_legend(output_dir)

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
