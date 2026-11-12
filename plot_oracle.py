"""Oracle diagnostic figures from a pre-built results.parquet, laid out by a plot
config.

Runs launched with `results.true_visit_check=True` record diagnostics against
the exact visit Q-function of the environment (src/oracle.py). The plot config
names the configurations and the environments to draw, as for plot_results.py,
and labels the statistics (see configs/plots/gcrl_oracle/oracle.yaml). Runs are
selected as plot_results.py selects them: a run is drawn only when its whole
configuration is the one its entry composes to. Every figure is written once
per environment group, as <figure>_<group>.png, in
<root>/oracle/<plot config>.

  selection         a column per environment and a row per statistic of goal
                    selection, a curve per entry: how often the selected goal
                    has the best true score, the true score it loses to the best
                    one, and the true visit value of the selected goal (solid)
                    and of the goal with the best true score (dotted).
  reselection       the same layout for the reselections: the fraction caused by
                    a random action, by a suboptimal action, by a stochastic
                    transition and by an error of the learned visit Q-function,
                    and the fraction in which the true scores keep the goal held.
  reselection_bars  a row per environment and a column per cause of
                    reselection, a bar per entry: the area under each seed's
                    curve divided by the span of its recorded steps, as its mean
                    over seeds and its 95% interval.
  q_error           the same layout as `selection` for the absolute error
                    between the true and the learned visit Q-function on the
                    update batch: over all actions, at the batch action, and at
                    the batch action for the positive and the negative goals.

The causes of a reselection are not exclusive: a reselection preceded by a
random action and by a suboptimal one counts for both. A subplot draws the
entries that recorded its statistic, and a row no entry recorded is left out.

Run with --help for the rest of the options.

Example

    python plot_oracle.py -f data_gcrl_oracle -v
"""

import argparse
import os
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

from src.utils.matching import ignored_cfg_keys, load_config_group
from src.utils.misc import make_vprint
from src.utils.plot import (
    PLOT_CONFIG_DIR,
    add_config_group_args,
    assign_groups,
    draw_bars,
    ensure_font,
    entry_rows,
    env_groups,
    error_shade_plot,
    hyperparameter_strings,
    load_plot_configs,
    plot_labels,
    rows_cols,
    run_scores,
    save_and_close,
    series_of,
    set_3_ticks,
    style_entries,
    summarize,
    tex_kwargs,
    write_legend,
)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

FONT_SIZE = 12
SUBPLOT_W = 2.6  # inches per subplot
SUBPLOT_H = 2.0

# Linestyle of the second statistic of a row, drawn over the first one in the
# entry's colour.
OVERLAY_LINESTYLE = ":"

# Rows of each curve figure, as (label key, statistics). The label key names
# the y-axis label in the plot config's `statistics`. A row holding two
# statistics draws the second one in OVERLAY_LINESTYLE.
CURVE_FIGURES = {
    "selection": [
        ("oracle_agree_%", ["oracle_agree_%"]),
        ("oracle_regret", ["oracle_regret"]),
        ("true_q", ["true_q_sel", "true_q_oracle"]),
    ],
    "reselection": [
        ("resel_random_%", ["resel_random_%"]),
        ("resel_suboptimal_%", ["resel_suboptimal_%"]),
        ("resel_stochastic_%", ["resel_stochastic_%"]),
        ("resel_value_error_%", ["resel_value_error_%"]),
        ("resel_oracle_keep_%", ["resel_oracle_keep_%"]),
    ],
    "q_error": [
        ("q_visit_abs_err", ["q_visit_abs_err"]),
        ("q_visit_abs_err_act", ["q_visit_abs_err_act"]),
        ("q_visit_abs_err_pos", ["q_visit_abs_err_pos"]),
        ("q_visit_abs_err_neg", ["q_visit_abs_err_neg"]),
    ],
}

# Columns of each bar figure, one statistic each.
BAR_FIGURES = {
    "reselection_bars": [
        "resel_random_%",
        "resel_suboptimal_%",
        "resel_stochastic_%",
        "resel_value_error_%",
    ],
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--folder",
    required=True,
    metavar="DIR",
    help="Data directory holding the parquet file.",
)
parser.add_argument("--parquet", default="results.parquet")
parser.add_argument(
    "-p", "--plot_config",
    default="gcrl_oracle/oracle",
    metavar="NAME",
    help=f"Plot config: which configurations to draw, and over which environments. "
         f"A name without .yaml is looked for in {PLOT_CONFIG_DIR}, or give a path; "
         f"a directory is a set of configs and every one of them is drawn, bar "
         f"those marked `# skip`.",
)
parser.add_argument(
    "-o", "--output",
    default="plots",
    help="Output root, under the data directory. Figures go in "
         "<root>/oracle/<plot config>.",
)
add_config_group_args(parser)
parser.add_argument(
    "--smoothing_window",
    type=int,
    default=0,
    help="Points either side to average a curve over. 0 draws it as recorded.",
)
parser.add_argument(
    "--bar_width",
    type=float,
    default=0.85,
    help="Bar width as a fraction of the slot it sits in.",
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

plot_cfgs = load_plot_configs(args.plot_config)
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
    the runs the entry selects there, and `curves` maps (environment, statistic, entry index) to the decoded
    curves of those runs, or None when they recorded none."""

    name: str
    cfg: dict
    entries: list
    stat_labels: dict
    env_labels: dict
    stat_ylims: dict
    ignored_keys: frozenset
    matched: pd.Series
    env_stem: pd.Series
    output_dir: str
    rows: dict = field(default_factory=dict)
    curves: dict = field(default_factory=dict)


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


def curve_of(drawing, env, stat, index):
    """Return the decoded curves of `stat` for entry `index` in `env`, as
    series_of returns them, or None when its runs recorded none."""

    key = (env, stat, index)
    if key not in drawing.curves:
        drawing.curves[key] = series_of(rows_of(drawing, env, index), stat)
    return drawing.curves[key]


def legend_on(drawing, fig, handles, offset):
    """Draw the legend of `handles` below `fig`, unless --no_legend."""

    if args.no_legend or not handles:
        return
    legend = fig.legend(
        handles,
        [h.get_label() for h in handles],
        fontsize=FONT_SIZE - 2,
        frameon=True,
        loc="lower center",
        ncol=rows_cols(drawing.cfg.get("legend_rows_cols"), len(handles))[1],
        bbox_to_anchor=(0.5, offset),
    )
    for text in legend.get_texts():
        text.set(**tex_kwargs(text.get_text()))


def draw_curve_figure(drawing, name, rows, envs):
    """Draw one curve figure, a row per (label key, statistics) of `rows` and a
    column per environment of `envs`, and return the number of figures written.
    """

    rows = [
        (key, stats) for key, stats in rows
        if any(
            curve_of(drawing, env, stat, i) is not None
            for env in envs
            for stat in stats
            for i in range(len(drawing.entries))
        )
    ]
    if not rows:
        vprint(f"  Nothing recorded for {name}, skipping.")
        return 0

    n_rows, n_cols = len(rows), len(envs)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(SUBPLOT_W * n_cols, SUBPLOT_H * n_rows),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    drawn, overlays = set(), {}
    for r, (key, stats) in enumerate(rows):
        ylabel = drawing.stat_labels.get(key, key)
        for c, env in enumerate(envs):
            ax = axs[r][c]
            if r == 0:
                title = drawing.env_labels.get(str(env), str(env))
                ax.set_title(title, fontsize=FONT_SIZE, **tex_kwargs(title))

            x_lo, x_hi = np.inf, 0.0
            for s, stat in enumerate(stats):
                for i, entry in enumerate(drawing.entries):
                    decoded = curve_of(drawing, env, stat, i)
                    if decoded is None:
                        continue
                    series, x, steps, _ = decoded
                    error_shade_plot(
                        ax,
                        series.to_numpy().T,
                        x=x,
                        smoothing_window=args.smoothing_window,
                        color=entry["color"],
                        linestyle=entry["linestyle"] if s == 0 else OVERLAY_LINESTYLE,
                        linewidth=2.0,
                    )
                    drawn.add(i)
                    if len(stats) > 1:
                        overlays.setdefault(stat, entry["linestyle"] if s == 0 else OVERLAY_LINESTYLE)
                    x_lo = min(x_lo, float(x[0]))
                    x_hi = max(x_hi, steps)

            ax.tick_params(axis="x", labelsize=FONT_SIZE - 2, pad=-2)
            ax.tick_params(axis="y", labelsize=FONT_SIZE - 2, pad=1)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(3, 3))
            ax.xaxis.offsetText.set_visible(False)
            ax.set_ylabel(
                ylabel if c == 0 else "",
                fontsize=FONT_SIZE,
                **tex_kwargs(ylabel),
            )
            if x_hi > x_lo:
                ax.set_xlim(x_lo, x_hi)
                ax.set_xticks([x_lo, (x_lo + x_hi) / 2, x_hi])
            bounds = drawing.stat_ylims.get((str(env), stats[0]))
            if bounds:
                ax.set_ylim(*bounds)
            set_3_ticks(ax, which="y")

    handles = [
        plt.Line2D(
            [], [],
            color=drawing.entries[i]["color"],
            linestyle=drawing.entries[i]["linestyle"],
            linewidth=2.0,
            label=drawing.entries[i]["label"],
        )
        for i in sorted(drawn)
    ] + [
        plt.Line2D(
            [], [],
            color="black",
            linestyle=linestyle,
            linewidth=2.0,
            label=drawing.stat_labels.get(stat, stat),
        )
        for stat, linestyle in overlays.items()
    ]
    legend_on(drawing, fig, handles, -0.15 / n_rows)
    return save_and_close(fig, drawing.output_dir, name, vprint)


def draw_bar_figure(drawing, name, stats, envs):
    """Draw one bar figure, a row per environment of `envs` and a column per
    statistic of `stats`, and return the number of figures written."""

    summaries = {
        (env, stat, i): summarize(run_scores(decoded, "auc"))
        for env in envs
        for stat in stats
        for i in range(len(drawing.entries))
        if (decoded := curve_of(drawing, env, stat, i)) is not None
    }
    stats = [s for s in stats if any(k[1] == s for k in summaries)]
    if not stats:
        vprint(f"  Nothing recorded for {name}, skipping.")
        return 0
    drawn = sorted({i for _, _, i in summaries})

    n_rows, n_cols = len(envs), len(stats)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(SUBPLOT_W * n_cols, SUBPLOT_H * n_rows),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    for r, env in enumerate(envs):
        ylabel = drawing.env_labels.get(str(env), str(env))
        for c, stat in enumerate(stats):
            ax = axs[r][c]
            if r == 0:
                title = drawing.stat_labels.get(stat, stat)
                ax.set_title(title, fontsize=FONT_SIZE, **tex_kwargs(title))
            draw_bars(
                ax,
                [(drawing.entries[i], summaries.get((env, stat, i))) for i in drawn],
                ylabel=ylabel if c == 0 else "",
                bar_width=args.bar_width,
                font_size=FONT_SIZE,
                annotate=True,
            )
            set_3_ticks(ax, which="y")

    legend_on(
        drawing,
        fig,
        [
            mpatches.Patch(
                facecolor=drawing.entries[i]["color"],
                hatch=drawing.entries[i]["hatch"],
                edgecolor="black",
                linewidth=0.6,
                label=drawing.entries[i]["label"],
            )
            for i in drawn
        ],
        -0.1 / n_rows,
    )
    return save_and_close(fig, drawing.output_dir, name, vprint)


def draw_config(config_name, cfg):
    """Draw the figures of one plot config into its own directory and return
    how many were written."""

    entries = style_entries(cfg)
    if not entries:
        print(f"WARNING: {config_name} declares no configurations to plot, skipping.")
        return 0
    groups = env_groups(cfg)
    if not groups:
        print(f"WARNING: {config_name} declares no environments to plot, skipping.")
        return 0
    stat_labels, env_labels, stat_ylims, _score_bounds = plot_labels(cfg)

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

    output_dir = os.path.join(args.folder, args.output, "oracle", config_name)
    os.makedirs(output_dir, exist_ok=True)

    drawing = Drawing(
        name=config_name,
        cfg=cfg,
        entries=entries,
        stat_labels=stat_labels,
        env_labels=env_labels,
        stat_ylims=stat_ylims,
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
        for figure, rows in CURVE_FIGURES.items():
            written += draw_curve_figure(drawing, f"{figure}{suffix}", rows, present)
        for figure, stats in BAR_FIGURES.items():
            written += draw_bar_figure(drawing, f"{figure}{suffix}", stats, present)

    for path in write_legend(drawing.entries, drawing.cfg, drawing.output_dir, FONT_SIZE):
        vprint(f"  Saved: {path}")
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
    f"{os.path.join(args.folder, args.output, 'oracle')}"
)
