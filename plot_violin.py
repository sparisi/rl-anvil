"""Violin plots of the normalized area under the curve, from a pre-built
results.parquet, laid out by a plot config.

The plot config is read as plot_results.py reads it (see
configs/plots/example/main.yaml): the entries, environment groups, labels,
colours and score bounds are the same. One figure is written per (environment
group, statistic), named <stat>_<group>_violin, with one violin per entry, named
by its label on the x axis.

A violin is the distribution of one entry's score over the environments of the
group, with a dot per environment. An environment's score is the mean over its
runs of the normalized area under the curve, the area and the normalization
bounds being those of plot_results.py. The black bar spans the quartiles of the
environment scores and the white tick is their median, the rliable Median of
the normalized AUC (see src/utils/aggregate.py).

The legend lists the colour groups named in the plot config's `group_labels`, a
mapping from a group name under `configs` to its label, each in the group's
unshaded colour and followed by the quartile bar and the median tick. A group
missing from `group_labels` has no legend entry:

    group_labels:
      <group>: "<label>"

--no_legend and the standalone legend files work as in plot_results.py.

An environment of the group where no entry has runs for a statistic is left out
of that statistic's figure. A group with an environment that has no bounds for a
statistic is skipped for that statistic. An entry without runs in every
remaining environment is left out of the figure. An entry whose environments all
scored the same is drawn without a violin body.

Example

    python plot_violin.py -f data_example -p example -v
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
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from src.utils.aggregate import env_means, stack_runs
from src.utils.matching import ignored_cfg_keys, load_config_group
from src.utils.misc import make_vprint, warn_once
from src.utils.plot import (
    BASE_BAR_GAP,
    PLOT_CONFIG_DIR,
    STEPS_STAT,
    add_config_group_args,
    assign_groups,
    bar_positions,
    ensure_font,
    entry_rows,
    env_groups,
    hyperparameter_strings,
    load_plot_configs,
    missing_bounds,
    norm_bounds,
    plot_labels,
    rows_cols,
    run_scores,
    save_and_close,
    series_of,
    style_entries,
    tex_kwargs,
    write_legend_handles,
)

# A plot config names its configurations whatever reads best, and a console
# encoding without one of its characters raises rather than prints.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

FONT_SIZE = 12
SUBPLOT_H = 2.0

# Offsets of the environment dots across a violin: the golden-ratio sequence
# over the environment index, so the dots spread evenly and land in the same
# place on every run of the script.
GOLDEN = 0.6180339887498949

# The quartile bar and the median tick, in points, and the tick's half-length
# as a fraction of the violin width.
IQR_WIDTH = 2.0
MEDIAN_WIDTH = 2.0
MEDIAN_HALF_LENGTH = 0.12
# How much wider than the line its outline in the legend is, in points.
LEGEND_OUTLINE = 1.2

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--folder",
    required=True,
    metavar="DIR",
    help="Data directory holding the parquet file.",
)
parser.add_argument("--parquet", default="results.parquet")
parser.add_argument(
    "-p", "--plot",
    required=True,
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
         "<root>/violin/<plot config>.",
)
add_config_group_args(parser)
parser.add_argument(
    "--stats",
    nargs="+",
    default=[],
    metavar="NAME",
    help="Statistics to plot, by name and in this order. Default: the statistics "
         "the plot config labels, or every statistic in the data.",
)
parser.add_argument(
    "--violin_width",
    type=float,
    default=0.85,
    help="Violin width as a fraction of the slot it sits in.",
)
parser.add_argument(
    "--width_scale",
    type=float,
    default=1.0,
    help="Scales the width of the figures, e.g. 0.5 for a narrow one to sit "
         "beside another plot.",
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

# The answers `assign_groups` has already given, by the exemptions that produced
# them.
_assigned: dict = {}

# `steps` is the x axis of every curve, not a statistic to draw.
stats_found = sorted({c[1] for c in df.columns if c[0] == "statistic" and c != STEPS_STAT})

ensure_font()
sns.set_context("paper")
sns.set_style("darkgrid", {"legend.frameon": True})
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.axisbelow"] = False
plt.rcParams["grid.linestyle"] = "--"

written = 0


@dataclass
class Drawing:
    """Everything one plot config decides, for the functions that draw it.

    `rows` is the cache entry_rows keeps."""

    name: str
    cfg: dict
    entries: list
    score_bounds: dict
    ignored_keys: frozenset
    matched: pd.Series
    output_dir: str
    rows: dict = field(default_factory=dict)


def env_scores(drawing, group_name, present, stat):
    """Return (environments, {entry index: score per environment}) for one
    (group, statistic), the scores being None when an environment has no bounds
    for `stat`.

    The environments are those of `present` where at least one entry has runs
    for `stat`. A score is the mean over the environment's runs of their
    normalized areas (see aggregate.env_means). The bounds are the environment's
    `score_bounds` for `stat`, or its `ylim` when it declares none."""

    aucs = {}
    for i, entry in enumerate(drawing.entries):
        for env in present:
            rows = entry_rows(
                df,
                hp_strings,
                drawing.rows,
                env,
                i,
                entry,
                drawing.ignored_keys,
                drawing.matched,
            )
            decoded = series_of(rows, stat)
            runs = [] if decoded is None else run_scores(decoded, "auc")
            if runs:
                aucs[(env, i)] = runs
    envs = [e for e in present if any(env == e for env, _ in aucs)]
    unused = [e for e in present if e not in envs]
    if unused:
        print(
            f"WARNING: {drawing.name}, group {group_name or '(unnamed)'}: no "
            f"entry has runs for {stat} in {unused}; they are left out of the "
            f"violins."
        )
    if not envs:
        return envs, {}

    bounds = norm_bounds(drawing.score_bounds, envs, stat)
    if bounds is None:
        missing = missing_bounds(drawing.score_bounds, envs, stat)
        print(
            f"WARNING: {drawing.name}, group {group_name or '(unnamed)'}: no "
            f"score_bounds or ylim for {stat} in {missing}, no violins."
        )
        return envs, None
    low, high = bounds

    scores = {}
    for i, entry in enumerate(drawing.entries):
        per_env = [aucs.get((env, i), []) for env in envs]
        if not any(per_env):
            continue
        missing = [env for env, runs in zip(envs, per_env) if not runs]
        if missing:
            warn_once(
                f"WARNING: {entry['label']} has no runs for {stat} in {missing}; "
                f"it is left out of the violins."
            )
            continue
        scores[i] = env_means((stack_runs(per_env) - low) / (high - low))
    return envs, scores


def legend_handles(entries):
    """Return the legend handles of `entries`: one patch per colour group with a
    group label, named by it in its unshaded colour, then the quartile bar
    outlined in white and the median tick outlined in black."""

    groups = {
        e["group"]: (e["group_label"], e["base_color"])
        for e in entries
        if e["group_label"] is not None
    }
    handles = [
        mpatches.Patch(
            facecolor=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.8,
            label=label,
        )
        for label, color in groups.values()
    ]
    handles += [
        Line2D(
            [], [],
            color="black",
            linewidth=IQR_WIDTH,
            solid_capstyle="projecting",
            path_effects=outlined("white", IQR_WIDTH),
            label="Interquartile Range",
        ),
        Line2D(
            [], [],
            color="white",
            linewidth=MEDIAN_WIDTH,
            solid_capstyle="projecting",
            path_effects=outlined("black", MEDIAN_WIDTH),
            label="Median",
        ),
    ]
    return handles


def outlined(color, width):
    """Return path effects that draw a line of `width` points over a `color`
    stroke LEGEND_OUTLINE points wider. Both take projecting caps, so the
    outline has square corners and the same thickness at the ends as along the
    sides."""

    return [
        patheffects.Stroke(
            linewidth=width + LEGEND_OUTLINE,
            foreground=color,
            capstyle="projecting",
        ),
        patheffects.Normal(),
    ]


def draw_violins(drawing, name, ylabel, present, scores):
    """Draw one violin per entry of `scores` over the environments of `present`,
    save the figure as <name>_violin and return 1, or 0 when it cannot be
    written."""

    order = list(scores)
    shown = [drawing.entries[i] for i in order]
    width = args.violin_width
    xs = bar_positions(shown, width)

    fig, axs = plt.subplots(
        1, 1,
        figsize=(max(len(order) * 0.9, 3.0) * args.width_scale, SUBPLOT_H * 1.25),
        squeeze=False,
    )
    ax = axs[0][0]
    jitter = ((np.arange(len(present)) * GOLDEN) % 1.0 - 0.5) * 0.5 * width
    for x, i in zip(xs, order):
        entry = drawing.entries[i]
        values = scores[i]
        # A kernel density over values that are all the same is singular.
        if len(np.unique(values)) > 1:
            parts = ax.violinplot(
                [values],
                positions=[x],
                widths=width,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(entry["color"])
                body.set_edgecolor("black")
                body.set_linewidth(0.6)
                body.set_alpha(0.8)
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        ax.vlines(x, q1, q3, color="black", linewidth=IQR_WIDTH, zorder=3)
        ax.hlines(
            median,
            x - MEDIAN_HALF_LENGTH * width,
            x + MEDIAN_HALF_LENGTH * width,
            color="white",
            linewidth=MEDIAN_WIDTH,
            zorder=4,
        )
        ax.scatter(
            x + jitter,
            values,
            facecolors="white",
            edgecolors="black",
            s=14,
            alpha=0.7,
            linewidths=0.6,
            zorder=5,
        )
        vprint(
            f"  {entry['label']}: median {median:.3f}, mean {values.mean():.3f} "
            f"over {len(values)} environment(s)"
        )

    slot = width + BASE_BAR_GAP
    ax.set_xlim(xs[0] - slot / 2, xs[-1] + slot / 2)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [e["label"] for e in shown],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    for text in ax.get_xticklabels():
        text.set(**tex_kwargs(text.get_text()))
    ax.tick_params(axis="x", labelsize=FONT_SIZE - 2, pad=1)
    ax.tick_params(axis="y", labelsize=FONT_SIZE - 2, pad=1)
    label = f"Normalized AUC {ylabel}"
    ax.set_ylabel(label, fontsize=FONT_SIZE, **tex_kwargs(label))
    # Round ticks (steps of 1, 2, 2.5 or 5 times a power of ten), drawn only
    # within the axis limits.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.grid(False)

    if not args.no_legend:
        handles = legend_handles(shown)
        legend = ax.legend(
            handles=handles,
            fontsize=FONT_SIZE - 2,
            frameon=True,
            loc="lower center",
            ncol=rows_cols(drawing.cfg.get("legend_rows_cols"), len(handles))[1],
            bbox_to_anchor=(0.5, 1.02),
        )
        for text in legend.get_texts():
            text.set(**tex_kwargs(text.get_text()))
    return save_and_close(fig, drawing.output_dir, f"{name}_violin", vprint)


def draw_config(config_name, cfg):
    """Draw one plot config's figures, into its own directory, and return how
    many it wrote."""

    entries = style_entries(cfg)
    if not entries:
        print(f"WARNING: {config_name} declares no configurations to plot, skipping.")
        return 0
    groups = env_groups(cfg)
    if not groups:
        print(f"WARNING: {config_name} declares no environments to plot, skipping.")
        return 0
    stat_labels, _env_labels, _stat_ylims, score_bounds = plot_labels(cfg)

    wanted = list(args.stats) or list(stat_labels) or stats_found
    missing = [s for s in wanted if s not in stats_found and s != STEPS_STAT[1]]
    if missing:
        print(f"WARNING: {config_name} asks for statistics not in the data: {missing}")
    stats = [s for s in wanted if s in stats_found]
    if not stats:
        print(f"WARNING: {config_name} has no statistic to plot, skipping.")
        return 0
    vprint(f"Statistics: {stats}")

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

    output_dir = os.path.join(args.folder, args.output, "violin", config_name)
    os.makedirs(output_dir, exist_ok=True)

    drawing = Drawing(
        name=config_name,
        cfg=cfg,
        entries=entries,
        score_bounds=score_bounds,
        ignored_keys=ignored_keys,
        matched=matched,
        output_dir=output_dir,
    )
    written = 0

    for group_name, envs in groups.items():
        vprint(
            f"\n{'=' * 60}\n"
            f"Environment group: {group_name or '(unnamed)'}\n"
            f"{'=' * 60}"
        )
        present = [e for e in envs if (env_stem == e).any()]
        absent = [e for e in envs if e not in present]
        if absent:
            print(f"WARNING: environments in {config_name} with no runs: {absent}")
        if not present:
            continue

        for stat in stats:
            envs_drawn, scores = env_scores(drawing, group_name, present, stat)
            if not scores:
                vprint(f"  Nothing to draw for {stat}, skipping.")
                continue
            name = f"{stat}_{group_name}" if group_name else stat
            written += draw_violins(drawing, name, stat_labels.get(stat, stat), envs_drawn, scores)

    for path in write_legend_handles(legend_handles(entries), cfg, output_dir, FONT_SIZE):
        vprint(f"  Saved: {path}")
    print(
        f"\n{config_name}: {len(entries)} configuration(s), "
        f"{len(groups)} environment group(s) -> {output_dir}"
    )
    return written


for config_name, plot_cfg_path, config in plot_cfgs:
    vprint(f"\n{'#' * 60}\nPlot config: {config_name}\n{'#' * 60}")
    written += draw_config(config_name, config)

print(
    f"\n{len(plot_cfgs)} plot config(s), {written} figure(s) -> "
    f"{os.path.join(args.folder, args.output, 'violin')}"
)
