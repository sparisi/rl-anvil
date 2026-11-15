"""Violin plots of the normalized area under the curve, from a pre-built
results.parquet, with the environments of a second results.parquet, the JaxGCRL
runs, added to every violin.

Both data directories are read as plot_violin.py reads one, and their runs are
matched against the same environment and algorithm YAMLs. The plot config of
`-p` lays out the figures as in plot_violin.py: one figure per (environment
group, statistic), named <stat>_<group>_violin, with one violin per entry, named
by its label on the x axis. The plot config of `--gcrl_plot` selects the
JaxGCRL runs: its entries, environments, statistics and score bounds apply to
the data of `--gcrl_folder`, and its environments in every group are added to
every group of `-p`.

A JaxGCRL entry is paired with the entry of `-p` that has the same colour group
and label, and a JaxGCRL statistic with the statistic of `-p` that has the same
label. A violin is the distribution of one entry's score over the environments
of the group and the JaxGCRL environments, with a dot per environment. An
environment's score is the mean over its runs of the normalized area under the
curve, the area and the normalization bounds being those of plot_results.py.
The black bar spans the quartiles of the environment scores and the white tick
is their median.

The legend lists the colour groups named in the plot config's `group_labels`,
as in plot_violin.py. --no_legend and the standalone legend files work as in
plot_results.py.

An environment where no entry has runs for a statistic is left out of that
statistic's figure. A group with an environment that has no bounds for a
statistic is skipped for that statistic, and so is every group when a JaxGCRL
environment has none. A statistic of `-p` whose label no JaxGCRL statistic has
is skipped. An entry without runs in every remaining environment of either data
directory, or without a JaxGCRL pair, is left out of the figure. An entry whose
environments all scored the same is drawn without a violin body.

Example

    python plot_violin_gcrl.py -f data_gcrl -p gcrl/ablation_beta -v
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

# The JaxGCRL plot data in rl-anvil's layout.
GCRL_DIR = "data_gcrl/sun_gcrl_beta_ablation_plot_data_2026-11-14/rl_anvil"

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
    "--gcrl_folder",
    default=f"{GCRL_DIR}/data_f24_logratio",
    metavar="DIR",
    help="Data directory holding the parquet file of the JaxGCRL runs.",
)
parser.add_argument(
    "--gcrl_plot",
    default="gcrl_jax/ablation_beta_sun",
    metavar="NAME",
    help=f"Plot config selecting the JaxGCRL runs. A name without .yaml is "
         f"looked for in {PLOT_CONFIG_DIR}, or give the path of a file.",
)
parser.add_argument(
    "-o", "--output",
    default="plots",
    help="Output root, under the data directory. Figures go in "
         "<root>/violin_gcrl/<plot config>.",
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
gcrl_cfg = load_plot_configs(args.gcrl_plot)[0][2]


@dataclass
class Source:
    """One data directory: its runs, every hyperparameter column as value_str
    renders it, the config groups its runs are matched against, and its
    statistics.

    `assigned` keeps the answers `assign_groups` has already given, by the
    exemptions that produced them."""

    df: pd.DataFrame
    hp_strings: dict
    env_configs: dict
    algo_configs: dict
    stats: list
    assigned: dict = field(default_factory=dict)


def load_source(folder, environments, algorithms):
    """Return the Source of the results.parquet in `folder`, matched against the
    YAMLs of `environments` and `algorithms`."""

    path = os.path.join(folder, args.parquet)
    frame = pd.read_parquet(path)
    vprint(f"\nLoaded parquet file from: {path}")
    if all(isinstance(c, str) and c.startswith("(") for c in frame.columns):
        frame.columns = pd.MultiIndex.from_tuples([literal_eval(c) for c in frame.columns])
    return Source(
        df=frame,
        hp_strings=hyperparameter_strings(frame),
        env_configs=load_config_group(environments) if environments else {},
        algo_configs=load_config_group(algorithms) if algorithms else {},
        # `steps` is the x axis of every curve, not a statistic to draw.
        stats=sorted({c[1] for c in frame.columns if c[0] == "statistic" and c != STEPS_STAT}),
    )


main_source = load_source(args.folder, args.environments, args.algorithms)
gcrl_source = load_source(args.gcrl_folder, args.environments, args.algorithms)

ensure_font()
sns.set_context("paper")
sns.set_style("darkgrid", {"legend.frameon": True})
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.axisbelow"] = False
plt.rcParams["grid.linestyle"] = "--"

written = 0


@dataclass
class Drawing:
    """Everything one plot config decides over one Source, for the functions
    that draw it.

    `env_stem` is the environment each run of the source was matched to, and
    `rows` is the cache entry_rows keeps."""

    name: str
    cfg: dict
    source: Source
    entries: list
    stat_labels: dict
    score_bounds: dict
    ignored_keys: frozenset
    env_stem: pd.Series
    matched: pd.Series
    output_dir: str
    rows: dict = field(default_factory=dict)


def make_drawing(name, cfg, source, output_dir):
    """Return the Drawing of plot config `cfg`, named `name`, over `source`."""

    stat_labels, _env_labels, _stat_ylims, score_bounds = plot_labels(cfg)
    ignored_keys = ignored_cfg_keys(cfg)
    if ignored_keys:
        vprint(f"Keys exempt from config matching: {sorted(ignored_keys)}")
    _env, _algo, env_stem, matched = assign_groups(
        source.df,
        source.env_configs,
        source.algo_configs,
        ignored_keys,
        algo_decides_usable=False,
        cache=source.assigned,
    )
    return Drawing(
        name=name,
        cfg=cfg,
        source=source,
        entries=style_entries(cfg),
        stat_labels=stat_labels,
        score_bounds=score_bounds,
        ignored_keys=ignored_keys,
        env_stem=env_stem,
        matched=matched,
        output_dir=output_dir,
    )


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
                drawing.source.df,
                drawing.source.hp_strings,
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
    """Draw one plot config's figures, with the JaxGCRL environments added to
    every violin, into its own directory, and return how many it wrote."""

    if not style_entries(cfg):
        print(f"WARNING: {config_name} declares no configurations to plot, skipping.")
        return 0
    groups = env_groups(cfg)
    if not groups:
        print(f"WARNING: {config_name} declares no environments to plot, skipping.")
        return 0

    output_dir = os.path.join(args.folder, args.output, "violin_gcrl", config_name)
    os.makedirs(output_dir, exist_ok=True)
    drawing = make_drawing(config_name, cfg, main_source, output_dir)
    gcrl = make_drawing(f"{config_name} (JaxGCRL)", gcrl_cfg, gcrl_source, output_dir)
    entries = drawing.entries

    wanted = list(args.stats) or list(drawing.stat_labels) or main_source.stats
    missing = [s for s in wanted if s not in main_source.stats and s != STEPS_STAT[1]]
    if missing:
        print(f"WARNING: {config_name} asks for statistics not in the data: {missing}")
    stats = [s for s in wanted if s in main_source.stats]
    if not stats:
        print(f"WARNING: {config_name} has no statistic to plot, skipping.")
        return 0
    vprint(f"Statistics: {stats}")

    # The JaxGCRL entry and statistic of each entry and statistic of `cfg`, by
    # colour group and label.
    gcrl_index = {(e["group"], e["label"]): j for j, e in enumerate(gcrl.entries)}
    pairs = {
        i: gcrl_index[(e["group"], e["label"])]
        for i, e in enumerate(entries)
        if (e["group"], e["label"]) in gcrl_index
    }
    unpaired = [e["label"] for i, e in enumerate(entries) if i not in pairs]
    if unpaired:
        print(
            f"WARNING: {config_name}: no JaxGCRL entry with the group and label of "
            f"{unpaired}; they are left out of the violins."
        )
    gcrl_stats = {label: s for s, label in gcrl.stat_labels.items()}
    gcrl_envs = [e for envs in env_groups(gcrl_cfg).values() for e in envs]
    gcrl_present = [e for e in gcrl_envs if (gcrl.env_stem == e).any()]
    gcrl_absent = [e for e in gcrl_envs if e not in gcrl_present]
    if gcrl_absent:
        print(f"WARNING: JaxGCRL environments with no runs: {gcrl_absent}")

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
            print(f"WARNING: environments in {config_name} with no runs: {absent}")
        if not present:
            continue

        for stat in stats:
            ylabel = drawing.stat_labels.get(stat, stat)
            gcrl_stat = gcrl_stats.get(ylabel)
            if gcrl_stat is None:
                print(
                    f"WARNING: {config_name}: no JaxGCRL statistic labelled "
                    f"{ylabel!r}, {stat} is skipped."
                )
                continue
            envs_drawn, scores = env_scores(drawing, group_name, present, stat)
            gcrl_drawn, gcrl_scores = env_scores(gcrl, "JaxGCRL", gcrl_present, gcrl_stat)
            if not scores or not gcrl_scores:
                vprint(f"  Nothing to draw for {stat}, skipping.")
                continue
            combined = {}
            for i, values in scores.items():
                if pairs.get(i) not in gcrl_scores:
                    if i in pairs:
                        warn_once(
                            f"WARNING: {entries[i]['label']} has no JaxGCRL runs "
                            f"for {gcrl_stat}; it is left out of the violins."
                        )
                    continue
                combined[i] = np.concatenate([values, gcrl_scores[pairs[i]]])
            if not combined:
                vprint(f"  Nothing to draw for {stat}, skipping.")
                continue
            name = f"{stat}_{group_name}" if group_name else stat
            written += draw_violins(drawing, name, ylabel, envs_drawn + gcrl_drawn, combined)

    for path in write_legend_handles(legend_handles(entries), cfg, output_dir, FONT_SIZE):
        vprint(f"  Saved: {path}")
    print(
        f"\n{config_name}: {len(entries)} configuration(s), "
        f"{len(groups)} environment group(s) and {len(gcrl_present)} JaxGCRL "
        f"environment(s) -> {output_dir}"
    )
    return written


for config_name, plot_cfg_path, config in plot_cfgs:
    vprint(f"\n{'#' * 60}\nPlot config: {config_name}\n{'#' * 60}")
    written += draw_config(config_name, config)

print(
    f"\n{len(plot_cfgs)} plot config(s), {written} figure(s) -> "
    f"{os.path.join(args.folder, args.output, 'violin_gcrl')}"
)
