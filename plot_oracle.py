"""Statistics recorded against the true visit Q-function over training, from a
pre-built results.parquet, as one grid per environment group: a row per
statistic and a column per environment.

The plot config names the configurations, the statistics and the environments
to draw, as for plot_results.py (see configs/plots/gcrl_selections/oracle.yaml),
and its `statistics` give the rows, in the order it lists them, with their
labels. Runs are selected as plot_results.py selects them: a run is drawn only
when its whole configuration is the one its entry composes to. A subplot holds
a curve per entry, the mean over seeds and its confidence band, as
plot_results.py draws them.

The statistics are the ones an actor records with results.true_visit_check
(see GoalConditioned in src/actor.py), each averaged over the goal selections
between two recorded points:

  v_err_unreach_max           the largest absolute error of the learned visit
                              value over the candidates unreachable from the
                              state of the selection
  v_err_unreach, v_err_reach  the mean absolute error over the unreachable and
                              the reachable candidates
  bound_holds_%               the fraction of the selections with an
                              unreachable candidate at which the perturbation
                              bound of the ratio score holds
  sel_unreach_bound_holds_%   the fraction of those selections that selected an
  sel_unreach_bound_fails_%   unreachable goal, where the bound holds and where
                              it fails

The files are written in <root>/<plot config>, the path of the plot config under
configs/plots without the extension: oracle_<group>.png for every environment
group, and the legend on its own as legend_horizontal.png and
legend_vertical.png.

Run with --help for the rest of the options.

Example

    python plot_oracle.py -f data_gcrl_oracle -v
"""

import argparse
import os
import sys
from ast import literal_eval

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.utils.matching import ignored_cfg_keys, load_config_group
from src.utils.misc import make_vprint
from src.utils.plot import (
    PLOT_CONFIG_DIR,
    STEPS_STAT,
    add_config_group_args,
    assign_groups,
    ensure_font,
    entry_rows,
    env_groups,
    error_shade_plot,
    hyperparameter_strings,
    load_plot_configs,
    plot_labels,
    save_and_close,
    series_of,
    set_3_ticks,
    style_entries,
    tex_kwargs,
    write_legend,
)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

FONT_SIZE = 12
SUBPLOT_W = 2.6  # inches per subplot
SUBPLOT_H = 2.0

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
    default="gcrl_selections/oracle",
    metavar="NAME",
    help=f"Plot config: which configurations and statistics to draw, and over "
         f"which environments. A name without .yaml is looked for in "
         f"{PLOT_CONFIG_DIR}, or give a path; a directory is a set of configs and "
         f"every one of them is drawn, bar those marked `# skip`.",
)
parser.add_argument(
    "-o", "--output",
    default="plots",
    help="Output root, under the data directory. Files go in <root>/<the path of "
         "the plot config under configs/plots, without .yaml>.",
)
add_config_group_args(parser)
parser.add_argument(
    "--smoothing_window",
    type=int,
    default=0,
    help="Points either side to average a curve over. 0 draws it as recorded.",
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

stats_found = {c[1] for c in df.columns if c[0] == "statistic" and c != STEPS_STAT}

ensure_font()
sns.set_context("paper")
sns.set_style("darkgrid", {"legend.frameon": True})
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.axisbelow"] = False
plt.rcParams["grid.linestyle"] = "--"


def draw_group(name, envs, stats, entries, curves, stat_labels, env_labels, stat_ylims, output_dir):
    """Draw the grid of one environment group to <name>.png in `output_dir`, a
    row per statistic of `stats` and a column per environment of `envs`, and
    return the number of figures written.

    `curves` maps (statistic, environment, entry index) to the series_of output
    of the runs the entry selects there. A subplot draws the mean over seeds and
    its confidence band of every entry it holds, and is left blank when it holds
    none. Its x axis spans the steps the statistic is recorded over, and its y
    axis takes the `ylim` the plot config gives the environment and statistic.
    Only the bottom row is labelled with steps, and only the first column with
    the statistic."""

    fig, axs = plt.subplots(
        len(stats),
        len(envs),
        figsize=(SUBPLOT_W * len(envs), SUBPLOT_H * len(stats)),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.25, wspace=0.3)
    for r, stat in enumerate(stats):
        ylabel = stat_labels.get(stat, stat)
        for c, env in enumerate(envs):
            ax = axs[r][c]
            if r == 0:
                title = env_labels.get(str(env), str(env))
                ax.set_title(title, fontsize=FONT_SIZE, **tex_kwargs(title))
            drawn = [i for i in range(len(entries)) if (stat, env, i) in curves]
            if not drawn:
                ax.set_axis_off()
                continue
            x_lo, x_hi = np.inf, 0.0
            for i in drawn:
                series, x, steps, _ = curves[(stat, env, i)]
                error_shade_plot(
                    ax,
                    series.to_numpy().T,
                    x=x,
                    smoothing_window=args.smoothing_window,
                    color=entries[i]["color"],
                    linestyle=entries[i]["linestyle"],
                    linewidth=2.0,
                )
                x_lo = min(x_lo, float(x[0]))
                x_hi = max(x_hi, steps)

            ax.tick_params(axis="x", labelsize=FONT_SIZE - 2, pad=-2)
            ax.tick_params(axis="y", labelsize=FONT_SIZE - 2, pad=1)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(3, 3))
            ax.xaxis.offsetText.set_visible(False)
            if x_hi > x_lo:
                ax.set_xlim(x_lo, x_hi)
                ax.set_xticks([x_lo, (x_lo + x_hi) / 2, x_hi])
            if r < len(stats) - 1:
                ax.tick_params(axis="x", labelbottom=False)
            ax.set_ylabel(
                ylabel if c == 0 else "",
                fontsize=FONT_SIZE,
                **tex_kwargs(ylabel),
            )
            bounds = stat_ylims.get((str(env), stat))
            if bounds:
                ax.set_ylim(*bounds)
            set_3_ticks(ax, which="y")

    return save_and_close(fig, output_dir, name, vprint)


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
    stat_labels, env_labels, stat_ylims, _score_bounds = plot_labels(cfg)
    missing = [s for s in stat_labels if s not in stats_found]
    if missing:
        print(f"WARNING: {config_name} asks for statistics not in the data: {missing}")
    stats = [s for s in stat_labels if s in stats_found]
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

    output_dir = os.path.join(args.folder, args.output, config_name)
    os.makedirs(output_dir, exist_ok=True)

    rows_cache = {}
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

        curves = {}
        for env in present:
            for i, entry in enumerate(entries):
                rows = entry_rows(df, hp_strings, rows_cache, env, i, entry, ignored_keys, matched)
                for stat in stats:
                    decoded = series_of(rows, stat)
                    if decoded is not None:
                        curves[(stat, env, i)] = decoded
        drawn_stats = [s for s in stats if any(key[0] == s for key in curves)]
        if not drawn_stats:
            vprint(f"  Nothing plottable for {group_name or '(unnamed)'}, skipping.")
            continue
        name = f"oracle_{group_name}" if group_name else "oracle"
        written += draw_group(
            name,
            present,
            drawn_stats,
            entries,
            curves,
            stat_labels,
            env_labels,
            stat_ylims,
            output_dir,
        )

    for path in write_legend(entries, cfg, output_dir, FONT_SIZE):
        vprint(f"  Saved: {path}")
    print(
        f"\n{config_name}: {len(entries)} configuration(s), "
        f"{len(groups)} environment group(s) -> {output_dir}"
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
