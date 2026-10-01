"""Curve figures from a pre-built results.gzip, laid out by a plot config.

A plot config names the configurations to draw and the environments to draw them
over (see configs/plots/example.yaml). Every key of an entry but `plot.label` is
a hyperparameter the runs must hold. An entry is a whole configuration:
its keys are composed through Hydra with everything configs/default.yaml and the
files it selects put under them, and a run is drawn only when its WHOLE
configuration is the one that comes out. Entries in one group share a hue and
differ in shade and line style.

Five flags choose the figures, each written once per (environment group,
statistic) and named after both, as <stat>_<group>_<suffix>. With none of the
flags given, all of them are written.

  --curves_mean    a subplot per environment and a curve per entry, the mean over
                   seeds and its confidence band (`curves`). --with_auc appends a
                   subplot of each entry's area under the curve, averaged over
                   the group.
  --curves_iqm     a curve per entry: the IQM over every run of the group, at
                   points along training, with its stratified-bootstrap band
                   (`curves_iqm`).
  --bars_peak      a subplot per environment and a bar per entry at the peak of
                   its curve (`peak_bar`), with --steps_to_peak a second row of
                   the steps taken to reach it; and a figure of each entry's area
                   under the curve averaged over the group (`auc`).
  --rliable_final  the rliable figures on the final value of every run.
  --rliable_auc    the rliable figures on the area under the curve of every run.

The environment and statistic labels come from the plot config, not from here.
`per_env_plot` in the plot config adds one figure per environment holding every
statistic, for --curves_mean and --bars_peak.

The rliable figures (Agarwal et al., NeurIPS 2021) reduce every run to one score
and aggregate the runs of every environment of a group with stratified-bootstrap
confidence intervals: the probability that one entry improves on another, for
every pair (`rliable_<score>_poi`); the median, IQM, mean and optimality gap
(`rliable_<score>_aggregate`); and the performance profiles
(`rliable_<score>_profile`). Every aggregate except the probability of
improvement pools scores across environments, so each environment's scores are
normalized by its `ylim` for that statistic, and a group with an environment that
has none is left out of them. The same holds for --curves_iqm.

A LaTeX table of each entry's area under the curve is written beside the figures
(table.tex), and with --rliable_final or --rliable_auc one of the IQM and the
probability of improvement on that score (rliable_final.tex, rliable_auc.tex).

--prepare_missing_runs writes a script relaunching the seeds an entry is short of,
counted against `rng_seed` in the config.

Run with --help for the rest of the options.

Example

    python plot_results.py -f data_example -p example --with_auc --curves_mean -v
"""

import argparse
import itertools
import json
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
from tqdm import tqdm

from src.utils.aggregate import (
    METRICS,
    aggregate_metrics,
    interval_estimates,
    iqm,
    performance_profile,
    probability_of_improvement,
    stack_runs,
)
from src.utils.plot import (
    PLOT_CONFIG_DIR,
    SEED,
    TIME,
    assign_groups,
    composed_mask,
    detect_precision,
    draw_bars,
    ensure_font,
    entry_config,
    env_groups,
    error_shade_plot,
    fitted_rows_cols,
    ignored_cfg_keys,
    launch_overrides,
    load_config_group,
    load_plot_configs,
    plot_labels,
    record_missing,
    rows_cols,
    save_figure,
    series_of,
    set_3_ticks,
    style_entries,
    summarize,
    tex_kwargs,
    write_missing_runs_script,
)

# A plot config names its configurations whatever reads best -- "ε-greedy (1 → 0)"
# -- and a console encoding that has no ε raises rather than prints. The figures
# carry the label either way; only the recap has to give ground.
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
    help="Data directory holding the gzip.",
)
parser.add_argument("-g", "--gzip", default="results.gzip")
parser.add_argument(
    "-p", "--plot_config",
    required=True,
    metavar="NAME",
    help=f"Plot config: which configurations to draw, and over which environments. "
         f"A name without .yaml is looked for in {PLOT_CONFIG_DIR}, or give a path; "
         f"a directory is a set of configs and every one of them is drawn.",
)
parser.add_argument(
    "-o", "--output",
    default="plots",
    help="Output root, under the data directory. Figures go in "
         "<root>/curves/<plot config>, so one data directory holds the output of "
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
    "--stats",
    nargs="+",
    default=[],
    metavar="NAME",
    help="Statistics to plot, by name and in this order. Default: every statistic "
         "in the data.",
)
parser.add_argument(
    "--smoothing_window",
    type=int,
    default=0,
    help="Points either side to average a curve over. 0 draws it as recorded.",
)
parser.add_argument(
    "--curves_mean",
    action="store_true",
    help="Write the curves: a subplot per environment, the mean over seeds and "
         "its confidence band.",
)
parser.add_argument(
    "--curves_iqm",
    action="store_true",
    help="Write the IQM curve: normalized scores pooled over the runs of every "
         "environment of a group, with a stratified-bootstrap band.",
)
parser.add_argument(
    "--bars_peak",
    action="store_true",
    help="Write the bars: each configuration's peak per environment, and its area "
         "under the curve averaged over the group.",
)
parser.add_argument(
    "--rliable_final",
    action="store_true",
    help="Write the rliable figures and tables on each run's final value: the "
         "mean of its last --rliable_last_k points.",
)
parser.add_argument(
    "--rliable_auc",
    action="store_true",
    help="Write the rliable figures and tables on each run's area under the "
         "curve.",
)
parser.add_argument(
    "--rliable_last_k",
    type=int,
    default=1,
    help="Points at the end of a curve averaged into its final value.",
)
parser.add_argument(
    "--rliable_reps",
    type=int,
    default=50000,
    help="Bootstrap replicates behind every rliable interval.",
)
parser.add_argument(
    "--iqm_points",
    type=int,
    default=20,
    help="Points along training, evenly spaced, at which --curves_iqm is "
         "computed.",
)
parser.add_argument(
    "--with_auc",
    action="store_true",
    help="Append an AUC subplot to the curves figure: one bar per configuration, "
         "its area averaged over the environments of the group.",
)
parser.add_argument(
    "--annotate_auc",
    action="store_true",
    help="Print the value above each bar of that appended AUC subplot. The "
         "standalone <stat>_auc figures are always annotated.",
)
parser.add_argument(
    "--steps_to_peak",
    action="store_true",
    help="Add a second row to the bar figures: the training steps each "
         "configuration took to reach its peak, best at the top.",
)
parser.add_argument(
    "--bar_width",
    type=float,
    default=0.85,
    help="Bar width as a fraction of the slot it sits in.",
)
parser.add_argument(
    "--auc_width_scale",
    type=float,
    default=1.0,
    help="Scales the width of the standalone <stat>_auc figures, e.g. 0.5 for a "
         "narrow one to sit beside another plot.",
)
parser.add_argument(
    "--no_legend",
    action="store_true",
    help="Leave the legend off the figures. The standalone legend is still "
         "written.",
)
parser.add_argument(
    "--prepare_missing_runs",
    action="store_true",
    help="Write a script relaunching the seeds each configuration is short of.",
)
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

# No figure flag at all writes every figure.
FIGURES = ("curves_mean", "curves_iqm", "bars_peak", "rliable_final", "rliable_auc")
if not any(getattr(args, f) for f in FIGURES):
    for f in FIGURES:
        setattr(args, f, True)


def vprint(*a):
    """Print only under --verbose."""

    if args.verbose:
        print(*a)


plot_cfgs = load_plot_configs(args.plot_config)
vprint(f"\n{len(plot_cfgs)} plot config(s): {[n for n, _, _ in plot_cfgs]}")

gzip_path = os.path.join(args.folder, args.gzip)
df = pd.read_parquet(gzip_path)
vprint(f"\nLoaded gzip from: {gzip_path}")
if all(isinstance(c, str) and c.startswith("(") for c in df.columns):
    df.columns = pd.MultiIndex.from_tuples([literal_eval(c) for c in df.columns])

# A config selects runs by hyperparameter, and `algorithm` and `environment` are
# among them -- neither is recorded, so both are matched back from their YAMLs.
# Without the YAMLs, the id the run recorded still distinguishes environments.
env_configs = load_config_group(args.environments) if args.environments else {}
algo_configs = load_config_group(args.algorithms) if args.algorithms else {}

# The answers `assign_groups` has already given, by the exemptions that produced
# them: configs declaring the same ones -- usually every one of them -- get the
# same answer, and the frame is compared against every YAML to arrive at it.
_assigned: dict = {}


stats_found = sorted({c[1] for c in df.columns if c[0] == "statistic"})

ensure_font()
sns.set_context("paper")
sns.set_style("darkgrid", {"legend.frameon": True})
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.axisbelow"] = False
plt.rcParams["grid.linestyle"] = "--"

# Accumulated over every config drawn: the relaunch script is one script for the
# whole run, whatever it took to find out what is missing.
missing_runs = {}
written = 0


def rows_of(drawing, env, index, entry):
    """The runs of one environment that an entry selects: the ones whose whole
    recorded configuration is the one the entry composes to there.

    The comparison covers every key, the ones the entry leaves out included: a
    run launched with another value for one of those is a different
    configuration, and belongs to whichever entry composes to it.

    Kept in `drawing.rows`: the same (environment, entry) is asked for once for
    the timings, once per statistic and once for the missing runs, and each
    answer costs a comparison against every hyperparameter column of the frame."""

    key = (env, index)
    if key not in drawing.rows:
        flat = entry_config(env, entry["selector"], entry["label"])
        drawing.rows[key] = (
            df.iloc[0:0] if flat is None
            else df[
                composed_mask(df, flat, ignore=drawing.ignored_keys)
                & drawing.matched
            ]
        )
    return drawing.rows[key]


def bar_values(decoded):
    """What the bars of one configuration are: its peak, the steps it took to
    reach that peak, and the area under its curve.

    Each is taken per seed and then summarized, so the interval says how much the
    seeds disagreed -- the same thing the band around the curve says. The area is
    the mean over the curve, i.e. divided by the span it covers, so it is in the
    units of the y axis and configurations trained for different lengths stay
    comparable."""

    series, stepsize, _steps, _seeds = decoded
    peaks, steps, areas = [], [], []
    for column in series.columns:
        run = series[column].to_numpy(dtype=float)
        finite = run[np.isfinite(run)]
        if not finite.size:
            continue
        peaks.append(float(finite.max()))
        # nanargmax raises on an all-NaN run, which the guard above has already
        # dropped; -inf stands in for the NaNs so they can never be the peak.
        steps.append(float(np.argmax(np.nan_to_num(run, nan=-np.inf))) * stepsize)
        areas.append(float(finite.mean()))
    return {
        "peak": summarize(peaks),
        "steps": summarize(steps),
        "auc": summarize(areas),
    }


def legend_on(drawing, fig, handles, offset):
    """Name the configurations drawn, once for the whole figure and below it.
    `--no_legend` leaves it off; the standalone legend figure is written either
    way, for a figure that has to carry one elsewhere."""

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


def pooled(summaries, envs, stat, key, index):
    """One configuration's `key` averaged over the environments of a group, as
    (mean, interval).

    The two are averaged separately: the interval is what the seeds of an
    environment disagreed by, averaged over environments, and not the spread
    between environments -- those differ in difficulty, not in what the
    configuration did."""

    values = [
        summaries[(env, stat, index)][key]
        for env in envs
        if (env, stat, index) in summaries
        and summaries[(env, stat, index)][key] is not None
    ]
    values = [v for v in values if np.isfinite(v[0])]
    if not values:
        return None
    return (
        float(np.mean([m for m, _ in values])),
        float(np.mean([e for _, e in values])),
    )


@dataclass
class Drawing:
    """Everything one plot config decides, for the functions that draw it.

    Passed rather than left on the module: a `-p` naming a directory draws
    several configs in turn, and two of them name different runs and compare
    different things, so nothing one of them settled may still be standing when
    the next is drawn.

    `summaries`, `rliable` and `time_table` are filled as the figures are drawn
    and read back by the tables and the recap; `drawn_envs` is which environments each
    group actually had runs for; `rows` is the cache rows_of keeps."""

    name: str
    cfg: dict
    entries: list
    groups: dict
    stats: list
    stat_labels: dict
    env_labels: dict
    stat_ylims: dict
    # The pair `assign_groups` returned, kept together: a mask built under one
    # set of exemptions says nothing about a comparison made under another.
    # `matched` here is the runs placed in an environment, the algorithm label
    # having no say in it.
    ignored_keys: frozenset
    matched: pd.Series
    env_stem: pd.Series
    expected_seeds: object
    output_dir: str
    summaries: dict = field(default_factory=dict)
    # (score kind, group, statistic) -> the rliable estimates the tables print.
    rliable: dict = field(default_factory=dict)
    time_table: dict = field(default_factory=dict)
    drawn_envs: dict = field(default_factory=dict)
    rows: dict = field(default_factory=dict)


RLIABLE_SCORES = {"final": "Final", "auc": "AUC"}


def run_scores(decoded, kind):
    """One score per seed of a configuration: the mean of the last
    --rliable_last_k finite points of its curve (`final`), or the mean of the
    whole curve (`auc`, the area bar_values reports)."""

    series = decoded[0]
    scores = []
    for column in series.columns:
        run = series[column].to_numpy(dtype=float)
        finite = run[np.isfinite(run)]
        if not finite.size:
            continue
        kept = finite if kind == "auc" else finite[-args.rliable_last_k:]
        scores.append(float(kept.mean()))
    return scores


def norm_bounds(drawing, envs, stat):
    """The `ylim` of each environment for `stat`, as (low, high) arrays, or None
    when an environment has none or an empty one."""

    bounds = [drawing.stat_ylims.get((str(env), stat)) for env in envs]
    if any(b is None or b[1] == b[0] for b in bounds):
        return None
    return (
        np.array([b[0] for b in bounds], dtype=float),
        np.array([b[1] for b in bounds], dtype=float),
    )


def complete_entries(drawing, present, stat, decoded_of, drawn):
    """The entries holding runs in every environment of the group. An aggregate
    over environments compares configurations on the same ones, so an entry
    missing any of them is left out."""

    kept = []
    for i in drawn:
        if all((env, i) in decoded_of for env in present):
            kept.append(i)
        else:
            print(
                f"WARNING: {drawing.entries[i]['label']} has no runs for {stat} "
                f"in some environment of the group; it is left out of the rliable "
                f"estimates."
            )
    return kept


def draw_iqm_curve(drawing, name, present, stat, decoded_of, drawn):
    """The IQM of the normalized scores of every run of the group at
    --iqm_points points along training, with its stratified-bootstrap band.
    Returns the number of figures written.

    A point is placed by the fraction of training it sits at, so environments
    recording a different number of points are read at the same fraction. The x
    axis is in steps when every run trained for the same number of them, and in
    the fraction of training otherwise. A seed with no value at one of the points
    is dropped."""

    bounds = norm_bounds(drawing, present, stat)
    if bounds is None:
        vprint(f"  No ylim for {stat} in every environment, no IQM curve.")
        return 0
    low, high = bounds
    fractions = np.linspace(0.0, 1.0, args.iqm_points)

    data, steps = {}, set()
    for i in complete_entries(drawing, present, stat, decoded_of, drawn):
        per_env = []
        for env in present:
            series, _stepsize, env_steps, _seeds = decoded_of[(env, i)]
            values = series.to_numpy(dtype=float).T
            index = np.round(fractions * (values.shape[1] - 1)).astype(int)
            picked = values[:, index]
            per_env.append(picked[np.all(np.isfinite(picked), axis=1)])
            steps.add(env_steps)
        if any(len(p) == 0 for p in per_env):
            continue
        # (points, runs, environments): the metrics reduce the last two axes.
        stacked = stack_runs(per_env).transpose(1, 0, 2)
        data[i] = (stacked - low) / (high - low)
    if not data:
        return 0

    estimates = {
        i: interval_estimates(iqm, s, reps=args.rliable_reps)
        for i, s in tqdm(
            data.items(),
            desc=f"{name} IQM curve",
            unit="estimate",
            leave=False,
        )
    }

    x = fractions * steps.pop() if len(steps) == 1 else fractions
    ylabel = drawing.stat_labels.get(stat, stat)
    fig, axs = plt.subplots(
        1, 1,
        figsize=(SUBPLOT_W * 1.5, SUBPLOT_H * 1.5),
        squeeze=False,
    )
    ax = axs[0][0]
    for i, (point, lo, hi) in estimates.items():
        entry = drawing.entries[i]
        ax.plot(
            x,
            point,
            color=entry["color"],
            linestyle=entry["linestyle"],
            linewidth=2.0,
        )
        ax.fill_between(
            x,
            lo,
            hi,
            alpha=0.2,
            linewidth=0.0,
            color=entry["color"],
        )
    ax.tick_params(axis="x", labelsize=FONT_SIZE - 2, pad=-2)
    ax.tick_params(axis="y", labelsize=FONT_SIZE - 2, pad=1)
    if x[-1] > 1:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(3, 3))
        ax.xaxis.offsetText.set_visible(False)
    else:
        ax.set_xlabel("Fraction Of Training", fontsize=FONT_SIZE)
    ax.set_xlim(0, x[-1])
    ax.set_xticks([0, x[-1] / 2, x[-1]])
    label = f"IQM Normalized {ylabel}"
    ax.set_ylabel(label, fontsize=FONT_SIZE, **tex_kwargs(label))
    set_3_ticks(ax, which="y")
    legend_on(
        drawing,
        fig,
        [
            plt.Line2D(
                [], [],
                color=drawing.entries[i]["color"],
                linestyle=drawing.entries[i]["linestyle"],
                linewidth=2.0,
                label=drawing.entries[i]["label"],
            )
            for i in data
        ],
        -0.15,
    )
    vprint(f"  Saved: {save_figure(fig, drawing.output_dir, f'{name}_curves_iqm')}")
    plt.close(fig)
    return 1


def draw_rliable(drawing, name, group_name, present, stat, decoded_of, drawn, kind):
    """The rliable figures of one (group, statistic) on one score per run, `kind`
    being `final` or `auc` (see run_scores). Returns the number of figures
    written.

    The probability of improvement is drawn for every pair of configurations,
    and needs no normalization: it compares runs within an environment and
    averages over environments. The aggregate metrics and the performance
    profile pool runs across environments, so they are drawn only when every
    environment has a `ylim` for `stat`, which the scores are normalized by."""

    scores = {}
    for i in complete_entries(drawing, present, stat, decoded_of, drawn):
        per_env = [run_scores(decoded_of[(env, i)], kind) for env in present]
        if all(per_env):
            scores[i] = stack_runs(per_env)
    if not scores:
        return 0

    ylabel = drawing.stat_labels.get(stat, stat)
    score_label = RLIABLE_SCORES[kind]
    stem = f"{name}_rliable_{kind}"
    record = drawing.rliable.setdefault((kind, group_name, stat), {})
    written = 0

    # --- Probability of improvement ------------------------------------------
    pairs = list(itertools.combinations(scores, 2))
    if pairs:
        record["poi"] = {
            (a, b): tuple(
                float(v) for v in interval_estimates(
                    probability_of_improvement,
                    scores[a],
                    scores[b],
                    reps=args.rliable_reps,
                )
            )
            for a, b in tqdm(
                pairs,
                desc=f"{stem} probability of improvement",
                unit="estimate",
                leave=False,
            )
        }
        fig, axs = plt.subplots(
            1, 1,
            figsize=(SUBPLOT_W * 1.5, max(0.4 * len(pairs) + 0.6, SUBPLOT_H)),
            squeeze=False,
        )
        ax = axs[0][0]
        for row, (a, b) in enumerate(pairs):
            p, lo, hi = record["poi"][(a, b)]
            entry = drawing.entries[a]
            ax.barh(
                row,
                hi - lo,
                left=lo,
                height=0.6,
                color=entry["color"],
                hatch=entry["hatch"],
                edgecolor="black",
                linewidth=0.6,
            )
            ax.vlines(p, row - 0.3, row + 0.3, color="black", linewidth=1.5)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(
            [
                f"{drawing.entries[a]['label']} vs. {drawing.entries[b]['label']}"
                for a, b in pairs
            ]
        )
        for text in ax.get_yticklabels():
            text.set(**tex_kwargs(text.get_text()))
        ax.invert_yaxis()
        ax.tick_params(axis="both", labelsize=FONT_SIZE - 2, pad=1)
        xlabel = f"P(X > Y), {score_label} {ylabel}"
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE, **tex_kwargs(xlabel))
        vprint(f"  Saved: {save_figure(fig, drawing.output_dir, f'{stem}_poi')}")
        plt.close(fig)
        written += 1

    # --- Aggregate metrics and performance profile -----------------------------
    bounds = norm_bounds(drawing, present, stat)
    if bounds is None:
        vprint(
            f"  No ylim for {stat} in every environment, no rliable aggregates "
            f"or profile."
        )
        return written
    low, high = bounds
    normalized = {i: (s - low) / (high - low) for i, s in scores.items()}
    order = list(normalized)

    record["aggregate"] = {}
    for i in tqdm(order, desc=f"{stem} aggregates", unit="estimate", leave=False):
        point, lo, hi = interval_estimates(
            aggregate_metrics,
            normalized[i],
            reps=args.rliable_reps,
        )
        record["aggregate"][i] = (point, np.stack([lo, hi]))

    fig, axs = plt.subplots(
        1, len(METRICS),
        figsize=(
            SUBPLOT_W * len(METRICS),
            max(0.4 * len(order) + 0.6, SUBPLOT_H),
        ),
        squeeze=False,
    )
    fig.subplots_adjust(wspace=0.15)
    for k, metric in enumerate(METRICS):
        ax = axs[0][k]
        for row, i in enumerate(order):
            entry = drawing.entries[i]
            values, bands = record["aggregate"][i]
            ax.barh(
                row,
                bands[1, k] - bands[0, k],
                left=bands[0, k],
                height=0.6,
                color=entry["color"],
                hatch=entry["hatch"],
                edgecolor="black",
                linewidth=0.6,
            )
            ax.vlines(values[k], row - 0.3, row + 0.3, color="black", linewidth=1.5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(
            [drawing.entries[i]["label"] for i in order] if k == 0 else []
        )
        for text in ax.get_yticklabels():
            text.set(**tex_kwargs(text.get_text()))
        ax.invert_yaxis()
        ax.set_title(metric, fontsize=FONT_SIZE)
        ax.tick_params(axis="both", labelsize=FONT_SIZE - 2, pad=1)
        set_3_ticks(ax, which="x")
    xlabel = f"Normalized {score_label} {ylabel}"
    fig.supxlabel(xlabel, fontsize=FONT_SIZE, y=-0.08, **tex_kwargs(xlabel))
    vprint(f"  Saved: {save_figure(fig, drawing.output_dir, f'{stem}_aggregate')}")
    plt.close(fig)
    written += 1

    every = np.concatenate([s.ravel() for s in normalized.values()])
    taus = np.linspace(np.nanmin(every), np.nanmax(every), 101)
    profiles = {
        i: interval_estimates(
            lambda s: performance_profile(s, taus),
            normalized[i],
            reps=args.rliable_reps,
        )
        for i in tqdm(order, desc=f"{stem} profiles", unit="estimate", leave=False)
    }
    fig, axs = plt.subplots(
        1, 1,
        figsize=(SUBPLOT_W * 1.5, SUBPLOT_H * 1.5),
        squeeze=False,
    )
    ax = axs[0][0]
    for i in order:
        entry = drawing.entries[i]
        profile, lo, hi = profiles[i]
        ax.plot(
            taus,
            profile,
            color=entry["color"],
            linestyle=entry["linestyle"],
            linewidth=2.0,
        )
        ax.fill_between(
            taus,
            lo,
            hi,
            alpha=0.2,
            linewidth=0.0,
            color=entry["color"],
        )
    ax.tick_params(axis="both", labelsize=FONT_SIZE - 2, pad=1)
    xlabel = f"Normalized {score_label} {ylabel} (τ)"
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE, **tex_kwargs(xlabel))
    ax.set_ylabel("Fraction Of Runs > τ", fontsize=FONT_SIZE)
    ax.set_ylim(0, 1)
    set_3_ticks(ax, which="both")
    legend_on(
        drawing,
        fig,
        [
            plt.Line2D(
                [], [],
                color=drawing.entries[i]["color"],
                linestyle=drawing.entries[i]["linestyle"],
                linewidth=2.0,
                label=drawing.entries[i]["label"],
            )
            for i in order
        ],
        -0.2,
    )
    vprint(f"  Saved: {save_figure(fig, drawing.output_dir, f'{stem}_profile')}")
    plt.close(fig)
    written += 1
    return written


def draw_config(config_name, cfg):
    """Draw one plot config's figures, into its own directory, and return how
    many it wrote.

    A `-p` naming a directory is a set of configs that belong together -- one
    sweep's -- and every one of them is drawn here in turn, so the run that
    reads the gzip once produces all of them."""

    entries = style_entries(cfg)
    if not entries:
        print(f"WARNING: {config_name} declares no configurations to plot, skipping.")
        return 0
    groups = env_groups(cfg)
    if not groups:
        print(f"WARNING: {config_name} declares no environments to plot, skipping.")
        return 0
    stat_labels, env_labels, stat_ylims = plot_labels(cfg)

    # The config's `statistics` is both the list to plot and their labels, in the
    # order it writes them; --stats narrows that further. Without either,
    # everything the data holds -- which is a lot of it that is not a curve.
    wanted = list(args.stats) or list(stat_labels) or stats_found
    missing = [s for s in wanted if s not in stats_found]
    if missing:
        print(f"WARNING: {config_name} asks for statistics not in the data: {missing}")
    stats = [s for s in wanted if s in stats_found]
    if not stats:
        print(f"WARNING: {config_name} has no statistic to plot, skipping.")
        return 0
    vprint(f"Statistics: {stats}")

    # This config's own exemptions, not every config's: exempting a key changes
    # which YAML a run matches, and one config's `ignored_cfg_keys` says nothing
    # about the runs another one draws.
    ignored_keys = ignored_cfg_keys(cfg)
    if ignored_keys:
        vprint(f"Keys exempt from config matching: {sorted(ignored_keys)}")
    # The environment label alone decides which runs are drawable; the algorithm
    # label is assigned for the figures to read. An entry here is a whole composed
    # configuration, so the comparison `rows_of` makes already holds every key an
    # algorithm YAML declares, at the value the configuration composes to, and
    # holds the interpolated ones the YAML comparison drops. A YAML declaring a
    # default a sweep overrode agrees with the runs it launched on every key but
    # that one, which leaves them "unknown" there.
    _env, _algo, env_stem, matched = assign_groups(
        df,
        env_configs,
        algo_configs,
        ignored_keys,
        algo_decides_usable=False,
        cache=_assigned,
    )

    # <data dir>/<root>/curves/<config>, where the config names itself by where it
    # sits under configs/plots -- so a directory of configs becomes a directory of
    # figures, rather than every `default.yaml` writing over the same place.
    output_dir = os.path.join(args.folder, args.output, "curves", config_name)
    os.makedirs(output_dir, exist_ok=True)

    drawing = Drawing(
        name=config_name,
        cfg=cfg,
        entries=entries,
        groups=groups,
        stats=stats,
        stat_labels=stat_labels,
        env_labels=env_labels,
        stat_ylims=stat_ylims,
        ignored_keys=ignored_keys,
        matched=matched,
        env_stem=env_stem,
        expected_seeds=cfg.get("rng_seed"),
        output_dir=output_dir,
    )
    written = 0

    for group_name, envs in drawing.groups.items():
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

        drawing.drawn_envs.setdefault(group_name, present)
        # (environment, statistic, entry index) -> decoded curve, kept across the
        # statistic loop because the per-environment figures cut the other way:
        # one figure per environment holding every statistic.
        curves_of = {}

        # The average wall-clock of a run, for the recap at the end. `time` is one
        # number per run rather than a curve, so it is read off the column.
        if TIME in df.columns:
            for env in present:
                for i, entry in enumerate(drawing.entries):
                    raw = rows_of(drawing, env, i, entry)[TIME].dropna()
                    if raw.empty:
                        continue
                    try:
                        seconds = [float(json.loads(v)) for v in raw]
                    except (TypeError, ValueError):
                        continue
                    drawing.time_table[(env, i)] = float(np.nanmean(seconds)) / 60

        for stat in drawing.stats:
            ylabel = drawing.stat_labels.get(stat, stat)

            # Decoded once and drawn twice: the curves figure and the bars figure
            # ask the same runs two questions.
            decoded_of = {}
            for env in present:
                for i, entry in enumerate(drawing.entries):
                    decoded = series_of(rows_of(drawing, env, i, entry), stat)
                    if decoded is None:
                        continue
                    decoded_of[(env, i)] = decoded
                    curves_of[(env, stat, i)] = decoded
                    drawing.summaries[(env, stat, i)] = bar_values(decoded)
            drawn = sorted({i for _, i in decoded_of})
            if not drawn:
                vprint(f"  Nothing plottable for {stat}, skipping.")
                continue
            shown = [drawing.entries[i] for i in drawn]

            name = f"{stat}_{group_name}" if group_name else stat

            # --- The curves, and the areas under them ----------------------
            if args.curves_mean:
                # The AUC bars take the cell after the last environment, so the
                # arrangement is asked for one more subplot than there are.
                n_cells = len(present) + (1 if args.with_auc else 0)
                n_rows, n_cols = fitted_rows_cols(
                    drawing.name,
                    "plots_rows_cols",
                    n_cells,
                    drawing.cfg.get("plots_rows_cols"),
                )
                fig, axs = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(SUBPLOT_W * n_cols, SUBPLOT_H * n_rows),
                    squeeze=False,
                )
                fig.subplots_adjust(wspace=0.3)

                for k, env in enumerate(present):
                    ax = axs[k // n_cols][k % n_cols]
                    title = drawing.env_labels.get(str(env), str(env))
                    ax.set_title(title, fontsize=FONT_SIZE, **tex_kwargs(title))

                    x_hi = 0.0
                    for i, entry in enumerate(drawing.entries):
                        if (env, i) not in decoded_of:
                            continue
                        series, stepsize, steps, _ = decoded_of[(env, i)]
                        error_shade_plot(
                            ax,
                            series.to_numpy().T,
                            stepsize=stepsize,
                            smoothing_window=args.smoothing_window,
                            color=entry["color"],
                            linestyle=entry["linestyle"],
                            linewidth=2.0,
                        )
                        x_hi = max(x_hi, steps)

                    ax.tick_params(axis="x", labelsize=FONT_SIZE - 2, pad=-2)
                    ax.tick_params(axis="y", labelsize=FONT_SIZE - 2, pad=1)
                    ax.ticklabel_format(style="sci", axis="x", scilimits=(3, 3))
                    ax.xaxis.offsetText.set_visible(False)
                    ax.set_ylabel(
                        ylabel if k % n_cols == 0 else "",
                        fontsize=FONT_SIZE,
                        **tex_kwargs(ylabel),
                    )
                    if x_hi > 0:
                        ax.set_xlim(0, x_hi)
                        ax.set_xticks([0, x_hi / 2, x_hi])
                    bounds = drawing.stat_ylims.get((str(env), stat))
                    if bounds:
                        ax.set_ylim(*bounds)
                    set_3_ticks(ax, which="y")

                if args.with_auc:
                    k = len(present)
                    ax = axs[k // n_cols][k % n_cols]
                    ax.set_title("Avg. Area-Under-Curve", fontsize=FONT_SIZE)
                    items = [
                        (
                            drawing.entries[i],
                            pooled(drawing.summaries, present, stat, "auc", i),
                        )
                        for i in drawn
                    ]
                    draw_bars(
                        ax,
                        items,
                        ylabel="",
                        bar_width=args.bar_width,
                        font_size=FONT_SIZE,
                        annotate=args.annotate_auc,
                    )
                    set_3_ticks(ax, which="y")

                # The cells of the arrangement that nothing landed in.
                for k in range(n_cells, n_rows * n_cols):
                    axs[k // n_cols][k % n_cols].axis("off")

                legend_on(
                    drawing,
                    fig,
                    [
                        plt.Line2D(
                            [], [],
                            color=e["color"],
                            linestyle=e["linestyle"],
                            linewidth=2.0,
                            label=e["label"],
                        )
                        for e in shown
                    ],
                    -0.15,
                )
                vprint(
                    f"  Saved: "
                    f"{save_figure(fig, drawing.output_dir, f'{name}_curves')}"
                )
                plt.close(fig)
                written += 1

            # --- The peak of each curve, and what it cost to get there ------
            if args.bars_peak:
                n_rows, n_cols = fitted_rows_cols(
                    drawing.name,
                    "plots_rows_cols",
                    len(present),
                    drawing.cfg.get("plots_rows_cols"),
                )
                stack = 2 if args.steps_to_peak else 1
                fig, axs = plt.subplots(
                    n_rows * stack,
                    n_cols,
                    figsize=(SUBPLOT_W * n_cols, SUBPLOT_H * n_rows * stack),
                    squeeze=False,
                )
                fig.subplots_adjust(wspace=0.3)

                for k, env in enumerate(present):
                    # With --steps_to_peak a subplot is a pair, the steps directly
                    # under the peak they belong to.
                    row, col = (k // n_cols) * stack, k % n_cols
                    ax = axs[row][col]
                    title = drawing.env_labels.get(str(env), str(env))
                    ax.set_title(title, fontsize=FONT_SIZE, **tex_kwargs(title))
                    draw_bars(
                        ax,
                        [
                            (
                                drawing.entries[i],
                                drawing.summaries.get((env, stat, i), {}).get("peak"),
                            )
                            for i in drawn
                        ],
                        ylabel=ylabel if col == 0 else "",
                        bar_width=args.bar_width,
                        font_size=FONT_SIZE,
                    )
                    set_3_ticks(ax, which="y")
                    if args.steps_to_peak:
                        ax2 = axs[row + 1][col]
                        draw_bars(
                            ax2,
                            [
                                (
                                    drawing.entries[i],
                                    drawing.summaries
                                    .get((env, stat, i), {}).get("steps"),
                                )
                                for i in drawn
                            ],
                            ylabel="Steps To Peak" if col == 0 else "",
                            bar_width=args.bar_width,
                            font_size=FONT_SIZE,
                            # The axis counts downward, so a bar reaching further
                            # down is the one that took longer. Bar length is
                            # still the step count: the shorter bar is the faster
                            # configuration, not the taller one.
                            invert_y=True,
                        )
                        set_3_ticks(ax2, which="y")

                for k in range(len(present), n_rows * n_cols):
                    row, col = (k // n_cols) * stack, k % n_cols
                    for r in range(stack):
                        axs[row + r][col].axis("off")

                legend_on(
                    drawing,
                    fig,
                    [
                        mpatches.Patch(
                            facecolor=e["color"],
                            hatch=e["hatch"],
                            edgecolor="black",
                            linewidth=0.6,
                            label=e["label"],
                        )
                        for e in shown
                    ],
                    -0.1,
                )
                vprint(
                    f"  Saved: "
                    f"{save_figure(fig, drawing.output_dir, f'{name}_peak_bar')}"
                )
                plt.close(fig)
                written += 1

            # --- One bar per configuration, over the whole group ------------
            # A bar figure, so it is written with the other bars.
            items = [
                (
                    drawing.entries[i],
                    pooled(drawing.summaries, present, stat, "auc", i),
                )
                for i in drawn
            ]
            if args.bars_peak and any(v is not None for _, v in items):
                fig, axs = plt.subplots(
                    1, 1,
                    figsize=(max(len(items) * 0.9, 3.0) * args.auc_width_scale, 1.8),
                    squeeze=False,
                )
                draw_bars(
                    axs[0][0],
                    items,
                    ylabel="",
                    bar_width=args.bar_width,
                    font_size=FONT_SIZE,
                    annotate=True,
                )
                set_3_ticks(axs[0][0], which="y")
                legend_on(
                    drawing,
                    fig,
                    [
                        mpatches.Patch(
                            facecolor=e["color"],
                            hatch=e["hatch"],
                            edgecolor="black",
                            linewidth=0.6,
                            label=e["label"],
                        )
                        for e, v in items if v is not None
                    ],
                    -0.25,
                )
                vprint(
                    f"  Saved: "
                    f"{save_figure(fig, drawing.output_dir, f'{name}_auc')}"
                )
                plt.close(fig)
                written += 1

            # --- Aggregates over the runs of every environment (rliable) ----
            if args.curves_iqm:
                written += draw_iqm_curve(
                    drawing, name, present, stat, decoded_of, drawn
                )
            for kind in ("final", "auc"):
                if getattr(args, f"rliable_{kind}"):
                    written += draw_rliable(
                        drawing, name, group_name, present, stat, decoded_of,
                        drawn, kind,
                    )

        # --- One figure per environment, holding every statistic -----------
        # The figures above cut the data one statistic at a time, across the
        # environments; this cuts it the other way, and is opt-in since it draws
        # the same data again. `per_env_plot:
        # {show: True, rows_cols: [rows, cols]}` in the plot config turns it on.
        per_env = drawing.cfg.get("per_env_plot") or {}
        if per_env.get("show", False):
            for env in present:
                env_stats = [
                    s for s in drawing.stats
                    if any(
                        (env, s, i) in curves_of
                        for i in range(len(drawing.entries))
                    )
                ]
                if not env_stats:
                    continue
                drawn = sorted({i for e, s, i in curves_of if e == env})
                shown = [drawing.entries[i] for i in drawn]
                n_rows, n_cols = fitted_rows_cols(
                    drawing.name,
                    "per_env_plot.rows_cols",
                    len(env_stats),
                    per_env.get("rows_cols"),
                )
                env_name = str(env).replace("/", "_")

                if args.curves_mean:
                    fig, axs = plt.subplots(
                        n_rows, n_cols,
                        figsize=(SUBPLOT_W * n_cols, SUBPLOT_H * n_rows),
                        squeeze=False,
                    )
                    fig.subplots_adjust(hspace=0.5, wspace=0.3)
                    for k, stat in enumerate(env_stats):
                        ax = axs[k // n_cols][k % n_cols]
                        title = drawing.stat_labels.get(stat, stat)
                        ax.set_title(title, fontsize=FONT_SIZE, **tex_kwargs(title))

                        x_hi = 0.0
                        for i, entry in enumerate(drawing.entries):
                            if (env, stat, i) not in curves_of:
                                continue
                            series, stepsize, steps, _ = curves_of[(env, stat, i)]
                            error_shade_plot(
                                ax,
                                series.to_numpy().T,
                                stepsize=stepsize,
                                smoothing_window=args.smoothing_window,
                                color=entry["color"],
                                linestyle=entry["linestyle"],
                                linewidth=2.0,
                            )
                            x_hi = max(x_hi, steps)

                        ax.tick_params(axis="x", labelsize=FONT_SIZE - 2, pad=-2)
                        ax.tick_params(axis="y", labelsize=FONT_SIZE - 2, pad=1)
                        ax.ticklabel_format(style="sci", axis="x", scilimits=(3, 3))
                        ax.xaxis.offsetText.set_visible(False)
                        if x_hi > 0:
                            ax.set_xlim(0, x_hi)
                            ax.set_xticks([0, x_hi / 2, x_hi])
                        bounds = drawing.stat_ylims.get((str(env), stat))
                        if bounds:
                            ax.set_ylim(*bounds)
                        set_3_ticks(ax, which="y")

                    for k in range(len(env_stats), n_rows * n_cols):
                        axs[k // n_cols][k % n_cols].axis("off")

                    legend_on(
                        drawing,
                        fig,
                        [
                            plt.Line2D(
                                [], [],
                                color=e["color"],
                                linestyle=e["linestyle"],
                                linewidth=2.0,
                                label=e["label"],
                            )
                            for e in shown
                        ],
                        -0.2,
                    )
                    vprint(
                        f"  Saved: "
                        f"{save_figure(fig, drawing.output_dir, f'{env_name}_curves')}"
                    )
                    plt.close(fig)
                    written += 1

                if args.bars_peak:
                    stack = 2 if args.steps_to_peak else 1
                    fig, axs = plt.subplots(
                        n_rows * stack, n_cols,
                        figsize=(SUBPLOT_W * n_cols, SUBPLOT_H * n_rows * stack),
                        squeeze=False,
                    )
                    fig.subplots_adjust(hspace=0.5, wspace=0.3)
                    used = set()
                    for k, stat in enumerate(env_stats):
                        row, col = (k // n_cols) * stack, k % n_cols
                        used.update((row + r, col) for r in range(stack))
                        ax = axs[row][col]
                        title = drawing.stat_labels.get(stat, stat)
                        ax.set_title(title, fontsize=FONT_SIZE, **tex_kwargs(title))
                        draw_bars(
                            ax,
                            [
                                (
                                    drawing.entries[i],
                                    drawing.summaries
                                    .get((env, stat, i), {}).get("peak"),
                                )
                                for i in drawn
                            ],
                            ylabel="",
                            bar_width=args.bar_width,
                            font_size=FONT_SIZE,
                        )
                        set_3_ticks(ax, which="y")
                        if args.steps_to_peak:
                            ax2 = axs[row + 1][col]
                            draw_bars(
                                ax2,
                                [
                                    (
                                        drawing.entries[i],
                                        drawing.summaries
                                        .get((env, stat, i), {}).get("steps"),
                                    )
                                    for i in drawn
                                ],
                                ylabel="Steps To Peak" if col == 0 else "",
                                bar_width=args.bar_width,
                                font_size=FONT_SIZE,
                                invert_y=True,
                            )
                            set_3_ticks(ax2, which="y")

                    for row in range(n_rows * stack):
                        for col in range(n_cols):
                            if (row, col) not in used:
                                axs[row][col].axis("off")

                    legend_on(
                        drawing,
                        fig,
                        [
                            mpatches.Patch(
                                facecolor=e["color"],
                                hatch=e["hatch"],
                                edgecolor="black",
                                linewidth=0.6,
                                label=e["label"],
                            )
                            for e in shown
                        ],
                        -0.1,
                    )
                    vprint(
                        f"  Saved: "
                        f"{save_figure(fig, drawing.output_dir, f'{env_name}_peak_bar')}"
                    )
                    plt.close(fig)
                    written += 1

    # --- The seeds each configuration is short of ------------------------------
    # Over every environment the config declares rather than the ones that had
    # runs, and outside the group loop for the same reason: an environment with no
    # runs at all is the one short of the most seeds, and it never reaches
    # `present` -- a group with none skips the loop body altogether.
    if args.prepare_missing_runs and drawing.expected_seeds is not None:
        declared = dict.fromkeys(
            env for envs in drawing.groups.values() for env in envs
        )
        for env in declared:
            for i, entry in enumerate(drawing.entries):
                # An entry naming another environment composes to nothing here,
                # and the seeds it is short of belong to the environment it names.
                # Recorded against this one, its own environment would still win
                # the override that keys the relaunch (see launch_overrides), so
                # every seed would be filed as missing under a configuration that
                # may well be complete.
                if entry_config(env, entry["selector"], entry["label"]) is None:
                    continue
                rows = rows_of(drawing, env, i, entry)
                seeds = sorted(int(s) for s in rows[SEED].dropna().unique()) \
                    if SEED in rows.columns else []
                # The entry's own keys are the overrides to relaunch it with:
                # they are what made it this configuration and not another.
                record_missing(
                    missing_runs,
                    launch_overrides({"environment": env}, dict(entry["selector"])),
                    seeds,
                    drawing.expected_seeds,
                )

    write_table(drawing)
    write_rliable_tables(drawing)
    recap(drawing)
    write_legend(drawing)
    print(
        f"\n{drawing.name}: {len(drawing.entries)} configuration(s), "
        f"{len(drawing.groups)} environment group(s) -> {drawing.output_dir}"
    )
    return written


def tex_escape(text):
    """A display label as LaTeX: an underscore in a name is a subscript."""

    return str(text).replace("_", r"\_")


def write_table(drawing):
    """The areas under the curves of one config as a LaTeX table, beside its
    figures."""

    summaries = drawing.summaries
    table_stats = [s for s in drawing.stats if any(k[1] == s for k in summaries)]
    table_entries = [
        i for i in range(len(drawing.entries)) if any(k[2] == i for k in summaries)
    ]
    table_groups = [
        (
            g,
            [
                e for e in envs
                if any(
                    (e, s, i) in summaries
                    for s in table_stats for i in table_entries
                )
            ],
        )
        for g, envs in drawing.drawn_envs.items()
    ]
    table_groups = [(g, envs) for g, envs in table_groups if envs]
    if not (table_stats and table_entries and table_groups):
        return

    n_entries = len(table_entries)
    rows = [
        r"\begin{tabular}{l" + "c" * (n_entries * len(table_stats)) + "}",
        r"\toprule",
    ]

    # One header row spanning each statistic, one naming the configurations under
    # it, and the configurations repeat: a row is an environment throughout.
    header, rules, first = [""], [], 2
    for stat in table_stats:
        header.append(
            r"\multicolumn{" + str(n_entries) + r"}{c}{\textbf{"
            + tex_escape(drawing.stat_labels.get(stat, stat)) + "}}"
        )
        rules.append(
            r"\cmidrule(lr){" + str(first) + "-"
            + str(first + n_entries - 1) + "}"
        )
        first += n_entries
    rows.append(" & ".join(header) + r" \\")
    rows.append(" ".join(rules))
    entry_labels = [
        tex_escape(drawing.entries[i]["label"])
        for _ in table_stats for i in table_entries
    ]
    rows.append(" & ".join([""] + entry_labels) + r" \\")
    rows.append(r"\midrule")

    for g, (group_name, group_envs) in enumerate(table_groups):
        if g > 0:
            rows.append(r"\midrule")
        for env in group_envs:
            cells = [tex_escape(drawing.env_labels.get(str(env), str(env)))]
            for stat in table_stats:
                # The area under the curve, not its peak: a peak is one lucky
                # evaluation, and the area is what the run did throughout.
                values = {
                    i: summaries[(env, stat, i)]["auc"]
                    for i in table_entries
                    if (env, stat, i) in summaries
                    and summaries[(env, stat, i)]["auc"] is not None
                }
                prec = detect_precision(
                    [m for m, _ in values.values()], min_prec=2, max_prec=6
                )
                for i in table_entries:
                    if i not in values:
                        cells.append("---")
                        continue
                    mean, ci = values[i]
                    cells.append(f"{mean:.{prec}f} $\\pm$ {ci:.{prec}f}")
            rows.append(" & ".join(cells) + r" \\")

    rows += [r"\bottomrule", r"\end{tabular}"]
    tex_path = os.path.join(drawing.output_dir, "table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    vprint(f"  LaTeX table: {tex_path}")


def write_rliable_tables(drawing):
    """The rliable estimates of one config as LaTeX, one file per score kind
    beside its figures: the IQM of every configuration with its interval, then
    the probability of improvement of every pair, a column per statistic and a
    block per environment group. A cell with no estimate is `---`."""

    def cell(value, lo, hi):
        return f"{value:.2f} [{lo:.2f}, {hi:.2f}]"

    for kind, score_label in RLIABLE_SCORES.items():
        records = {
            (g, s): r for (k, g, s), r in drawing.rliable.items() if k == kind
        }
        if not records:
            continue
        stats = [s for s in drawing.stats if any(k[1] == s for k in records)]
        groups = list(dict.fromkeys(g for g, _ in records))
        header = (
            " & ".join(
                [""] + [
                    r"\textbf{" + tex_escape(drawing.stat_labels.get(s, s)) + "}"
                    for s in stats
                ]
            )
            + r" \\"
        )
        rows = []

        def block(title, keys_of, name_of, value_of):
            body = []
            for group_name in groups:
                keys = keys_of(group_name)
                if not keys:
                    continue
                if body:
                    body.append(r"\midrule")
                if group_name:
                    body.append(
                        r"\multicolumn{" + str(len(stats) + 1) + r"}{l}{\textit{"
                        + tex_escape(group_name) + r"}} \\"
                    )
                for key in keys:
                    cells = [name_of(key)]
                    for stat in stats:
                        value = value_of(records.get((group_name, stat), {}), key)
                        cells.append("---" if value is None else cell(*value))
                    body.append(" & ".join(cells) + r" \\")
            if not body:
                return
            rows.extend([
                f"% {title}",
                r"\begin{tabular}{l" + "c" * len(stats) + "}",
                r"\toprule",
                header,
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabular}",
                "",
            ])

        iqm_k = METRICS.index("IQM")

        def iqm_of(record, i):
            if i not in record.get("aggregate", {}):
                return None
            values, bands = record["aggregate"][i]
            return values[iqm_k], bands[0, iqm_k], bands[1, iqm_k]

        block(
            f"IQM of the normalized {score_label} score, 95% stratified-bootstrap CI",
            lambda grp: list(dict.fromkeys(
                i for (g, _), r in records.items() if g == grp
                for i in r.get("aggregate", {})
            )),
            lambda i: tex_escape(drawing.entries[i]["label"]),
            iqm_of,
        )
        block(
            f"Probability of improvement P(X > Y) on the {score_label} score, "
            f"95% stratified-bootstrap CI",
            lambda grp: list(dict.fromkeys(
                pair for (g, _), r in records.items() if g == grp
                for pair in r.get("poi", {})
            )),
            lambda pair: (
                tex_escape(drawing.entries[pair[0]]["label"]) + " vs. "
                + tex_escape(drawing.entries[pair[1]]["label"])
            ),
            lambda record, pair: record.get("poi", {}).get(pair),
        )
        if not rows:
            continue
        tex_path = os.path.join(drawing.output_dir, f"rliable_{kind}.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows))
        vprint(f"  LaTeX table: {tex_path}")


def recap(drawing):
    """What the runs cost: the average wall-clock minutes per configuration and
    environment."""

    entries = drawing.entries
    if not drawing.time_table:
        return
    shown = [
        i for i in range(len(entries))
        if any(k[1] == i for k in drawing.time_table)
    ]
    label_w = max(len(entries[i]["label"]) for i in shown)
    print(f"\nAvg run time (minutes) -- {drawing.name}")
    for group_name, group_envs in drawing.drawn_envs.items():
        envs_here = [
            e for e in group_envs if any((e, i) in drawing.time_table for i in shown)
        ]
        if not envs_here:
            continue
        names = [drawing.env_labels.get(str(e), str(e)) for e in envs_here]
        widths = [max(len(n), 6) for n in names]
        header = f"{'':>{label_w}}  " + "  ".join(
            f"{n:>{w}}" for n, w in zip(names, widths)
        )
        print(f"\n  {group_name or '(unnamed)'}\n  {'=' * len(header)}")
        print(f"  {header}\n  {'-' * len(header)}")
        for i in shown:
            cells = "  ".join(
                f"{drawing.time_table[(e, i)]:>{w}.1f}"
                if (e, i) in drawing.time_table
                else f"{'---':>{w}}"
                for e, w in zip(envs_here, widths)
            )
            print(f"  {entries[i]['label']:>{label_w}}  {cells}")
        print(f"  {'=' * len(header)}")


def write_legend(drawing):
    """The legend on its own, for a figure that has to carry one elsewhere.

    Both shapes are written every time, since which one fits depends on the
    document the figure goes into rather than on the data. The horizontal is laid
    out by `legend_rows_cols`; the vertical is always one entry per row."""

    # Filled blocks, not the line the curves are drawn with: at the size a legend
    # is read at, a bar of colour carries the colour better than a stroke of it.
    handles = [
        mpatches.Patch(facecolor=e["color"], label=e["label"])
        for e in drawing.entries
    ]
    n_rows, n_cols = rows_cols(drawing.cfg.get("legend_rows_cols"), len(handles))

    for name, cols in (("legend_horizontal", n_cols), ("legend_vertical", 1)):
        fig_legend = plt.figure()
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis("off")
        legend = ax_legend.legend(
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
        # by a rule of thumb is a floor the crop cannot go below -- the axes fills
        # it whether the legend does or not -- and one entry per row in a canvas
        # wide enough for four is all margin.
        fig_legend.canvas.draw()
        box = legend.get_window_extent().transformed(
            fig_legend.dpi_scale_trans.inverted())
        fig_legend.set_size_inches(box.width, box.height)
        vprint(f"  Saved: {save_figure(fig_legend, drawing.output_dir, name)}")
        plt.close(fig_legend)


for config_name, plot_cfg_path, config in plot_cfgs:
    vprint(f"\n{'#' * 60}\nPlot config: {config_name}\n{'#' * 60}")
    written += draw_config(config_name, config)

if args.prepare_missing_runs:
    path, n_jobs = write_missing_runs_script(args.folder, missing_runs)
    print(f"\nMissing runs script: {path} ({n_jobs} entries)")

print(
    f"\n{len(plot_cfgs)} plot config(s), {written} figure(s) -> "
    f"{os.path.join(args.folder, args.output, 'curves')}"
)
