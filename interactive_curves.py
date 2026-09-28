"""Training curves from a pre-built results.gzip, as one interactive page.

Writes <folder>/plots/<sweep>_curves.html: a self-contained Vega-Lite page with a
tab per environment and, inside a tab, a nested grid of subplots.

Within a tab there is a dropdown per curve hyperparameter and per statistic, plus
one for the seed -- "(all)" is the mean and its confidence band, a single seed is
that run's own curve.

"Show" picks what a subplot draws:

  Curves        the training curves themselves.
  AUC Bars      one bar per curve: the area under it divided by the training
                steps it covers, so the bar is in the units of the y axis and
                configurations trained for different lengths stay comparable.
  Time Bars     one bar per curve: the wall-clock minutes its runs took, averaged
                over its seeds. A run is timed once, whatever statistics it
                recorded, so these bars ignore the statistic dropdown -- and
                `time` itself is not one of the statistics it offers.
  Violin AUC    one violin per subplot: the density over the score of every run
  Violin max    the subplot pools -- one point per seed of every curve in it,
  Violin final  i.e. over the global_sweep combinations -- with the quartile box,
                a dot per run, and the mean over the best of them annotated. The
                three differ only in what a run scores: the area under its curve,
                its best point, or its last.

Every row ends in an extra "all" subplot, pooling every run of that row. In the
violin modes it is their density; in Curves, AUC Bars and Time Bars it is their
average.

The "avg. AUC" tab that follows the environments is the AUC bars with the
environment averaged away, and "parameters" regroups the same runs by one
hyperparameter at a time. Both are tabs, not modes of this dropdown.

After the environment tabs comes "parameters", which regroups the same
runs by one hyperparameter at a time, chosen from a dropdown. A row of its grid
is an environment and the last row pools them; a column is one of the three scores
(AUC, max, final); and a cell holds a violin per value that hyperparameter takes.
Only runs whose sweep entry VARIES it are counted (configurations that do not
declare the parameter or do not vary it are ignored).

"Log scale" puts the y axis into a symlog.

"Same y-lim" puts every cell of the tab on one y range, running from the lowest
value drawn in it to the highest. It is taken over what is on screen, so it
follows the statistic, the mode and the curve dropdowns.

Drag and the wheel move the x-axis, shift and either moves the y-axis, a block
at a time -- or the whole tab at once under "Same y-lim", which is one scale.

--sweep is required, and the layout mirrors it: one block of subplots per
config_sweep entry, subplots inside a block for what that entry varies, one curve
per global_sweep combination. Only the environments it declares are plotted,
unless some entry declares none, in which case every environment in the data is.
Hyperparameters it does not declare are averaged into one curve, with a warning.

A run is plotted when its whole configuration is one the sweep COMPOSES -- the
keys an entry names, plus everything the config files it selects and
configs/default.yaml fill in under them -- not merely when it agrees on the keys
the entry writes down. Runs matching nothing are reported and dropped. Composing
goes through Hydra, so this script needs hydra installed and CONFIG_DIR readable.

--prepare_missing_runs writes a script to submit jobs for missing runs (e.g.,
due to failed jobs). They are checked against the sweep, so a configuration it
declares that has no successful run is still detected -- including an environment
that left no run at all, which is reported and written out in full against the
highest seed found anywhere. The number of seed is always inferred from the
highest seed found.

--stats to filter the statistics to plot. By default it plots ALL, potentially
generating very large HTML files.

--include and --exclude to further filter configurations.

Example

    python interactive_curves.py -f data_example --sweep=example --stats train test td_err
"""

import argparse
import itertools
import json
import os
from ast import literal_eval

import numpy as np
import pandas as pd

from src.utils.plot import (
    ALGO,
    ENV,
    ENV_ID,
    SEED,
    TIME,
    SweepLayout,
    apply_filters,
    assign_configs,
    ci_bounds,
    declared_environments,
    html_page,
    hyperparameter_to_label,
    iter_cells,
    keep_composed_runs,
    keep_declared_environments,
    label_of,
    launch_overrides,
    load_config_group,
    mask_of,
    record_missing,
    select_rows,
    series_of,
    split_config_label,
    swept_keys,
    value_str,
    vega_field_name,
    write_missing_runs_script,
)
from src.utils.sweep import load_sweep

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--folder",
    required=True,
    metavar="DIR",
    help="Data directory to read.",
)
parser.add_argument("-g", "--gzip", default="results.gzip")
parser.add_argument(
    "-e", "--environments",
    default="configs/environment",
    help="Directory of environment YAMLs.",
)
parser.add_argument(
    "-a", "--algorithms",
    default="configs/algorithm",
    help="Directory of algorithm YAMLs.",
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
    "--sweep",
    required=True,
    help="Sweep file in SWEEP_DIR (name without .yaml, or a path). It decides the "
         "layout: one block per config_sweep entry, and an axis per key it "
         "declares.",
)
parser.add_argument(
    "--include",
    nargs="+",
    default=[],
    metavar="KEY=VALUE",
    help="Keep only matching rows. Same KEY twice is OR, different KEYs are AND.",
)
parser.add_argument(
    "--exclude",
    nargs="+",
    default=[],
    metavar="KEY=VALUE",
    help="Drop matching rows.",
)
parser.add_argument(
    "--top_fraction",
    type=float,
    default=0.2,
    help="Fraction of the best runs a violin's annotated mean is taken over. "
         "0 to annotate the mean over all of them.",
)
parser.add_argument(
    "--prepare_missing_runs",
    action="store_true",
    help="Write a submit script for configurations missing seeds below the "
         "highest seed of that environment.",
)
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

# Moving-average half-widths the slider offers, evenly spaced so it can step
# through them. A window covers 2 * span + 1 points, and 0 is the raw curve.
SMOOTH_SPANS = [0, 5, 10, 15, 20]

# One cell of the grid.
CELL_WIDTH = 160
CELL_HEIGHT = 120

# What a subplot draws. Every filter and expression that names a mode names it
# from here: a mode spelled out twice is a mode that can be renamed in one place
# and quietly stop selecting anything in the other.
#
# The violin modes are told apart by their shared prefix rather than by being
# listed, since which score they take is the only thing that separates them.
VIOLIN_PREFIX = "Violin"
MODES = [
    "Curves",
    "AUC Bars",
    "Time Bars",
    f"{VIOLIN_PREFIX} AUC",
    f"{VIOLIN_PREFIX} max",
    f"{VIOLIN_PREFIX} final",
]

# The name of the column that pools a whole row.
ALL_COL = "all"
# What the pooled column is drawn in: one colour of its own, since it is not one
# of the curves the legend names but the average over all of them.
POOLED_COLOR = "#444444"

# TIME is where a run records how long it took: one number for the whole run, in
# seconds, which is why it cannot be a curve and gets a mode of its own.

# How many recorded times did not read as a finite number. Timing bars missing
# because the values did not decode look the same as bars missing because no run
# was timed, so the two cases are counted separately and reported.
unreadable_times = 0

# The curves whose seeds were scored over different numbers of points (see
# count_ragged). Nothing on the page shows it, so they are collected and
# reported. A set of curves rather than a count: rows_of runs once per
# (curve, statistic), and a curve ragged in four statistics is one ragged curve.
ragged_curves = set()


def vprint(*a):
    """Print only under --verbose."""

    if args.verbose:
        print(*a)


def axes_of(block, row_vals, col_vals, curve_vals):
    """The three axes a curve sits on, as (hyperparameter columns, their values)
    pairs, so a caller can walk all of them the same way.

    Together they are what identifies a curve: everything the rows, columns and
    curves of this block vary. Callers that need one flat mapping of
    hyperparameter to value build it by updating over these three."""

    return (
        (block.row_params, row_vals),
        (block.col_params, col_vals),
        (curve_params, curve_vals),
    )


def iter_curves(cell):
    """Walk a cell's curves, yielding (index, values, rows) for each one that has
    runs behind it.

    The index is the curve's position among every combination the figure draws, not
    among the ones this cell happens to hold, so a curve keeps its colour and its
    place across cells and a cell missing one leaves a gap rather than shifting the
    rest along."""

    for i, curve_vals in enumerate(curve_combos):
        curve = cell[mask_of(cell, curve_params, curve_vals)]
        if not curve.empty:
            yield i, curve_vals, curve


def rows_of(frame, stat):
    """The rows one curve contributes and the seeds behind them, as
    `(rows, seeds)`: mean/CI through the same ci_bounds the figures use, one raw
    value per (seed, x), and one area for the bars.

    `seeds` is what actually became a curve, which is not what the frame holds:
    series_of drops a seed recorded twice and one whose curve has no finite
    point, and counting the frame's rows would report more seeds than the band
    was drawn from. It is returned rather than decoded again by the caller --
    decoding is the expensive part of this function.

    The values are the raw ones. Smoothing is a moving average over the points
    either side, which a Vega-Lite window transform does in the page -- carrying a
    copy of every curve per window would multiply the data instead."""

    decoded = series_of(frame, stat)
    if decoded is None:
        return [], []
    series, stepsize, _, seeds = decoded
    # `alive` is carried, not dropped: it is how many seeds still had a value at
    # that point. Where it is 1 the half-width is 0 and the band is drawn from a
    # single run, so it is kept per point and shown in the tooltip.
    mean, half, alive = ci_bounds(series.to_numpy().T)
    rows = [
        {
            "x": i * stepsize,
            "mean": float(m),
            "ci_lo": float(m - h),
            "ci_hi": float(m + h),
            "alive": int(n),
            "_kind": "agg",
        }
        for i, (m, h, n) in enumerate(zip(mean, half, alive)) if not np.isnan(m)
    ]
    for j, seed in enumerate(seeds):
        seed_id = int(seed) if seed is not None and pd.notna(seed) else None
        rows += [
            {"x": i * stepsize, "value": float(v), "seed": seed_id, "_kind": "seed"}
            for i, v in enumerate(series.iloc[:, j]) if pd.notna(v)
        ]
    # Scored once and used twice: the bar is the average of these, and the violin
    # is their density.
    scores = scores_of(series)
    count_ragged(frame, scores)
    # One row per curve for the bars: the same area the --auc figure draws, so the
    # two agree, and it is over every seed by construction.
    area = auc_of(scores)
    if area is not None:
        auc_mean, auc_half = area
        rows.append({
            "auc": auc_mean,
            "auc_lo": auc_mean - auc_half,
            "auc_hi": auc_mean + auc_half,
            "_kind": "auc",
        })
    # One row per run for the violins, holding the score. A violin is the density
    # over these, so it needs the runs themselves rather than the average of them
    # the bar carries.
    rows += [
        {**{k: v for k, v in s.items() if not k.startswith("_")},
         "_run": k, "_kind": "violin"}
        for k, s in enumerate(scores)
    ]
    return rows, seeds


def scores_of(series):
    """The three scores of every seed of a curve, one dict per run.

    `score_auc` is the mean over the curve, i.e. its area divided by the span it
    covers, which rewards getting there early as well as ending high;
    `score_final` is where it ended;
    `score_max` its best point.
    All three are carried, and the page picks between them -- they are three
    numbers per run against a curve's hundreds.

    `_n_finite`, which is how many points each was taken over, comes with them
    for count_ragged; it is stripped before the scores reach a row.

    Non-finite points are dropped rather than propagated: one inf anywhere in a
    curve would carry through the mean and end up as an axis limit. A seed with no
    finite point at all scores nothing and is left out."""

    scores = []
    for j in range(series.shape[1]):
        values = series.iloc[:, j].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            scores.append({
                "score_auc": float(values.mean()),
                "score_max": float(values.max()),
                "score_final": float(values[-1]),
                "_n_finite": int(len(values)),
            })
    return scores


def count_ragged(frame, scores):
    """Note a curve whose seeds were scored over different numbers of points.

    `score_auc` is the mean over the points a seed HAS, so a seed that stopped
    early is scored over its own span rather than over the training run. That is
    not a like-for-like comparison with the seeds beside it -- a run still
    climbing when it died scores higher for having died -- and nothing on the
    page says it happened, so it is collected here and reported per environment.

    Noted by the runs the curve is drawn from, so a curve ragged in several
    statistics is counted once: it is one curve either way, and the report says
    curves."""

    if len({s["_n_finite"] for s in scores}) > 1:
        ragged_curves.add(tuple(frame.index))


def auc_of(scores):
    """Reduce a cell's seeds to one number and return (mean area under the curve,
    half-width of its 95% confidence interval), or None when no seed has a curve.

    `scores` is what scores_of returned for the cell. The area is divided by the
    span it covers, so it comes out in the units of the y axis -- the average
    value over training rather than a number that grows with the number of steps.
    With evenly spaced points that is the mean of the curve, which also skips the
    gaps a seed that died early leaves behind (see count_ragged).

    Taken per seed and then averaged, so the interval says how much the seeds
    disagreed, the same thing the shaded band around the curves says."""

    per_seed = np.array([s["score_auc"] for s in scores])
    if not len(per_seed):
        return None
    mean, half, _alive = ci_bounds(per_seed.reshape(-1, 1))
    return float(mean[0]), float(half[0])


def time_of(frame):
    """The wall-clock row one curve contributes: the mean minutes its runs took
    and the half-width of the 95% interval over them, or None when nothing was
    timed.

    One row per curve, not per (curve, statistic): a run is timed once whatever
    it recorded, so the row carries no statistic and the bars drawn from it stay
    put as the statistic dropdown moves.

    Seeds are deduplicated the way series_of does it, for the same reason -- a
    (configuration, seed) recorded twice would count one run's time twice.

    A duration is read as a number the column already holds or as a JSON scalar
    encoding one, since the statistic block is written both ways. Anything else,
    a list among them, is counted as unreadable rather than guessed at: there is
    no reading of several numbers as one duration that is not an assumption."""

    global unreadable_times

    if TIME not in frame.columns:
        return None
    if SEED in frame.columns:
        # Only among the rows that record one: drop_duplicates counts two NaNs
        # as the same value, so runs recording no seed would collapse into one.
        seeded = frame[SEED].notna()
        frame = pd.concat([
            frame[seeded].drop_duplicates(subset=[SEED]),
            frame[~seeded],
        ])
    minutes = []
    for raw in frame[TIME].dropna():
        numeric = isinstance(raw, (int, float, np.integer, np.floating))
        if numeric and not isinstance(raw, bool):
            value = float(raw)
        else:
            try:
                decoded = json.loads(raw)
            except (ValueError, TypeError):
                decoded = None
            value = (
                float(decoded)
                if isinstance(decoded, (int, float)) and not isinstance(decoded, bool)
                else None
            )
        if value is None or not np.isfinite(value):
            # One count for both: a value that did not parse and an infinity are
            # equally absent from the bars.
            unreadable_times += 1
        else:
            # Recorded in seconds, drawn in minutes: converted here, so the bar,
            # its interval and the floor under it are all in the same unit.
            minutes.append(value / 60.0)
    if not minutes:
        return None
    mean, half, _alive = ci_bounds(np.array(minutes).reshape(-1, 1))
    return {
        "time": float(mean[0]),
        "time_lo": float(mean[0] - half[0]),
        "time_hi": float(mean[0] + half[0]),
        "n_times": len(minutes),
        "_kind": "time",
    }


def declared_rows(df_env, env_stem):
    """Walk what the sweep declares for `env_stem`, yielding (configuration, its
    rows) for every combination, run or not.

    A configuration whose runs all failed has no row to be found by, and those are
    the ones worth relaunching, so the walk goes over the sweep rather than over
    the data.

    A sweep may declare several environments while this is called once per
    environment, so combinations belonging to another environment are skipped
    rather than counted as missing here."""

    for entry in sweep:
        keys = list(global_sweep) + [k for k in entry if k not in global_sweep]
        declared = [entry.get(k, global_sweep.get(k)) for k in keys]
        for combo in itertools.product(*declared):
            hp = dict(zip(keys, combo))
            if "environment" in hp and value_str(hp["environment"]) != env_stem:
                continue
            # One value per key here, not a list of alternatives -- and a value
            # can itself be a list, so wrap before selecting.
            mask, _ = select_rows(df_env, {k: [v] for k, v in hp.items()})
            yield hp, df_env[mask]


def algorithm_of(frame):
    """The config group these rows were launched with, for relaunching them --
    submit_jobs needs an `algorithm=` and the runs do not record one.

    None when nothing resolved it, and None when the rows disagree, which happens
    when the sweep does not declare the algorithm and several ran: better to omit
    it from the relaunch than to pick one of them."""

    if ALGO not in frame.columns or frame.empty:
        return None
    values = frame[ALGO].dropna().unique()
    return values[0] if len(values) == 1 else None


def seeds_of(frame):
    """Return the seeds present in `frame`, sorted and deduplicated, or an empty
    list when the data records no seed at all."""

    if SEED not in frame.columns:
        return []
    return sorted(int(s) for s in frame[SEED].dropna().unique())


def pooled_rows(rows):
    """The rows of the pooled column: one averaged curve and one averaged bar per
    (block, row, statistic), plus one averaged timing bar per (block, row), from
    the rows already built for the cells.

    The average is over every configuration of the row and every curve under
    them -- of the means, and of the bands. A band over the averages would say
    how much the configurations differed, which is what the cells beside it
    already show; the average of the bands carries through how much the seeds of
    each disagreed.

    Only the fields the pooled layers draw come along. A copy of the cells' own
    rows would carry the curve, its index and its seed count into a cell that
    draws one line."""

    curves, areas, times = {}, {}, {}
    for r in rows:
        if r["_kind"] == "agg":
            key = (r["_block"], r["_row"], r["statistic"])
            acc = curves.setdefault(key, {}).setdefault(r["x"], [0.0, 0.0, 0.0, 0])
            acc[0] += r["mean"]
            acc[1] += r["ci_lo"]
            acc[2] += r["ci_hi"]
            acc[3] += 1
        elif r["_kind"] == "auc":
            key = (r["_block"], r["_row"], r["statistic"])
            acc = areas.setdefault(key, [0.0, 0.0, 0.0, 0])
            acc[0] += r["auc"]
            acc[1] += r["auc_lo"]
            acc[2] += r["auc_hi"]
            acc[3] += 1
        elif r["_kind"] == "time":
            # Keyed without the statistic, which a timing row does not carry.
            acc = times.setdefault((r["_block"], r["_row"]), [0.0, 0.0, 0.0, 0])
            acc[0] += r["time"]
            acc[1] += r["time_lo"]
            acc[2] += r["time_hi"]
            acc[3] += 1

    out = []
    for (block, row, stat), per_x in curves.items():
        out += [
            {
                "_block": block,
                "_row": row,
                "_col": ALL_COL,
                "statistic": stat,
                "_kind": "agg_all",
                "x": x,
                "mean": s[0] / s[3],
                "ci_lo": s[1] / s[3],
                "ci_hi": s[2] / s[3],
                # How many curves reached this x. The average is over the curves
                # that HAVE a point here, not over every curve of the row, so
                # where one of them ends the line steps to the average of the
                # rest -- the same thing `alive` says about a band and a seed
                # that died early. Carried per point and shown in the tooltip.
                "n_curves": s[3],
            }
            for x, s in sorted(per_x.items())
        ]
    out += [
        {
            "_block": block,
            "_row": row,
            "_col": ALL_COL,
            "statistic": stat,
            "_kind": "auc_all",
            "auc": s[0] / s[3],
            "auc_lo": s[1] / s[3],
            "auc_hi": s[2] / s[3],
        }
        for (block, row, stat), s in areas.items()
    ]
    out += [
        {
            "_block": block,
            "_row": row,
            "_col": ALL_COL,
            "_kind": "time_all",
            "time": s[0] / s[3],
            "time_lo": s[1] / s[3],
            "time_hi": s[2] / s[3],
        }
        for (block, row), s in times.items()
    ]
    return out


def recap(env_stem, df_env):
    """Print which seeds are on disk per configuration.

    The configurations come from the sweep file, so one that never ran is listed
    with no seeds instead of being absent."""

    declared = list(declared_rows(df_env, env_stem))
    keys = [
        k for k in dict.fromkeys(k for hp, _ in declared for k in hp)
        if len({value_str(hp.get(k)) for hp, _ in declared}) > 1
    ]
    # A sweep declaring one configuration has no key to tell it from another, so
    # it is named by its environment rather than left as a blank row.
    rows = [
        (
            ", ".join(f"{LABELS[k]}={value_str(hp.get(k))}" for k in keys)
            or env_stem,
            seeds_of(sub),
        )
        for hp, sub in declared
    ]

    if not rows:
        return
    empty = sum(1 for _, seeds in rows if not seeds)
    width = max(len(label) for label, _ in rows) + 2
    num_width = len(str(len(rows))) + 2
    print(f"\n{len(rows)} configuration(s), {empty} with no run at all")
    print(f"{'':<{num_width}} {'Configuration':<{width}} Seeds found")
    print(f"{'-' * num_width} {'-' * width} {'-' * 20}")
    for i, (label, seeds) in enumerate(rows, 1):
        print(f"{'[' + str(i) + ']':<{num_width}} {label:<{width}} {seeds}")


# How a panel of this page is drawn, handed to plot.html_page and dropped into
# the shell it keeps (which defines `panels`, `views`, `embedInto` and `show`).
RENDER_JS = """
  // Which axis each panel is showing, and whether its cells share one y scale.
  // A scale's type and resolve are fixed when the spec is compiled, so the boxes
  // pick between specs rather than moving a signal.
  const logAxis = [];
  const sameY = [];

  // Everything the reader has chosen, carried across the re-embed so that ticking
  // the box changes the axis and nothing else. Read off the spec rather than
  // listed here: a panel has a dropdown per curve hyperparameter, and those are
  // named after the data, so a list written by hand drops exactly the ones the
  // reader was using.
  function carriedNames(i) {
    return (panels[i].spec.params || []).filter(p => p.bind).map(p => p.name);
  }

  function render(i, carry) {
    const spec = panels[i]['spec' + (logAxis[i] ? '_log' : '') + (sameY[i] ? '_same_y' : '')]
                 || panels[i].spec;
    // The alternative spec ships without rows; it shares the ones already here.
    if (!spec.data) { spec.data = panels[i].spec.data; }
    if (carry) {
      spec.params.forEach(p => {
        if (p.name in carry) { p.value = carry[p.name]; }
      });
    }
    return embedInto(i, spec).then(function (view) {
      // Undefined when the spec was rejected; the shell has put the error where
      // the plots go.
      if (!view) { return; }
      // No alternative spec, no control: a panel with one axis would otherwise
      // offer a box that re-embeds the same chart.
      if (!panels[i].spec_log) { return; }
      const controls = document.getElementById('controls' + i);
      [[logAxis, 'Log scale'], [sameY, 'Same y-lim']].forEach(([state, text]) => {
        const label = document.createElement('label');
        const box = document.createElement('input');
        box.type = 'checkbox';
        box.checked = !!state[i];
        box.onchange = () => {
          state[i] = box.checked;
          const carried = {};
          carriedNames(i).forEach(name => {
            try { carried[name] = view.signal(name); } catch (e) {}
          });
          render(i, carried);
        };
        label.appendChild(box);
        label.appendChild(document.createTextNode(' ' + text));
        controls.appendChild(label);
      });
    });
  }

  // Nothing to redo when a tab is shown again: a panel keeps the view and the
  // controls it was drawn with.
  function redraw(i) {}
"""


def sorted_values(values):
    """Hyperparameter values in numeric order where they all are numbers, textual
    order otherwise -- the order every other axis of the page puts them in."""

    try:
        return sorted(values, key=float)
    except (TypeError, ValueError):
        return sorted(values, key=str)


def build_avg_auc_spec(rows, block_labels, curve_domain):
    """Build the panel that averages the AUC bars over the environments, and
    return it, or None when there is nothing to average.

    The environment tabs each answer "how did this configuration do here"; this
    one answers "how did it do overall", which is the number a table in a paper
    reports. The grid is the same, so a bar sits where its curve sat, in the same
    colour: only the environment dimension is gone, averaged away.

    The interval is the average of the per-environment intervals, so it carries
    how much the SEEDS disagreed, not how much the environments did -- two
    environments of different difficulty would otherwise show as uncertainty
    about a configuration that is behaving consistently in both."""

    if not rows:
        return None
    frame = pd.DataFrame(rows)
    keys = ["statistic", "_block", "_row", "_col", "_curve", "_curve_i"]
    averaged = frame.groupby(keys, as_index=False).agg(
        auc=("auc", "mean"),
        auc_lo=("auc_lo", "mean"),
        auc_hi=("auc_hi", "mean"),
        n_envs=("_env", "nunique"),
    )

    # The same floor the per-environment bars stand on, for the same reason: a
    # negative average is a real result, and a bar hanging from zero beside one
    # standing on it compares badly.
    lowest = averaged.groupby("statistic")["auc_lo"].transform("min")
    averaged["auc_base"] = lowest - 0.1 * lowest.abs()

    statistics = sorted(averaged["statistic"].unique())
    # The pooled column goes last, as it does on the environment tabs.
    col_orders = {
        int(b): sorted(
            g["_col"].dropna().unique().tolist(),
            key=lambda c: (c == ALL_COL, str(c)),
        )
        for b, g in averaged.groupby("_block")
    }
    curve_labels = [
        c for _, c in sorted(
            {(int(i), c) for i, c in zip(averaged["_curve_i"], averaged["_curve"])}
        )
    ]

    def block_spec(index, label):
        return {
            "title": label or None,
            "transform": [{"filter": f"datum._block === {index}"}],
            "facet": {
                "row": {
                    "field": "_row",
                    "type": "nominal",
                    "title": None,
                    "header": {"labelAngle": 0, "labelAlign": "left"},
                },
                "column": {
                    "field": "_col",
                    "type": "nominal",
                    "title": None,
                    "sort": col_orders.get(index, "ascending"),
                },
            },
            "spec": {
                "width": CELL_WIDTH,
                "height": CELL_HEIGHT,
                "transform": [{"filter": "datum.statistic === p_stat"}],
                "layer": [
                    {
                        "mark": {
                            "type": "bar",
                            "stroke": "black",
                            "strokeWidth": 0.6,
                        },
                        "encoding": {
                            # Ordinal, not the training steps the per-environment
                            # bars stand in: an average over environments has no
                            # step axis to stand on, and the curve order is the
                            # only thing the position has to carry.
                            "x": {
                                "field": "_curve",
                                "type": "nominal",
                                "axis": None,
                                "sort": curve_labels,
                            },
                            "y": {
                                "field": "auc",
                                "type": "quantitative",
                                "title": None,
                                "scale": {"zero": False},
                            },
                            "y2": {"field": "auc_base"},
                            "color": {
                                "field": "_curve",
                                "type": "nominal",
                                "title": None,
                                "scale": {"domain": curve_domain},
                                "legend": {"values": curve_labels},
                            },
                            "tooltip": [
                                {
                                    "field": "_curve",
                                    "type": "nominal",
                                    "title": "curve",
                                },
                                {
                                    "field": "auc",
                                    "type": "quantitative",
                                    "title": "mean area",
                                    "format": ".3f",
                                },
                                {
                                    "field": "n_envs",
                                    "type": "quantitative",
                                    "title": "environments",
                                },
                            ],
                        },
                    },
                    {
                        "mark": {"type": "rule", "strokeWidth": 0.8},
                        "encoding": {
                            "x": {
                                "field": "_curve",
                                "type": "nominal",
                                "axis": None,
                                "sort": curve_labels,
                            },
                            "y": {
                                "field": "auc_lo",
                                "type": "quantitative",
                                "title": None,
                            },
                            "y2": {"field": "auc_hi"},
                        },
                    },
                ],
            },
            "resolve": {"scale": {"y": "independent"}},
        }

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {
            "values": [
                {k: v for k, v in record.items() if v == v}
                for record in averaged.to_dict(orient="records")
            ]
        },
        "params": [{
            "name": "p_stat",
            "value": statistics[0],
            "bind": {
                "input": "select",
                "options": statistics,
                "name": "Statistic: ",
            },
        }],
        "vconcat": [block_spec(i, label) for i, label in block_labels],
        # One legend for the whole tab, above the blocks and unabbreviated, as on
        # the environment tabs: at the side it is cut to the width of a column,
        # and these labels are whole configurations.
        "resolve": {"scale": {"color": "shared"}},
        "config": {
            "view": {"stroke": "#888"},
            "axis": {"grid": False},
            "legend": {"orient": "top", "labelLimit": 0, "symbolOpacity": 1},
        },
    }


def build_param_spec(rows):
    """The last panel: one hyperparameter at a time, whatever varies under it.

    A row of the grid is an environment and the last row pools them; a column is
    one of the three scores; and a cell holds a violin per value the chosen
    hyperparameter takes. So it answers "does this setting matter, and the same
    way everywhere" in one picture, where the environment tabs answer it one
    environment and one sweep entry at a time.

    Only runs whose sweep entry VARIES the hyperparameter are in it (see where
    `param_rows` is filled). An entry that pins it is not evidence about it, and
    one that never declared it at all -- an algorithm the hyperparameter does not
    apply to, say -- would otherwise pile all its runs onto whichever value came
    first."""

    # A hyperparameter belongs in the dropdown only if the runs disagree about
    # it: one that came out the same everywhere is a single violin and nothing to
    # compare it with.
    reserved = ("_env", "statistic", "_run", "score_auc", "score_max", "score_final")
    candidates = {}
    for key in sorted({k for r in rows for k in r if k not in reserved}):
        values = sorted_values({r[key] for r in rows if key in r})
        if len(values) > 1:
            candidates[key] = values
    if not candidates:
        return None

    names = {k: vega_field_name(LABELS, k) for k in candidates}
    stats = sorted({r["statistic"] for r in rows})

    # One record per run, carrying the value and the slot it takes for every
    # hyperparameter it varies -- and nothing at all for the ones it does not, so
    # `_val` comes out null there and the run drops out.
    #
    # The jitter is computed here, from the golden-ratio sequence over the record
    # index: an expression using random() would re-scatter the dots on every
    # signal change.
    values = []
    for i, r in enumerate(rows):
        rec = {k: r[k] for k in reserved}
        rec["_j"] = ((i * 0.6180339887498949) % 1.0) - 0.5
        for key, seen in candidates.items():
            if key in r:
                rec[f"hp_{names[key]}"] = r[key]
                rec[f"i_{names[key]}"] = seen.index(r[key])
        values.append(rec)
    # The pooled row is every run again under one name, so the last row of the
    # grid asks the question of the whole sweep.
    values += [{**rec, "_env": ALL_COL} for rec in values]

    def by_param(prefix):
        """The chosen hyperparameter's `prefix_` column, as a ternary over the
        candidates: one dropdown reaching one column per hyperparameter."""

        return " : ".join(
            f"p_param === '{names[k]}' ? datum.{prefix}_{names[k]}" for k in candidates
        ) + " : null"

    # Slots are shared across hyperparameters, so the tick labels are padded to
    # the widest list: a Vega expression that indexes past the end of a short one
    # would print "undefined" under the empty slots.
    widest = max(len(v) for v in candidates.values())
    ticks = " : ".join(
        f"p_param === '{names[k]}' ? {json.dumps(v + [''] * (widest - len(v)))}"
        for k, v in candidates.items()
    ) + " : []"
    x_enc = {
        "type": "quantitative",
        "title": None,
        "scale": {"zero": False},
        "axis": {
            "tickMinStep": 1,
            "labelAngle": 0,
            "grid": False,
            "labelExpr": f"datum.value === floor(datum.value) && datum.value >= 0"
                         f" ? ({ticks})[datum.value] : ''",
        },
    }
    y_enc = {"type": "quantitative", "title": None, "scale": {"zero": False}}
    cell_width = min(460, max(180, 52 * widest))

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": values},
        "params": [
            {
                "name": "p_param",
                "value": names[next(iter(candidates))],
                "bind": {
                    "input": "select",
                    "options": list(names.values()),
                    "name": "Hyperparameter: ",
                },
            },
            {
                "name": "p_stat",
                "value": stats[0],
                "bind": {
                    "input": "select",
                    "options": stats,
                    "name": "Statistic: ",
                },
            },
        ],
        # Cut before the facet, not inside it: a run that does not vary the chosen
        # hyperparameter has no slot to be drawn in, so there is no cell to empty.
        "transform": [
            {"filter": "datum.statistic === p_stat"},
            {"calculate": by_param("hp"), "as": "_val"},
            {"calculate": by_param("i"), "as": "_i"},
            {"filter": "datum._val != null"},
            {
                "fold": ["score_auc", "score_max", "score_final"],
                "as": ["_score_kind", "score"],
            },
        ],
        "facet": {
            "row": {
                "field": "_env",
                "type": "nominal",
                "sort": sorted(dict.fromkeys(r["_env"] for r in rows)) + [ALL_COL],
                "title": None,
                "header": {"labelAngle": 0, "labelAlign": "left"},
            },
            "column": {
                "field": "_score_kind",
                "type": "nominal",
                "title": None,
                "sort": ["score_auc", "score_max", "score_final"],
                "header": {
                    "labelExpr": "datum.value === 'score_auc' ? 'AUC'"
                                 " : datum.value === 'score_max'"
                                 " ? 'max' : 'final'",
                },
            },
        },
        "spec": {
            "width": cell_width,
            "height": CELL_HEIGHT,
            "layer": [
                # One violin per slot, each normalised by its own peak so a value
                # with two runs is as wide as one with twenty -- the shape is the
                # point, and the dots say how many there were.
                {
                    "transform": [
                        {
                            "density": "score",
                            "groupby": ["_i"],
                            "steps": 48,
                            "as": ["_v", "_d"],
                        },
                        {
                            "joinaggregate": [
                                {"op": "max", "field": "_d", "as": "_dmax"},
                            ],
                            "groupby": ["_i"],
                        },
                        {
                            "calculate": "datum._i - datum._d / datum._dmax * 0.4",
                            "as": "_x_lo",
                        },
                        {
                            "calculate": "datum._i + datum._d / datum._dmax * 0.4",
                            "as": "_x_hi",
                        },
                    ],
                    "mark": {
                        "type": "area",
                        "orient": "horizontal",
                        "opacity": 0.55,
                        "stroke": "#333",
                        "strokeWidth": 0.6,
                        "clip": True,
                    },
                    "encoding": {
                        "y": {"field": "_v", **y_enc},
                        "x": {"field": "_x_lo", **x_enc},
                        "x2": {"field": "_x_hi"},
                        "color": {"value": "#7f7f7f"},
                        # One path per slot. Without it the areas of two slots
                        # are one shape, drawn as a band of stripes across the
                        # cell -- an area mark joins whatever shares its colour.
                        "detail": {"field": "_i", "type": "nominal"},
                    },
                },
                # The quartiles and the median, as in the environment tabs.
                {
                    "transform": [
                        {
                            "aggregate": [
                                {"op": "q1", "field": "score", "as": "_q1"},
                                {"op": "q3", "field": "score", "as": "_q3"},
                                {"op": "median", "field": "score", "as": "_med"},
                            ],
                            "groupby": ["_i"],
                        },
                        {"calculate": "datum._i - 0.06", "as": "_m_lo"},
                        {"calculate": "datum._i + 0.06", "as": "_m_hi"},
                    ],
                    "layer": [
                        {
                            "mark": {
                                "type": "rule",
                                "strokeWidth": 7,
                                "color": "#4d4d4d",
                                "clip": True,
                            },
                            "encoding": {
                                "y": {"field": "_q1", **y_enc},
                                "y2": {"field": "_q3"},
                                "x": {"field": "_i", **x_enc},
                            },
                        },
                        {
                            "mark": {
                                "type": "rule",
                                "strokeWidth": 1.5,
                                "color": "white",
                                "clip": True,
                            },
                            "encoding": {
                                "y": {"field": "_med", **y_enc},
                                "x": {"field": "_m_lo", **x_enc},
                                "x2": {"field": "_m_hi"},
                            },
                        },
                    ],
                },
                # The runs themselves, scattered across the slot.
                {
                    "transform": [
                        {"calculate": "datum._i + 0.3 * datum._j", "as": "_dot_x"},
                    ],
                    "mark": {
                        "type": "point",
                        "filled": True,
                        "size": 14,
                        "opacity": 0.7,
                        "color": "#262626",
                        "clip": True,
                    },
                    "encoding": {
                        "y": {"field": "score", **y_enc},
                        "x": {"field": "_dot_x", **x_enc},
                        "tooltip": [
                            {"field": "_val", "type": "nominal", "title": "value"},
                            {
                                "field": "score",
                                "type": "quantitative",
                                "title": "score",
                                "format": ".3f",
                            },
                        ],
                    },
                },
            ],
        },
        "resolve": {"scale": {"y": "independent"}},
        "config": {"view": {"stroke": "#ccc"}, "axisY": {"minExtent": 30}},
    }


def build_spec(
    long_df,
    filter_fields,
    block_labels,
    curve_domain,
    log_axis=False,
    same_y=False,
):
    """Vega-Lite spec: one dropdown-bound param per filter field, combined into
    one expression applied inside the layer marks. Filtering before the facet
    would delete whole columns from the grid instead of emptying cells.

    `block_labels` is [(sweep entry index, its label)]. The index is what a row
    carries in `_block`, and it is the entry's own number rather than the block's
    place in the list -- an entry with no runs in this environment leaves no
    block, and the numbers have to mean the same thing in every environment.

    `curve_domain` is every curve of every environment, in the order the colour
    scheme is handed out in, so a curve is one colour on every tab."""

    # An extra column at the end of every row, holding that row's runs over
    # again: one violin per configuration says how each did, and this one says
    # how the row did.
    #
    # Only the violins are copied. A density is over the runs themselves, so the
    # rows have to be here to be counted; the pooled curve and bar were averaged
    # in Python (see pooled_rows) and arrive already summarised.
    pooled = long_df[long_df["_kind"] == "violin"].assign(_col=ALL_COL)
    long_df = pd.concat([long_df, pooled], ignore_index=True)

    # Plain ints: the seeds reach the spec through json.dumps, which has no
    # encoder for numpy scalars.
    recorded_seeds = long_df.get("seed", pd.Series(dtype=float)).dropna().unique()
    seeds = sorted({int(s) for s in recorded_seeds})

    # The colour scale is told its domain rather than left to read it off the
    # data. In Violins mode nothing on screen is coloured by curve, and a colour
    # scale whose every source is filtered out leaves an empty legend that takes
    # the layout to infinity with it. It also pins a curve to one colour in every
    # mode and every block, however few of them a block draws.
    #
    # The domain is `curve_domain`, every environment's curves, so a curve is one
    # colour on every tab. The legend lists only this environment's, in
    # `_curve_i` order: the order the bars stand in, which it should read left
    # to right with.
    curve_labels = [
        c for _, c in sorted({
            (int(i), c)
            for i, c in zip(long_df["_curve_i"], long_df["_curve"])
            if pd.notna(i) and pd.notna(c)
        })
    ]

    # The pooled column goes last; the rest keep the alphabetical order the facet
    # would have given them on its own.
    col_orders = {
        int(b): sorted(
            g["_col"].dropna().unique().tolist(),
            key=lambda c: (c == ALL_COL, str(c)),
        )
        for b, g in long_df.groupby("_block")
    }

    # Every mode is drawn on the curves' x scale: a scale-bound zoom cannot
    # live on a channel that resolves independently, so the bars, the timing bars
    # and the violins cannot have an x scale of their own. A shared scale takes
    # its domain from whatever is on screen, which would be a different domain
    # per mode and per block -- and one the zoom pins to the training steps the
    # moment the view is dragged. So the bars, the timing bars and the violins
    # are placed IN training steps, across the run they summarise, and two
    # invisible points per block hold the domain at exactly that span whichever
    # mode is showing.
    x_max = float(long_df["x"].max()) if long_df["x"].notna().any() else 1.0
    if x_max <= 0:
        x_max = 1.0
    first_cell = long_df.groupby("_block")[["_row", "_col"]].first()
    anchors = [
        {"_block": int(b), "_row": row, "_col": col, "_anchor": 1, "x": x}
        for b, (row, col) in first_cell.iterrows()
        for x in (0.0, x_max)
    ]

    # The span is cut into one slot per curve INDEX rather than per curve present,
    # so a cell missing a curve leaves a gap instead of shifting the rest along,
    # and every block puts the same curve in the same place. A bar is 80% of its
    # slot: wide enough to read, never wide enough to touch its neighbour.
    bar_step = x_max / (int(long_df["_curve_i"].max()) + 2)

    def bar_at(side):
        """The x of a bar's left edge, middle or right edge, as an expression."""

        return f"{bar_step} * (datum._curve_i + 1) + {side * 0.4 * bar_step}"

    # The violin stands in the middle of the same span, four fifths of it wide at
    # its widest, with the quartile box on its centre line.
    v_mid = x_max / 2
    v_half = 0.4 * x_max
    # The runs are scattered across the middle of it rather than stacked on the
    # centre line, so runs that scored alike stay countable. The offsets are the
    # golden-ratio sequence over (curve, run) rather than random(): a Vega
    # expression is re-evaluated on every signal change, and dots that jump when
    # a dropdown moves read as new data.
    jitter = (
        f"{v_mid} + {0.3 * x_max} * ((((datum._curve_i * 7 + datum._run)"
        f" * 0.6180339887498949) % 1) - 0.5)"
    )
    # What the annotation reports: tune within this configuration and keep the
    # best, rather than the average of choosing at random. Rounded up, so a small
    # violin still keeps one run.
    kept = (
        f"max(1, ceil(datum._n * {args.top_fraction}))" if args.top_fraction > 0
        else "datum._n"
    )
    share = "all" if args.top_fraction <= 0 else f"top {args.top_fraction:.0%}"

    def labelled(options):
        """What a dropdown shows for its options: the value 'All', which the
        filters compare against, reads "(all)"."""

        return ["(all)" if o == "All" else str(o) for o in options]

    seed_options = ["All"] + seeds
    params = [{
        "name": "p_seed",
        "value": "All",
        # 'All' is every seed at once, which is the mean and its confidence band;
        # one seed is that run's own curve.
        "bind": {
            "input": "select",
            "options": seed_options,
            "labels": labelled(seed_options),
            "name": "Seed: ",
        },
    }]
    terms = []
    for field, name in filter_fields:
        options = sorted(long_df[field].dropna().unique().tolist(), key=str)
        # No 'All' for the statistic: mixing units in one chart is meaningless.
        if field != "statistic":
            options = ["All"] + options
        params.append({
            "name": f"p_{field}",
            "value": options[0] if options else "",
            "bind": {
                "input": "select",
                "options": options,
                "labels": labelled(options),
                "name": f"{name[:1].upper()}{name[1:]}: ",
            },
        })
        terms.append(
            f"datum.{field} === p_{field}" if field == "statistic"
            else f"(p_{field} === 'All' || datum.{field} === p_{field})"
        )
    params.append({
        "name": "p_mode",
        "value": MODES[0],
        "bind": {"input": "select", "options": MODES, "name": "Show: "},
    })
    params.append({
        "name": "p_smooth",
        "value": 0,
        "bind": {
            "input": "range",
            "min": SMOOTH_SPANS[0],
            "max": SMOOTH_SPANS[-1],
            "step": SMOOTH_SPANS[1] - SMOOTH_SPANS[0],
            "name": "Smoothing: ",
        },
    })
    shared = " && ".join(terms)
    # The pooled rows arrive already averaged over every curve of their row (see
    # pooled_rows), so they carry no hyperparameter field to compare a curve
    # dropdown against. Filtered on the statistic alone: the curve terms would
    # read `undefined === <value>` and empty the whole column the moment a
    # dropdown moved off "(all)".
    pooled_shared = " && ".join(
        term for (field, _), term in zip(filter_fields, terms) if field == "statistic"
    ) or "true"
    # The modes are alternatives, not overlays: a cell draws whichever the "Show"
    # dropdown is on, and every layer of the others filters to nothing.
    agg_filter = (
        f"p_mode === '{MODES[0]}' && p_seed === 'All' && datum._kind === 'agg' "
        f"&& ({shared})"
    )
    seed_filter = (
        f"p_mode === '{MODES[0]}' && p_seed !== 'All' && datum._kind === 'seed' "
        f"&& datum.seed === p_seed && ({shared})"
    )
    # The area is taken over every seed, so it is the same bar whichever seed the
    # dropdown is on -- it answers a question about the configuration, not a run.
    # Named by MODES, not spelled out again: a mode renamed there and not here
    # filters to nothing, and an empty cell is all it says about itself.
    auc_filter = f"p_mode === '{MODES[1]}' && datum._kind === 'auc' && ({shared})"
    # The pooled column's own two: the same rows again, averaged over everything
    # the row holds -- every configuration of it and every global_sweep
    # combination under them. The average is taken before the dropdowns exist, so
    # it is over every curve of the row whichever of them is on screen.
    agg_all_filter = (
        f"p_mode === '{MODES[0]}' && datum._kind === 'agg_all'"
        f" && ({pooled_shared})"
    )
    auc_all_filter = (
        f"p_mode === '{MODES[1]}' && datum._kind === 'auc_all'"
        f" && ({pooled_shared})"
    )
    # A run is timed once, whatever it recorded, so a timing row carries no
    # statistic: it reads the curve dropdowns and not that one. Filtered on the
    # statistic it would read `undefined === <value>` and empty every cell.
    curve_shared = " && ".join(
        term for (field, _), term in zip(filter_fields, terms) if field != "statistic"
    ) or "true"
    time_filter = (
        f"p_mode === '{MODES[2]}' && datum._kind === 'time'"
        f" && ({curve_shared})"
    )
    # The pooled one is already averaged over every curve of the row, so there is
    # nothing left for a dropdown to select from.
    time_all_filter = f"p_mode === '{MODES[2]}' && datum._kind === 'time_all'"
    # One point per run, seeds included, which is what the density is over -- so
    # the seed dropdown does not narrow it either.
    # `(p_mode || '')`, not `p_mode`: a select binding is momentarily null while
    # the dropdown changes, and indexof() of null throws -- which aborts the
    # dataflow in the middle of the update and leaves the cells it had not
    # reached empty until something else redraws them.
    violin_filter = (
        f"indexof(p_mode || '', '{VIOLIN_PREFIX}') === 0"
        f" && datum._kind === 'violin' && ({shared})"
    )
    # Which of the three numbers a run carries the violin is of. Picked in the
    # page rather than at build time: three scalars per run cost little next to
    # the curve they came from, and the mode is a dropdown.
    # Named by MODES like the filters: a mode that fell through to score_auc would
    # draw a plausible violin of the wrong number. All three are spelled out and
    # the fallthrough is null, so a mode renamed there and not here leaves the
    # cell empty instead.
    pick_score = [
        {
            "calculate": (
                f"p_mode === '{MODES[3]}' ? datum.score_auc"
                f" : p_mode === '{MODES[4]}' ? datum.score_max"
                f" : p_mode === '{MODES[5]}' ? datum.score_final"
                f" : null"
            ),
            "as": "score",
        }
    ]

    # A scale's `type` is the one thing a Vega-Lite param cannot drive, so the log
    # axis is the values put into log space and the ticks labelled back out of it.
    #
    # A symlog axis: linear within `constant` of zero, logarithmic beyond it.
    # That is what a log scale means for data that goes negative -- it is
    # increasing everywhere, it puts zero at zero, and it never sends two values
    # to the same place, none of which is true of sign(y) * log(|y|), which folds
    # everything inside (-1, 1) onto the wrong side and drops zero entirely.
    #
    # Vega draws it, so the ticks and labels are its own: no transform in the
    # data, no expression to invert them, and the tooltips read the same numbers
    # as the axis. The cost is that a scale's `type` is the one thing a param
    # cannot drive, so the toggle rebuilds the spec and re-embeds it (see render
    # in RENDER_JS) rather than flipping a signal.
    #
    # The constant is where the changeover sits. Small values are the ones a log
    # axis is for, so it is the 10th percentile of the non-zero magnitudes in the
    # panel -- big enough to keep the noise around zero from filling the axis,
    # small enough to leave the decades above it spread out.
    #
    # One constant for the panel, not one per statistic: the scale is compiled
    # into the spec, and the statistic is a dropdown that moves under it. A panel
    # whose statistics differ by orders of magnitude gets a changeover that suits
    # whichever of them dominates the tenth percentile.
    def symlog_constant(values):
        """The threshold a symlog axis stays linear within, from the data itself,
        or 1 when the statistic has nothing to go on."""

        magnitudes = np.abs(values[np.isfinite(values) & (values != 0)])
        if not magnitudes.size:
            return 1.0
        return float(max(np.percentile(magnitudes, 10), np.finfo(float).tiny))

    y_scale = {"zero": False}
    if log_axis:
        # Minutes and the units of the statistic on one scale: the layers of a
        # cell share it, so the constant cannot differ per mode. The changeover
        # is taken per field and the smaller of them kept, rather than over the
        # two poured together: a curve contributes a point per step and a timing
        # row contributes one, so a pooled percentile would be the curves' alone
        # and the timing bars would sit in whatever part of the axis was left.
        # The smaller constant leaves both sets of magnitudes above it.
        constants = [
            symlog_constant(long_df[f].to_numpy(dtype=float))
            for f in ("mean", "time") if f in long_df
        ]
        if constants:
            y_scale["type"] = "symlog"
            y_scale["constant"] = min(constants)
    y_axis = {
        # Minutes in the timing mode and the statistic's own units everywhere
        # else. Nothing else on the cell says which, and the two are not
        # comparable numbers.
        #
        # A space rather than nothing in the other modes: a title that appears and
        # disappears takes its band of the gutter with it, and every cell of the
        # tab would shift sideways as the mode changed. This one is always drawn
        # and carries text only in the timing mode.
        #
        # It is also the only title on the channel -- the y encodings carry none
        # of their own -- since a field definition's `title` and an axis's set
        # the same thing and only one of them is read.
        "title": {"expr": f"p_mode === '{MODES[2]}' ? 'minutes' : ' '"},
    }
    # "Same y-lim": one y scale for every cell of every block, so its domain runs
    # from the lowest configuration to the highest.
    y_resolve = "shared" if same_y else "independent"

    def layer(filt, mark, y, tooltip=None, window=0, calc=(), **encoding):
        fields = [y] + ([encoding["y2"]["field"]] if "y2" in encoding else [])
        transforms = [{"filter": filt}]
        if window:
            # The moving average, computed in the page over the points either side
            # rather than carried in the data. `as` writes back over the field, so
            # nothing downstream has to know whether it was smoothed. Grouping by
            # curve and seed keeps one run's window off its neighbours.
            transforms.append({
                "window": [
                    {"op": "mean", "field": f, "as": f} for f in fields
                ],
                "frame": [-window, window],
                "sort": [{"field": "x"}],
                "groupby": ["_curve", "seed"],
            })
        calcs = list(calc)
        enc = {
            # Bars and violins are placed in training steps too, but at positions
            # that mean nothing as a number, so the axis reads only for the
            # curves. Blanked rather than dropped: an axis appearing and
            # disappearing would resize the cell under the pointer.
            "x": {
                "field": "x",
                "type": "quantitative",
                "title": "Training steps",
                "axis": {
                    "labelExpr": f"p_mode === '{MODES[0]}' ? datum.label : ''",
                },
            },
            "y": {
                "field": y,
                "type": "quantitative",
                "scale": y_scale,
                "axis": y_axis,
            },
            "color": {
                "field": "_curve",
                "type": "nominal",
                "title": None,
                "scale": {"domain": curve_domain},
                "legend": {"values": curve_labels},
            },
            **encoding,
        }
        if tooltip:
            enc["tooltip"] = tooltip
        return {"transform": [*transforms, *calcs], "mark": mark, "encoding": enc}

    def tooltip(*fields):
        """A tooltip list from (field, type, extra keys) triples."""

        return [{"field": f, "type": t, **rest} for f, t, rest in fields]

    def curve_layers(index, span):
        """The band, the mean and the per-seed lines at one smoothing window.

        Only the window the slider is on has any data; the rest filter to nothing.
        The zoom bindings ride on the unsmoothed mean line and only there: every
        layer they sit on instantiates them, and two instances collide on the
        signal names Vega gives them. Scales are shared within a block, so a block
        moves as a whole and blocks move independently of each other."""

        picked = f"({{}}) && p_smooth === {span}"
        line = layer(
            picked.format(agg_filter),
            {"type": "line", "clip": True},
            "mean",
            tooltip(
                ("_curve", "nominal", {"title": "curve"}),
                ("x", "quantitative", {"title": "step"}),
                ("mean", "quantitative", {"format": ".3f"}),
                # How many seeds were still running AT THIS POINT, not how many
                # the configuration started with: where that is 1 the band has no
                # width and is drawn from a single run.
                ("alive", "quantitative", {"title": "seeds alive"}),
            ),
            window=span,
        )
        if span == SMOOTH_SPANS[0]:
            line = {
                **line,
                "params": [
                    {
                        "name": f"zoom_x{index}",
                        "select": {
                            "type": "interval",
                            "encodings": ["x"],
                            "zoom": "wheel![!event.shiftKey]",
                            "translate": (
                                "[mousedown[!event.shiftKey], window:mouseup] "
                                "> window:mousemove!"
                            ),
                        },
                        "bind": "scales",
                    },
                    {
                        # One name across the blocks when they share the y scale:
                        # the scale follows a single selection, and a block
                        # zooming under another name would move nothing.
                        "name": "zoom_y" if same_y else f"zoom_y{index}",
                        "select": {
                            "type": "interval",
                            "encodings": ["y"],
                            "zoom": "wheel![event.shiftKey]",
                            "translate": (
                                "[mousedown[event.shiftKey], window:mouseup] "
                                "> window:mousemove!"
                            ),
                        },
                        "bind": "scales",
                    },
                ],
            }
        return [
            layer(
                picked.format(agg_filter),
                {"type": "area", "opacity": 0.2, "clip": True},
                "ci_lo",
                y2={"field": "ci_hi"},
                window=span,
            ),
            line,
            # detail: keep each seed's path separate instead of letting
            # Vega-Lite join runs that share a color.
            layer(
                picked.format(seed_filter),
                {"type": "line", "clip": True, "opacity": 0.7, "strokeWidth": 1},
                "value",
                tooltip(
                    ("_curve", "nominal", {"title": "curve"}),
                    ("seed", "nominal", {"title": "seed"}),
                    ("x", "quantitative", {"title": "step"}),
                    ("value", "quantitative", {"format": ".3f"}),
                ),
                window=span,
                detail={"field": "seed", "type": "nominal"},
            ),
        ]

    def block_spec(index, label):
        return {
            "title": label or None,
            # Position, not title: an entry that pins nothing has no title, and two
            # of them would otherwise select each other's rows.
            #
            # Everything else the dropdowns decide is filtered inside the marks,
            # so a cell empties rather than leaving the grid. The pooled column
            # stays in every mode for the same reason: taking it away changes the
            # facet's own columns, and the only way to do that is to rebuild the
            # view -- three seconds against the one it costs to leave an empty
            # cell at the end of each row.
            "transform": [{"filter": f"datum._block === {index}"}],
            "facet": {
                "row": {
                    "field": "_row",
                    "type": "nominal",
                    "title": None,
                    "header": {"labelAngle": 0, "labelAlign": "left"},
                },
                "column": {
                    "field": "_col",
                    "type": "nominal",
                    "title": None,
                    "sort": col_orders.get(index, "ascending"),
                },
            },
            "spec": {
                "width": CELL_WIDTH,
                "height": CELL_HEIGHT,
                "layer": [
                    # One set of curve layers per smoothing window. A window
                    # transform's frame is fixed when the spec is written, so the
                    # slider chooses between layers rather than resizing one --
                    # every layer but the selected one filters to nothing, which
                    # costs spec text rather than a copy of the data per window.
                    *[
                        spec
                        for span in SMOOTH_SPANS
                        for spec in curve_layers(index, span)
                    ],
                    # The bars the --auc figure draws, on the same grid and the
                    # same scales: one bar per curve in its own colour with a black
                    # edge, and a black rule over the confidence interval. They
                    # stand at `bar_at`, i.e. in training steps, and carry their
                    # width as x/x2 rather than a pixel size, so a zoomed or
                    # resized view spaces them the way it spaces everything else.
                    layer(
                        auc_filter,
                        {
                            "type": "bar",
                            "stroke": "black",
                            "strokeWidth": 0.6,
                            "clip": True,
                        },
                        "auc",
                        tooltip(
                            ("_curve", "nominal", {"title": "curve"}),
                            (
                                "auc",
                                "quantitative",
                                {"title": "area", "format": ".3f"},
                            ),
                            ("n_seeds", "quantitative", {"title": "seeds"}),
                        ),
                        calc=[
                            {"calculate": bar_at(-1), "as": "_bar_lo"},
                            {"calculate": bar_at(1), "as": "_bar_hi"},
                        ],
                        x={
                            "field": "_bar_lo",
                            "type": "quantitative",
                            "title": None,
                        },
                        x2={"field": "_bar_hi"},
                        # Where the bar starts (see the floor in the main loop).
                        y2={"field": "auc_base"},
                    ),
                    layer(
                        auc_filter,
                        {"type": "rule", "clip": True, "strokeWidth": 0.8},
                        "auc_lo",
                        y2={"field": "auc_hi"},
                        calc=[{"calculate": bar_at(0), "as": "_bar_mid"}],
                        x={
                            "field": "_bar_mid",
                            "type": "quantitative",
                            "title": None,
                        },
                        color={"value": "black"},
                    ),
                    # The wall-clock bars, in the same slots and drawn the same
                    # way: the height is minutes rather than the units of the y
                    # axis, so the only thing that says which is which is the
                    # "Show" dropdown they are chosen from.
                    layer(
                        time_filter,
                        {
                            "type": "bar",
                            "stroke": "black",
                            "strokeWidth": 0.6,
                            "clip": True,
                        },
                        "time",
                        tooltip(
                            ("_curve", "nominal", {"title": "curve"}),
                            (
                                "time",
                                "quantitative",
                                {"title": "minutes", "format": ".1f"},
                            ),
                            ("n_times", "quantitative", {"title": "runs timed"}),
                        ),
                        calc=[
                            {"calculate": bar_at(-1), "as": "_bar_lo"},
                            {"calculate": bar_at(1), "as": "_bar_hi"},
                        ],
                        x={
                            "field": "_bar_lo",
                            "type": "quantitative",
                            "title": None,
                        },
                        x2={"field": "_bar_hi"},
                        y2={"field": "time_base"},
                    ),
                    layer(
                        time_filter,
                        {"type": "rule", "clip": True, "strokeWidth": 0.8},
                        "time_lo",
                        y2={"field": "time_hi"},
                        calc=[{"calculate": bar_at(0), "as": "_bar_mid"}],
                        x={
                            "field": "_bar_mid",
                            "type": "quantitative",
                            "title": None,
                        },
                        color={"value": "black"},
                    ),
                    # --- The pooled column ------------------------------------
                    # One curve and two bars, over everything the row holds. All
                    # three arrive averaged (see pooled_rows), so the layers only
                    # draw them -- and, being one series rather than a copy of
                    # every curve, they do not follow the curve dropdowns.
                    {
                        "transform": [
                            {"filter": agg_all_filter},
                            # Smoothed like every other curve. The slider moves
                            # one signal, and a layer that does not read it is a
                            # curve that stays jagged while its neighbours settle.
                            # One window here rather than a layer per span: this
                            # is a single averaged line, so the frame can be a
                            # signal instead of being written out per width.
                            # An empty groupby, not a missing one: the facet
                            # pushes its own fields into it, and Vega-Lite reads
                            # the key to do it, so leaving it out is a crash.
                            {
                                "window": [
                                    {"op": "mean", "field": f, "as": f}
                                    for f in ("mean", "ci_lo", "ci_hi")
                                ],
                                "frame": [
                                    {"signal": "-p_smooth"},
                                    {"signal": "p_smooth"},
                                ],
                                "sort": [{"field": "x"}],
                                "groupby": [],
                            },
                        ],
                        "layer": [
                            {
                                "mark": {
                                    "type": "area",
                                    "opacity": 0.2,
                                    "clip": True,
                                    "color": POOLED_COLOR,
                                },
                                "encoding": {
                                    "x": {
                                        "field": "x",
                                        "type": "quantitative",
                                        "title": None,
                                    },
                                    "y": {
                                        "field": "ci_lo",
                                        "type": "quantitative",
                                        "scale": y_scale,
                                        "axis": y_axis,
                                    },
                                    "y2": {"field": "ci_hi"},
                                },
                            },
                            {
                                "mark": {
                                    "type": "line",
                                    "clip": True,
                                    "color": POOLED_COLOR,
                                },
                                "encoding": {
                                    "x": {
                                        "field": "x",
                                        "type": "quantitative",
                                        "title": None,
                                    },
                                    "y": {
                                        "field": "mean",
                                        "type": "quantitative",
                                        "scale": y_scale,
                                        "axis": y_axis,
                                    },
                                    "tooltip": tooltip(
                                        ("x", "quantitative", {"title": "step"}),
                                        (
                                            "mean",
                                            "quantitative",
                                            {"format": ".3f"},
                                        ),
                                        # How many of the row's curves reach
                                        # this point: where it drops, the line
                                        # steps because a curve ended, not
                                        # because the runs did anything.
                                        (
                                            "n_curves",
                                            "quantitative",
                                            {"title": "curves averaged"},
                                        ),
                                    ),
                                },
                            },
                        ],
                    },
                    {
                        "transform": [
                            {"filter": auc_all_filter},
                            # One bar, in the middle of the span the others share.
                            {
                                "calculate": f"{v_mid} - {0.4 * bar_step}",
                                "as": "_bar_lo",
                            },
                            {
                                "calculate": f"{v_mid} + {0.4 * bar_step}",
                                "as": "_bar_hi",
                            },
                        ],
                        "layer": [
                            {
                                "mark": {
                                    "type": "bar",
                                    "stroke": "black",
                                    "strokeWidth": 0.6,
                                    "clip": True,
                                    "color": POOLED_COLOR,
                                },
                                "encoding": {
                                    "x": {
                                        "field": "_bar_lo",
                                        "type": "quantitative",
                                        "title": None,
                                    },
                                    "x2": {"field": "_bar_hi"},
                                    "y": {
                                        "field": "auc",
                                        "type": "quantitative",
                                        "scale": y_scale,
                                        "axis": y_axis,
                                    },
                                    "y2": {"field": "auc_base"},
                                    "tooltip": tooltip(
                                        (
                                            "auc",
                                            "quantitative",
                                            {"title": "area", "format": ".3f"},
                                        ),
                                    ),
                                },
                            },
                            {
                                "mark": {
                                    "type": "rule",
                                    "clip": True,
                                    "strokeWidth": 0.8,
                                    "color": "black",
                                },
                                "encoding": {
                                    "x": {"datum": v_mid},
                                    "y": {
                                        "field": "auc_lo",
                                        "type": "quantitative",
                                        "scale": y_scale,
                                        "axis": y_axis,
                                    },
                                    "y2": {"field": "auc_hi"},
                                },
                            },
                        ],
                    },
                    # The row's own timing bar, averaged over every curve it
                    # holds, in the middle of the same span the pooled area bar
                    # stands in.
                    {
                        "transform": [
                            {"filter": time_all_filter},
                            {
                                "calculate": f"{v_mid} - {0.4 * bar_step}",
                                "as": "_bar_lo",
                            },
                            {
                                "calculate": f"{v_mid} + {0.4 * bar_step}",
                                "as": "_bar_hi",
                            },
                        ],
                        "layer": [
                            {
                                "mark": {
                                    "type": "bar",
                                    "stroke": "black",
                                    "strokeWidth": 0.6,
                                    "clip": True,
                                    "color": POOLED_COLOR,
                                },
                                "encoding": {
                                    "x": {
                                        "field": "_bar_lo",
                                        "type": "quantitative",
                                        "title": None,
                                    },
                                    "x2": {"field": "_bar_hi"},
                                    "y": {
                                        "field": "time",
                                        "type": "quantitative",
                                        "scale": y_scale,
                                        "axis": y_axis,
                                    },
                                    "y2": {"field": "time_base"},
                                    "tooltip": tooltip(
                                        (
                                            "time",
                                            "quantitative",
                                            {"title": "minutes", "format": ".1f"},
                                        ),
                                    ),
                                },
                            },
                            {
                                "mark": {
                                    "type": "rule",
                                    "clip": True,
                                    "strokeWidth": 0.8,
                                    "color": "black",
                                },
                                "encoding": {
                                    "x": {"datum": v_mid},
                                    "y": {
                                        "field": "time_lo",
                                        "type": "quantitative",
                                        "scale": y_scale,
                                        "axis": y_axis,
                                    },
                                    "y2": {"field": "time_hi"},
                                },
                            },
                        ],
                    },
                    # The two ends of the training run, drawn as nothing. They are
                    # here for the x scale alone (see `anchors`), in every mode
                    # and past every dropdown, so the domain is the same span
                    # whichever mode is showing and whichever curves a block has.
                    {
                        "transform": [{"filter": "datum._anchor === 1"}],
                        "mark": {"type": "point", "opacity": 0},
                        "encoding": {
                            "x": {
                                "field": "x",
                                "type": "quantitative",
                                "title": None,
                            },
                        },
                    },
                    # One violin per cell: the density over every run the cell pools,
                    # which is the global_sweep combinations and the seeds under
                    # them. The bars beside it are one number per curve; this is
                    # the spread those numbers came from.
                    #
                    # The density is normalised by its own peak and laid out as
                    # x/x2 about the middle of the span, so the shape is symmetric
                    # and every violin is the same width -- Vega-Lite's stacking
                    # would decide both from the data instead.
                    {
                        "transform": [
                            {"filter": violin_filter},
                            *pick_score,
                            {"calculate": "datum.score", "as": "_score"},
                            # An empty groupby, not a missing one: the facet
                            # pushes its own fields into it, which is what makes
                            # this one density per cell -- and Vega-Lite reads
                            # the key to do it, so leaving it out is a crash.
                            {
                                "density": "_score",
                                "steps": 64,
                                "as": ["_v", "_d"],
                                "groupby": [],
                            },
                            {
                                "joinaggregate": [
                                    {"op": "max", "field": "_d", "as": "_dmax"},
                                ],
                                "groupby": [],
                            },
                            {
                                "calculate":
                                    f"{v_mid} - datum._d / datum._dmax * {v_half}",
                                "as": "_v_lo",
                            },
                            {
                                "calculate":
                                    f"{v_mid} + datum._d / datum._dmax * {v_half}",
                                "as": "_v_hi",
                            },
                        ],
                        "mark": {
                            "type": "area",
                            "orient": "horizontal",
                            "opacity": 0.55,
                            "stroke": "#333",
                            "strokeWidth": 0.6,
                            "clip": True,
                        },
                        "encoding": {
                            "y": {
                                "field": "_v",
                                "type": "quantitative",
                                "scale": y_scale,
                                "axis": y_axis,
                            },
                            "x": {
                                "field": "_v_lo",
                                "type": "quantitative",
                                "title": None,
                            },
                            "x2": {"field": "_v_hi"},
                            "color": {"value": "#7f7f7f"},
                        },
                    },
                    # The quartiles down the middle of it and the median across them.
                    {
                        "transform": [
                            {"filter": violin_filter},
                            *pick_score,
                            {"calculate": "datum.score", "as": "_score"},
                            {
                                "aggregate": [
                                    {"op": "q1", "field": "_score", "as": "_q1"},
                                    {"op": "q3", "field": "_score", "as": "_q3"},
                                    {
                                        "op": "median",
                                        "field": "_score",
                                        "as": "_med",
                                    },
                                ],
                                "groupby": [],
                            },
                        ],
                        "layer": [
                            {
                                "mark": {
                                    "type": "rule",
                                    "strokeWidth": 7,
                                    "color": "#4d4d4d",
                                    "clip": True,
                                },
                                "encoding": {
                                    "y": {
                                        "field": "_q1",
                                        "type": "quantitative",
                                        "scale": y_scale,
                                        "axis": y_axis,
                                    },
                                    "y2": {"field": "_q3"},
                                    "x": {"datum": v_mid},
                                },
                            },
                            {
                                "mark": {
                                    "type": "rule",
                                    "strokeWidth": 1.5,
                                    "color": "white",
                                    "clip": True,
                                },
                                "encoding": {
                                    "y": {
                                        "field": "_med",
                                        "type": "quantitative",
                                        "scale": y_scale,
                                        "axis": y_axis,
                                    },
                                    "x": {"datum": v_mid - 0.035 * x_max},
                                    "x2": {"datum": v_mid + 0.035 * x_max},
                                },
                            },
                        ],
                    },
                    # The runs themselves, at the height each scored: a density
                    # drawn over six points looks like a distribution it has not
                    # earned. Which curve a dot belongs to is in its tooltip --
                    # the dots are one colour.
                    layer(
                        violin_filter,
                        {
                            "type": "point",
                            "filled": True,
                            "size": 14,
                            "opacity": 0.7,
                            "clip": True,
                        },
                        "score",
                        tooltip(
                            ("_curve", "nominal", {"title": "curve"}),
                            (
                                "score",
                                "quantitative",
                                {"title": "score", "format": ".3f"},
                            ),
                        ),
                        calc=[*pick_score, {"calculate": jitter, "as": "_dot_x"}],
                        x={
                            "field": "_dot_x",
                            "type": "quantitative",
                            "title": None,
                        },
                        color={"value": "#262626"},
                    ),
                    # The mean over the best of them, at the height it describes
                    # and against a line across the cell -- a mean means nothing
                    # without the axis. Pinned to the left edge in pixels: the
                    # text is wider than the room beside a violin, and a cell has
                    # no margin to spill into.
                    {
                        "transform": [
                            {"filter": violin_filter},
                            *pick_score,
                            {"calculate": "datum.score", "as": "_score"},
                            {
                                "window": [{"op": "rank", "as": "_rank"}],
                                "sort": [
                                    {"field": "_score", "order": "descending"},
                                ],
                                "groupby": [],
                            },
                            {
                                "joinaggregate": [
                                    {"op": "count", "as": "_n"},
                                ],
                                "groupby": [],
                            },
                            {"filter": f"datum._rank <= {kept}"},
                            {
                                "aggregate": [
                                    # Two means over the same runs: one where the
                                    # line goes, one to read out. They differ only
                                    # when "Log scale" is on, and the number quoted
                                    # is the one in the units of the statistic.
                                    {
                                        "op": "mean",
                                        "field": "_score",
                                        "as": "_top",
                                    },
                                    {
                                        "op": "mean",
                                        "field": "score",
                                        "as": "_top_raw",
                                    },
                                    {"op": "count", "as": "_top_n"},
                                ],
                                "groupby": [],
                            },
                        ],
                        "layer": [
                            {
                                "mark": {
                                    "type": "rule",
                                    "strokeDash": [3, 2],
                                    "strokeWidth": 0.8,
                                    "color": "#262626",
                                    "opacity": 0.8,
                                    "clip": True,
                                },
                                "encoding": {
                                    "y": {
                                        "field": "_top",
                                        "type": "quantitative",
                                        "scale": y_scale,
                                        "axis": y_axis,
                                    },
                                },
                            },
                            # Two marks rather than one two-line one: a text
                            # channel takes a field, and a field holding two
                            # lines would be read as one comma-joined string.
                            *[
                                {
                                    "transform": [
                                        {"calculate": expr, "as": "_label"},
                                    ],
                                    "mark": {
                                        "type": "text",
                                        "align": "left",
                                        "baseline": "top",
                                        "dx": 3,
                                        "dy": dy,
                                        "fontSize": 8,
                                        "color": "#262626",
                                        "clip": True,
                                    },
                                    "encoding": {
                                        "y": {
                                            "field": "_top",
                                            "type": "quantitative",
                                            "scale": y_scale,
                                            "axis": y_axis,
                                        },
                                        "x": {"value": 0},
                                        "text": {
                                            "field": "_label",
                                            "type": "nominal",
                                        },
                                    },
                                }
                                for expr, dy in (
                                    (f"'{share} of ' + datum._top_n", 2),
                                    ("'mean ' + format(datum._top_raw, '.4g')", 12),
                                )
                            ],
                        ],
                    },
                ],
            },
            "resolve": {"scale": {"y": y_resolve, "color": "shared"}},
        }

    # A row is an aggregate, a seed's curve, an area, or a timing, and each
    # carries only its own fields -- so every row would be written with a NaN in
    # the fields of the others, about 60% of the page here, and an object no JSON
    # parser will read. An entry that is not there means the same thing to Vega
    # as one that is NaN: invalid, and filtered out of the mark. NaN is the only
    # value that fails `v == v`.
    #
    # Built only for the spec that keeps it. The log-axis and same-y variants
    # differ in a scale and are written without their data (the page hands them
    # the base spec's), so doing this for them would walk the largest object in
    # the program to throw the answer away.
    values = [] if log_axis or same_y else [
        {k: v for k, v in record.items() if v == v}
        for record in long_df.to_dict(orient="records")
    ] + anchors

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": values},
        "params": params,
        "vconcat": [block_spec(i, label) for i, label in block_labels],
        "resolve": {"scale": {"y": y_resolve, "color": "shared"}},
        "config": {
            "legend": {"orient": "top", "labelLimit": 0, "symbolOpacity": 1},
            # Floor for the y-axis gutter: independent scales size it per cell,
            # so switching statistic would otherwise resize every block.
            "axisY": {"minExtent": 30},
        },
    }


gzip_path = os.path.join(args.folder, args.gzip)
df = pd.read_parquet(gzip_path)
vprint(f"\nLoaded gzip from: {gzip_path}")
if all(isinstance(c, str) and c.startswith("(") for c in df.columns):
    df.columns = pd.MultiIndex.from_tuples([literal_eval(c) for c in df.columns])

global_sweep, sweep = load_sweep(args.sweep)
vprint(
    f"Sweep {args.sweep}: {len(sweep)} block(s), "
    f"curves from {list(global_sweep)}"
)

# Labels are decided over the whole set of hyperparameters at once: a name is the
# last dotted segment where that is unique and longer where it is not, so every
# key that will ever be labelled has to be here -- the data's columns and the
# sweep's keys, which need not be columns.
LABELS = hyperparameter_to_label(
    sorted(
        {c[1] for c in df.columns if c[0] == "hyperparameter"}
        | set(global_sweep)
        | {k for entry in sweep for k in entry}
    )
)

swept = swept_keys(global_sweep, sweep)

env_configs = load_config_group(args.environments) if args.environments else {}
df[ENV] = (
    assign_configs(df, env_configs, swept) if env_configs
    else df[ENV_ID].map(value_str)
)

algo_configs = load_config_group(args.algorithms) if args.algorithms else {}
if algo_configs:
    df[ALGO] = assign_configs(
        df, algo_configs, swept, overridden_by=(df[ENV], env_configs)
    )

for pairs, flag, keep in (
    (args.include, "--include", True),
    (args.exclude, "--exclude", False),
):
    try:
        df, filter_messages = apply_filters(df, pairs, flag, keep)
    except ValueError as e:
        parser.error(str(e))
    for message in filter_messages:
        vprint(message)

missing_runs = {}

# Only the environments the sweep declares. The gzip holds whatever ran in the
# folder, and a sweep that names two of them is a figure about those two.
#
# An entry that declares no environment of its own and finds none in
# global_sweep constrains none of them, so there is nothing to filter by and the
# data decides as before -- the same reading `declared_rows` gives an entry with
# no `environment` key. `declared_envs` stays None there, so what follows can
# tell "declared nothing" from "declared these".
#
# Matched on the stem, as the sweep names an environment by its config file
# while an overridden run is labelled `<stem> (key=value)`.
declared_envs = None
# Nothing left to filter, and nothing the sweep did: --include/--exclude emptied
# the frame, and the exit below would put the sweep's name to their doing.
if not df.empty:
    declared_envs = declared_environments(sweep, global_sweep)
    if declared_envs is None:
        vprint(
            "\nAt least one sweep entry declares no environment, so every "
            "environment in the data is plotted."
        )
    df = keep_declared_environments(df, declared_envs, "plotted")

# Only the runs the sweep composes.
df = keep_composed_runs(df, sweep, global_sweep, "plotted", vprint)

# The y limits a labels file can carry are left out on purpose: an explicit domain
# overrides a scale-bound zoom, and the y axis would stop moving.

envs_present = sorted(df[ENV].unique(), key=str)
vprint(f"\n{len(envs_present)} environment(s) found: {envs_present}")

# The grid. `curves=True`: the keys `global_sweep` declares become the curve
# axes, a line per combination drawn over each other in one cell, which is the
# dimension interactive_heatmaps.py spends on a dropdown instead.
layout = SweepLayout(df, sweep, global_sweep, LABELS, curves=True, vprint=vprint)

# The same root every plotting script writes under, so one data directory holds
# all of their output side by side.
out_dir = os.path.join(args.folder, "plots")
os.makedirs(out_dir, exist_ok=True)
# Browser tab name: the data directory, then the sweep it was laid out from.
sweep_name = os.path.splitext(os.path.basename(args.sweep))[0]
folder_name = os.path.basename(os.path.normpath(args.folder))
page_title = f"{folder_name} ({sweep_name}) - curves"
# File name: the sweep the page was laid out from, and what it draws.
out_path = os.path.join(out_dir, f"{sweep_name}_curves.html")

total_cells = 0
panels = []
# (environment, rows, filter fields, blocks), drawn once every environment has
# been read: the curves' colours are decided over all of them (curve_domain).
env_tabs = []
param_rows = []
# One row per (configuration, environment) bar, kept to be averaged over the
# environments once they have all been drawn.
avg_auc_rows = []
# {sweep entry index: its label}, so the averaged panel names a block by the
# entry it came from rather than by where it landed in one environment's list.
avg_block_labels = {}

for env in envs_present:
    # An overridden run is labelled `<stem> (key=value)`, and the sweep names an
    # environment by its config file, so the stem is what a declared
    # configuration is matched against.
    env_stem = split_config_label(str(env))[0]
    # The tab is named as the data writes it, overrides and all: the page has no
    # plot config to read display names from, and a tab has to be findable by the
    # name the runs were recorded under.
    env_name = str(env)
    df_env = df[df[ENV] == env]
    vprint(f"\n{'=' * 60}\nEnvironment: {env_name}\n{'=' * 60}")

    # Not configured: the fullest configuration actually seen for this environment.
    has_seeds = SEED in df_env.columns and df_env[SEED].notna().any()
    expected_seeds = int(df_env[SEED].dropna().max()) + 1 if has_seeds else None

    blocks, curve_axes = layout.blocks(df_env)
    if not blocks:
        vprint(f"Nothing to plot for {env_name}, skipping.")
        continue
    curve_params = [c for c, _ in curve_axes]
    curve_combos = list(itertools.product(*[v for _, v in curve_axes]))
    for i, block in enumerate(blocks):
        vprint(
            f"Block {block.label or i + 1}: "
            f"{len(block.row_combos)}x{len(block.col_combos)} cell(s), "
            f"columns {[c[1] for c in block.col_params]}, "
            f"rows {[c[1] for c in block.row_params]}"
        )
    vprint(f"Curves: {[c[1] for c in curve_params]} -> {len(curve_combos)} per cell")

    if args.verbose:
        recap(env_stem, df_env)

    if args.prepare_missing_runs and expected_seeds is None:
        print(
            f"WARNING: {env_name} records no seed, so its missing runs were not "
            f"checked."
        )

    if args.prepare_missing_runs and expected_seeds is not None:
        declared = incomplete = 0
        for hp, rows in declared_rows(df_env, env_stem):
            declared += 1
            incomplete += record_missing(
                missing_runs,
                launch_overrides(
                    {"environment": env, "algorithm": algorithm_of(rows)},
                    hp,
                ),
                seeds_of(rows),
                expected_seeds,
            )
        vprint(
            f"{declared} configuration(s) declared by the sweep, "
            f"{incomplete} with missing seeds"
        )

    # `time` is not among them: it has a mode of its own, and offering it here
    # would draw a run's duration a second time, in seconds, as a curve, as an
    # AUC bar and as a violin on the parameters tab.
    stats_found = sorted(
        {c[1] for c in df_env.columns if c[0] == "statistic" and c[1] != TIME[1]}
    )
    if args.stats:
        # `time` is ignored rather than reported absent -- it is drawn by its own
        # mode rather than selected here -- but only where the data records it:
        # a name that is nowhere in the gzip should still be reported, whichever
        # name it is.
        records_time = TIME in df_env.columns
        absent = [
            s for s in args.stats
            if s not in stats_found and not (records_time and s == TIME[1])
        ]
        if absent:
            print(f"WARNING: --stats not found in the data for {env_name}: {absent}")
        stats = [s for s in args.stats if s in stats_found]
    else:
        stats = stats_found
    if not stats:
        # Loud when --stats asked for something: the environment leaves no tab,
        # and the reason is in what was asked for rather than in the data. The
        # timing bars go with it -- they are a mode of a tab that is not drawn.
        message = (
            f"No statistic to plot for {env_name}, skipping it and its "
            f"timing bars."
        )
        if args.stats:
            print(f"WARNING: {message}")
        else:
            vprint(message)
        continue
    vprint(f"Statistics: {stats}")

    # Only curve hyperparameters get a dropdown: row/column ones already have
    # their own cell, so filtering one would just blank cells out.
    fields = {"statistic": "statistic"}
    for c in curve_params:
        fields.setdefault(vega_field_name(LABELS, c[1]), LABELS[c[1]])

    long_rows = []
    # The curve is the outer loop and the statistic the inner one, the other way
    # round from the grid: a run's time belongs to the curve rather than to
    # anything it recorded, so it is written once per curve and the statistics
    # are walked underneath it.
    for block in blocks:
        for _, row_vals, _, col_vals, cell in iter_cells(block):
            for curve_i, curve_vals, curve in iter_curves(cell):
                base = {
                    # The sweep entry, not the block's place in the list: see
                    # where the blocks are built.
                    "_block": block.index,
                    "_row": label_of(LABELS, block.row_params, row_vals) or " ",
                    "_col": label_of(LABELS, block.col_params, col_vals) or " ",
                    "_curve": label_of(LABELS, curve_params, curve_vals) or " ",
                    # Where the bar sits: the curve's own index, so a cell
                    # missing a curve leaves a gap instead of shifting the
                    # rest along, as the AUC bars are drawn.
                    "_curve_i": curve_i,
                }
                # Only the axes: the last panel groups runs by a hyperparameter
                # that varies under them, so a block holding one value of it --
                # or none at all -- is not part of that comparison and is left
                # out of the field entirely.
                varied = {}
                for params, values in axes_of(block, row_vals, col_vals, curve_vals):
                    base.update({
                        vega_field_name(LABELS, p[1]): value_str(v)
                        for p, v in zip(params, values)
                    })
                    varied.update({
                        p[1]: value_str(v) for p, v in zip(params, values)
                    })
                timing = time_of(curve)
                if timing is not None:
                    long_rows.append({**base, **timing})
                for stat in stats:
                    decoded, seeds = rows_of(curve, stat)
                    if not decoded:
                        continue
                    stat_base = {
                        **base,
                        "statistic": stat,
                        # The seeds that reached the figure, not the rows the
                        # frame holds: series_of drops a duplicate seed and one
                        # whose curve has no finite point, and counting those
                        # would report more seeds than the band was drawn from.
                        "n_seeds": len(seeds),
                    }
                    long_rows += [{**stat_base, **r} for r in decoded]
                    param_rows += [
                        {
                            "_env": env_name,
                            "statistic": stat,
                            "_run": r["_run"],
                            "score_auc": r["score_auc"],
                            "score_max": r["score_max"],
                            "score_final": r["score_final"],
                            **varied,
                        }
                        for r in decoded if r["_kind"] == "violin"
                    ]

    # A time that did not read as a finite number leaves a bar silently absent,
    # so the count is reported and reset for the next environment.
    if unreadable_times:
        print(
            f"WARNING: {unreadable_times} recorded time(s) in {env_name} did not "
            f"read as a finite number, so they are not in the timing bars."
        )
        unreadable_times = 0

    # Nothing on the page says a curve's seeds were scored over different spans,
    # so the count is reported and reset the same way.
    if ragged_curves:
        print(
            f"WARNING: {len(ragged_curves)} curve(s) in {env_name} have seeds "
            f"whose curves end at different points, so each was scored over its "
            f"own span rather than over the same one."
        )
        ragged_curves.clear()

    # A timing row alone is not a panel: it has no x of its own, and the grid is
    # laid out on the span the curves cover.
    if not any(r["_kind"] != "time" for r in long_rows):
        vprint(f"No decodable curve for {env_name}, skipping.")
        continue
    total_cells += sum(len(b.col_combos) * len(b.row_combos) for b in blocks)

    # The pooled column, averaged here rather than in the page. Copying every
    # aggregate row into that column so the browser could average them cost about
    # a quarter of the file size; this is one series per (block, row, statistic),
    # and it carries only what the pooled layers draw. The price is that it cannot
    # follow the curve dropdowns: the average is taken before they exist, so it is
    # over every curve of the row whichever of them is on screen.
    long_rows += pooled_rows(long_rows)

    long_df = pd.DataFrame(long_rows)

    # The bars are drawn from a floor rather than from zero. A negative area is a
    # real result but a bar hanging down from zero beside one standing up from it
    # compares badly. One floor per statistic, shared by every cell of every block.
    areas = long_df["_kind"].isin(("auc", "auc_all"))
    if areas.any():
        lowest = long_df[areas].groupby("statistic")["auc_lo"].transform("min")
        long_df.loc[areas, "auc_base"] = lowest - 0.1 * lowest.abs()
    # The timing bars get the same floor -- runs differ by a few percent
    # between configurations far more often than by a factor, and bars from zero
    # would draw that as no difference at all. One floor for the panel rather
    # than one per statistic: a timing row belongs to none of them.
    # Never below zero, unlike the area floor: an interval wider than the mean
    # puts `time_lo` under it, and no run took negative time -- a floor down
    # there would stand every bar on the same distant line and draw the
    # differences between them as nothing.
    timed = long_df["_kind"].isin(("time", "time_all"))
    if timed.any():
        lowest = max(0.0, float(long_df.loc[timed, "time_lo"].min()))
        long_df.loc[timed, "time_base"] = lowest - 0.1 * lowest
    avg_auc_fields = (
        "statistic", "_block", "_row", "_col",
        "_curve", "_curve_i", "auc", "auc_lo", "auc_hi",
    )
    avg_auc_rows += [
        {"_env": env_name, **{k: r[k] for k in avg_auc_fields}}
        for r in long_rows if r.get("_kind") == "auc"
    ]
    # The averaged panel stacks the environments on each other by sweep entry, so
    # it needs a name for every entry ANY environment drew -- not only the ones
    # the first environment happened to have runs for.
    for block in blocks:
        avg_block_labels.setdefault(block.index, block.label)

    # One panel per environment, written as one file after the loop: the
    # environments are what you flip between, so they belong in one page.
    env_tabs.append((
        env_name,
        long_df,
        list(fields.items()),
        [(b.index, b.label) for b in blocks],
    ))
    vprint(
        f"  Panel: {env_name}  ({len(long_df)} rows, {len(fields)} filters, "
        f"{len(blocks)} block(s))"
    )

# Environments the sweep declares that left no run at all. The loop above walks
# what the data holds, so it never reaches them.
if declared_envs is not None:
    absent_envs = sorted(
        declared_envs - {split_config_label(str(e))[0] for e in envs_present}
    )
    if absent_envs:
        print(
            f"WARNING: {len(absent_envs)} environment(s) the sweep declares have "
            f"no run in the data: {absent_envs}"
        )
    if absent_envs and args.prepare_missing_runs:
        # No rows, so no seed count of its own. The highest seed anywhere in the
        # data is used instead, and reported, since the environment recorded
        # nothing to derive it from.
        has_any_seed = SEED in df.columns and df[SEED].notna().any()
        if not has_any_seed:
            print(
                f"WARNING: the data records no seed, so the missing runs of "
                f"{absent_envs} were not written."
            )
        else:
            everywhere = int(df[SEED].dropna().max()) + 1
            print(
                f"Their missing runs are written against {everywhere} seed(s), "
                f"the highest seed found anywhere in the data."
            )
            # An empty slice, columns and all: `declared_rows` walks the sweep
            # and selects rows for each combination, and for these there are
            # none to find -- every seed of every configuration is missing.
            empty = df.iloc[0:0]
            for env_stem in absent_envs:
                for hp, _ in declared_rows(empty, env_stem):
                    record_missing(
                        missing_runs,
                        launch_overrides(
                            {"environment": env_stem, "algorithm": None}, hp
                        ),
                        [],
                        everywhere,
                    )

# An empty Time Bars mode looks the same whether the runs were never timed or
# the column did not decode, so the case that reached no environment is named.
timed_anywhere = any(
    (frame["_kind"] == "time").any() for _, frame, _, _ in env_tabs
)
if env_tabs and not timed_anywhere:
    print(
        "WARNING: no run recorded a readable time, so the Time Bars mode draws "
        "nothing."
    )

# Every curve any environment drew, in `_curve_i` order and each once. It is the
# colour scale's domain on every tab: Vega hands out a scheme's colours by
# position in the domain, so a domain per tab would shift a curve's colour
# wherever an environment is missing a curve that comes before it.
curve_domain = list(dict.fromkeys(
    c for _, c in sorted({
        (int(i), c)
        for _, frame, _, _ in env_tabs
        for i, c in zip(frame["_curve_i"], frame["_curve"])
        if pd.notna(i) and pd.notna(c)
    })
))

for env_name, long_df, filter_fields, laid_out in env_tabs:
    panels.append({
        "title": env_name,
        "spec": build_spec(long_df, filter_fields, laid_out, curve_domain),
        # The same panel with a symlog y axis, which the page swaps in when the
        # box is ticked: a scale's type cannot be driven by a signal, so the
        # alternative is written out rather than computed in the browser.
        #
        # Written WITHOUT its data: the two specs differ in one scale and agree on
        # every row, and the rows are the whole weight of the page. The page hands
        # this one the other's data before embedding it.
        "spec_log": {
            k: v for k, v in build_spec(
                long_df, filter_fields, laid_out, curve_domain, log_axis=True
            ).items()
            if k != "data"
        },
        # "Same y-lim", on either axis: a scale's resolve is fixed at compile time
        # just as its type is. Without data for the same reason.
        **{
            f"spec{'_log' if log else ''}_same_y": {
                k: v for k, v in build_spec(
                    long_df,
                    filter_fields,
                    laid_out,
                    curve_domain,
                    log_axis=log,
                    same_y=True,
                ).items()
                if k != "data"
            }
            for log in (False, True)
        },
    })

if args.prepare_missing_runs:
    path, n_jobs = write_missing_runs_script(args.folder, missing_runs)
    print(f"\nMissing runs script: {path} ({n_jobs} entries)")

if not panels:
    raise SystemExit("No environment produced a curve, so no page was written.")

# After the environments and before the parameters: the same bars with the
# environment dimension averaged away, which is the number a table would report.
avg_auc_spec = build_avg_auc_spec(
    avg_auc_rows, sorted(avg_block_labels.items()), curve_domain
)
if avg_auc_spec is None:
    vprint("\nNo area to average, so the avg. AUC tab was not written.")
else:
    panels.append({"title": "avg. AUC", "spec": avg_auc_spec})
    vprint(
        f"\navg. AUC: {len(avg_auc_rows)} bar(s) over "
        f"{len({r['_env'] for r in avg_auc_rows})} environment(s)"
    )

# Last tab, after the environments: the same runs regrouped by one hyperparameter
# at a time rather than by the sweep's layout.
param_spec = build_param_spec(param_rows) if param_rows else None
if param_spec is None:
    print(
        "No hyperparameter varies under any sweep entry, so the "
        "parameters tab was not written."
    )
else:
    panels.append({"title": "parameters", "spec": param_spec})
    vprint(f"\nparameters: {len(param_rows)} run score(s)")

with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_page(page_title, panels, RENDER_JS))

print(
    f"\n{len(envs_present)} environment(s), {total_cells} grid cell(s) total, "
    f"{len(panels)} tab(s) -> {out_path}"
)
