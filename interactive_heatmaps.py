"""Visitation heatmaps from memory.npz, as one interactive page.

Writes <folder>/plots/<sweep>_heatmaps.html, laid out on interactive_curves.py's
grid: one stacked block per `config_sweep` entry, and subplots
inside a block for the hyperparameters the sweep declares. What fills a cell is
the difference. interactive_curves.py draws a curve per `global_sweep`
combination, averaged over seeds; here a cell is one heatmap of one run, so there
are no curves at all and every declared hyperparameter becomes an axis.

Seeds are not averaged, so every seed is read and one is shown at a time, chosen
from a dropdown. --rng_seed cuts the dropdown down to the seeds it names.

There is a slider selecting which checkpoint of the replay memory the visits are
counted up to (see --progression_step). It moves over the checkpoints themselves
rather than over percentages, so every one of them is reachable whatever the
step; the percentage a map stands for is in its tooltip.

"Log scale" and "Shared scale" control the colour scale. The latter pins each
VISIBLE map to the range that map takes across the whole environment -- one range
per map, not one for all of them, since the maps hold different quantities.

Environments split tabs rather than sharing one: their maps have different shapes
and hold counts on incomparable scales. Only the environments the sweep declares
get one.

A run is drawn when its whole configuration is one the sweep COMPOSES -- the keys
an entry names, plus everything the config files it selects and
configs/default.yaml fill in under them -- not merely when it agrees on the keys
the entry writes down. Runs matching nothing are reported and dropped. Composing
goes through Hydra, so this script needs hydra installed and CONFIG_DIR readable.

--include and --exclude narrow the runs that are read, which keeps the page down
to a size a browser can open when a sweep declares many hyperparameters.

Example

    python interactive_heatmaps.py -f data_encoders --sweep encoders --progression_step 0.1
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.plot import (
    ALGO,
    ENV,
    ENV_ID,
    SEED,
    SweepLayout,
    apply_filters,
    assign_configs,
    declared_environments,
    flatten_cfg,
    html_page,
    hyperparameter_to_label,
    keep_composed_runs,
    keep_declared_environments,
    load_config_group,
    memory_fractions,
    swept_keys,
    value_str,
    iter_cells,
    label_of,
    values_of,
    vega_field_name,
)
from src.utils.heatmaps import compute_maps_all_fractions, pooled_range
from src.utils.slurm import parse_seeds
from src.utils.sweep import load_sweep

SUBPLOT_SIZE = 1.3  # inches per heatmap, the size a cell is drawn from
NEEDED_MEM_KEYS = ("obs", "count")

# The seed a row carries is the directory name, not cfg.yaml's own
# experiment.rng_seed, which every seed of a configuration shares. SEED is
# outside the "hyperparameter" section, so it can never become an axis of the
# grid either way.

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--folder",
    required=True,
    metavar="DIR",
    help="Data directory holding <config_id>/<rng_seed>/memory.npz.",
)
parser.add_argument(
    "--sweep",
    required=True,
    help="Sweep file in SWEEP_DIR (name without .yaml, or a path). It decides the "
         "layout: one block per config_sweep entry, an axis per key it declares, "
         "and a cell for every configuration it expects, run or not.",
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
    "--include",
    nargs="+",
    default=[],
    metavar="KEY=VALUE",
    help="Keep only matching runs. Same KEY twice is OR, different KEYs are AND.",
)
parser.add_argument(
    "--exclude",
    nargs="+",
    default=[],
    metavar="KEY=VALUE",
    help="Drop matching runs.",
)
parser.add_argument(
    "--rng_seed",
    nargs="+",
    default=[],
    metavar="SEEDS",
    help="Draw only these seeds: ints are literal seeds, a-b an inclusive range. "
         "E.g. --rng_seed=0, --rng_seed 0-5 9. Omit to draw every seed found.",
)
parser.add_argument(
    "--progression_step",
    type=float,
    default=1.0,
    help="Step (in (0, 1]) between memory-fraction checkpoints. E.g. 0.25 gives "
         "checkpoints at 25/50/75/100%% of the memory. Default 1.0 = the full "
         "memory only.",
)
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

try:
    FRACTIONS = memory_fractions(args.progression_step)
except ValueError as err:
    parser.error(f"--progression_step {err}")

# Configuration directories whose cfg.yaml is missing, and runs a cell already
# holds a map for. Both are collected while reading and reported once: a cell
# draws one map, so a second run of the same seed and dropdown values would be
# painted over the first with nothing on the page to say so.
missing_cfg = []
duplicate_runs = []

def vprint(*a):
    """Print only under --verbose."""

    if args.verbose:
        print(*a)


def runs_of(cell):
    """The runs a cell stands for, as [(run name, seed, {field: value})].

    A cell holds one run per (seed, combination of the `global_sweep` keys), since
    those keys are not on the grid, and the dropdowns pick which of them is drawn
    -- one per dropdown, or, on "(all)", every one of them, split across the grid's
    rows. The third element is what the page filters and splits on, keyed by the
    field names the rows carry.

    One run per (seed, combination): a cell draws one map, so a second run
    passing the same filters would be painted over the first. The later ones are
    left out and collected in `duplicate_runs`, which is reported rather than
    silently resolved -- two runs in one cell means a hyperparameter the sweep
    does not declare is varying under it."""

    found, seen = [], {}
    for name, seed in zip(cell.index, cell[SEED]):
        values = {
            vega_field_name(LABELS, key): value_str(
                cell.loc[name, ("hyperparameter", key)]
            )
            for key in GLOBAL_KEYS
        }
        shown_as = (
            int(seed),
            tuple(values[vega_field_name(LABELS, k)] for k in GLOBAL_KEYS),
        )
        if shown_as in seen:
            duplicate_runs.append((name, seen[shown_as]))
            continue
        seen[shown_as] = name
        found.append((name, int(seed), values))
    return found


def resolve_runs(root, keep_seeds=None):
    """Walk the data directory and return (seed directories to read, configuration
    directories with nothing to draw).

    Every seed of every configuration is read, and the page's dropdown chooses
    which one is drawn, so that the maps of two seeds can be compared without
    reading the data twice. `keep_seeds`, when given, is the set of seeds the
    dropdown is cut down to; every other seed directory is skipped.

    The second list is what a configuration still running looks like -- or one
    whose only seeds were cut by `keep_seeds` -- and it is read too: its
    hyperparameters are what put a column in the grid, so it holds its place
    there and draws empty instead of dropping out and shifting the rest along."""

    found, without = [], []
    for cfg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        seeds = sorted(
            (
                p for p in cfg_dir.iterdir()
                if p.is_dir() and p.name.isdigit()
                and (p / "memory.npz").is_file()
                and (keep_seeds is None or int(p.name) in keep_seeds)
            ),
            key=lambda p: int(p.name),
        )
        found.extend(seeds)
        if not seeds and (cfg_dir / "cfg.yaml").is_file():
            without.append(cfg_dir)
    return found, without


# How a panel of this page is drawn, handed to plot.html_page and dropped into
# the shell it keeps (which defines `panels`, `views`, `embedInto` and `show`).
RENDER_JS = """
  // Rows and columns trade places for every tab at once, so that two of them are
  // never read side by side under different orientations. `flipState` is the one
  // each panel's spec is currently built for: a tab drawn under the other is
  // swapped and redrawn when it is next shown, since a hidden chart has no width
  // to lay out from.
  const flipState = panels.map(() => false);
  let flipped = false;

  // Every bound param's value, to be carried across a rebuild. Nothing to carry
  // before the panel has a view.
  function carriedOf(i) {
    const view = views[i];
    return view && Object.fromEntries(
      panels[i].spec.params.filter(p => p.bind).map(p => [p.name, view.signal(p.name)]));
  }

  // `carry` is every bound param's value, kept across a rebuild. The rows split
  // by the keys whose dropdown is on "(all)" (split_<field>), and the facet cannot
  // regroup in place, so moving a dropdown onto or off "(all)" embeds the spec
  // again with the split to match.
  function render(i, carry) {
    const spec = panels[i].spec;
    // A facet's channels are fixed when the spec is compiled, so the swap lives
    // in the spec, and every rebuild -- this one and the ones "(all)" asks for --
    // keeps it.
    if (flipState[i] !== flipped) {
      flipState[i] = flipped;
      spec.vconcat.forEach(b => {
        [b.facet.row, b.facet.column] = [b.facet.column, b.facet.row];
      });
    }
    const splits = spec.params.filter(p => p.name.startsWith('split_'))
                              .map(p => p.name.slice('split_'.length));
    spec.params.forEach(p => {
      if (carry && p.name in carry) { p.value = carry[p.name]; }
    });
    splits.forEach(f => {
      spec.params.find(p => p.name === 'split_' + f).value =
        spec.params.find(p => p.name === 'p_' + f).value === 'All';
    });
    embedInto(i, spec).then(function (view) {
      // Undefined when the spec was rejected; the shell has put the error where
      // the plots go.
      if (!view) { return; }
      const carried = () => Object.fromEntries(
        spec.params.filter(p => p.bind).map(p => [p.name, view.signal(p.name)]));
      splits.forEach(f => {
        view.addSignalListener('p_' + f, (name, value) => {
          // null for a moment while the dropdown changes.
          if (value == null || (value === 'All') === view.signal('split_' + f)) { return; }
          // Not from inside the listener: the view is in the middle of an update.
          const kept = carried();
          setTimeout(() => render(i, kept), 0);
        });
      });
      // Rows and columns trade places in every block, and in every other tab
      // the next time it is shown. Its own text says which way it is round: the
      // tab it was pressed on may have been left and come back to since.
      const flip = document.createElement('button');
      flip.textContent = flipped ? 'Unflip rows/cols' : 'Flip rows/cols';
      flip.onclick = () => {
        flipped = !flipped;
        render(i, carried());
      };
      document.getElementById('controls' + i).appendChild(flip);
    });
  }

  // Shown again after being drawn: only the orientation can have moved under it,
  // and only from another tab.
  function redraw(i) {
    if (flipState[i] !== flipped) { render(i, carriedOf(i)); }
  }
"""


def build_spec(
    long_df,
    blocks,
    keys,
    fractions,
    seeds,
    grid,
    log_default,
    shared_default,
    shared_ranges,
    cell_px,
):
    """Vega-Lite spec: one heatmap per cell, a slider over the checkpoints and
    dropdowns over the maps and the seeds.

    Every seed is in the data and one is shown at a time, so switching seed
    redraws from what is already loaded and the grid keeps its shape.

    The filters are applied inside the mark rather than before the facet, so a
    cell with no run stays an empty cell instead of collapsing the grid.

    `grid` is the (height, width) of the maps, and it pins the x and y domains. A
    bin never visited holds zero, which `orient` turns into NaN and which
    therefore contributes no row, so an ordinal scale left to infer its domain
    takes it from the bins that happen to be filled -- and a policy visiting few
    of them, or an early checkpoint, would draw those few stretched across the
    whole panel and shift them as the slider fills the map in."""

    # The slider moves over the checkpoints' positions, not their percentages: a
    # --progression_step that does not divide 1 puts the appended full memory off
    # the percentage grid, where a slider cannot land on it. The percentage is
    # in the binding's name and in the tooltip.
    percents = [round(f * 100) for f in fractions]
    span = f"{percents[0]}-{percents[-1]}%" if len(percents) > 1 else f"{percents[0]}%"

    params = [
        {
            "name": "p_i",
            "value": len(fractions) - 1,
            "bind": {
                "input": "range",
                "min": 0,
                "max": len(fractions) - 1,
                "step": 1,
                "name": f"Memory checkpoint ({span}): ",
            },
        },
        {
            "name": "p_key",
            "value": keys[0],
            # The map's own key, as the data writes it: the page has no plot
            # config to read display names from.
            "bind": {"input": "select", "options": keys, "name": "Map: "},
        },
        {
            "name": "p_seed",
            "value": seeds[0],
            "bind": {"input": "select", "options": seeds, "name": "Seed: "},
        },
        {
            "name": "p_shared",
            "value": bool(shared_default),
            "bind": {"input": "checkbox", "name": "Shared scale: "},
        },
        {
            "name": "p_log",
            "value": bool(log_default),
            "bind": {"input": "checkbox", "name": "Log scale: "},
        },
    ]
    # One dropdown per `global_sweep` key. A cell holds one map, so "(all)" does not
    # overlay the values the way interactive_curves.py overlays curves: it splits
    # every row of the grid into one row per value, one under the other.
    for key in GLOBAL_KEYS:
        field = vega_field_name(LABELS, key)
        params.append({
            "name": f"p_{field}",
            "value": GLOBAL_OPTIONS[key][0],
            # Shown as "(all)"; the value stays 'All', which the filters and the
            # page's rebuild compare against.
            "bind": {
                "input": "select",
                "options": ["All"] + GLOBAL_OPTIONS[key],
                "labels": ["(all)"] + GLOBAL_OPTIONS[key],
                "name": f"{LABELS[key]}: ",
            },
        })
        # Unbound: the page sets it from the dropdown above (see _rowx).
        params.append({"name": f"split_{field}", "value": False})
    # log1p, not log: a bin holding a single visit stays distinguishable from an
    # empty one, and the counts start at 1 rather than at 0.
    value_expr = "p_log ? log(datum.count + 1) : datum.count"
    shown = " && ".join(
        ["datum._key === p_key", "datum._seed === p_seed", "datum._i === p_i"]
        + [
            f"(p_{f} === 'All' || datum.{f} === p_{f})"
            for f in (vega_field_name(LABELS, k) for k in GLOBAL_KEYS)
        ]
    )
    # The group a datum lands in on the facet's row channel -- its column channel
    # once the page's flip button has swapped the two, which is what the "row" in
    # these names is: the orientation the page opens under, not the only one it
    # draws. It is its grid row, split by the value of every key on
    # "All". Which keys split is `split_<field>`, set by the page from the
    # dropdowns when it embeds the spec, and the page embeds it again whenever a
    # dropdown moves onto or off "All". Following the dropdown in place does not
    # work: a Vega dataset ends in a Sieve, which drops the record of which fields
    # changed, so the facet never moves a modified row -- and a crossed facet
    # never drops a cell it has drawn, so rows added and removed leave empty maps
    # behind.
    #
    # The group is prefixed with its place in the order and sorted as text, and
    # the header strips the prefix off again: the grid's rows in the sweep's
    # order, and within one the values in the dropdowns' order (GLOBAL_COMBOS's,
    # first key outermost). A sort by another field would cost an aggregate over
    # every row.
    #
    # A value the dropdown does not offer -- a run the sweep does not declare --
    # is not in the list, and `indexof` answers -1 for it. Taken as an index that
    # would place the group before the one it belongs after, and on top of
    # whichever group is already there; it is pinned to 0 instead, which draws it
    # first among its own. Its label carries the value, so it still has a group of
    # its own to be drawn in.
    # Values reach the expression through json.dumps, never repr: a Python repr
    # of a list of strings is valid JavaScript by coincidence, and stops being so
    # the moment a label holds a quote or a backslash.
    split_label, split_index, stride = "", "", len(GLOBAL_COMBOS)
    for k in GLOBAL_KEYS:
        f = vega_field_name(LABELS, k)
        stride //= len(GLOBAL_OPTIONS[k])
        prefix = json.dumps(f" {LABELS[k]}=")
        split_label += f" + (split_{f} ? {prefix} + datum.{f} : '')"
        split_index += (
            f" + (split_{f} ? "
            f"max(indexof({json.dumps(GLOBAL_OPTIONS[k])}, datum.{f}), 0)"
            f" * {stride} : 0)"
        )

    # The bounds "Shared scale" pins the colour to, one pair per map: the maps of
    # one environment hold different quantities on different scales, and a single
    # pooled pair would compress the smaller of them into one colour. The map is
    # a dropdown, so the bound follows it as an expression.
    #
    # Both units are computed here rather than in the page: the log values come
    # out of numpy, where a bound the logarithm is undefined for can be dropped,
    # while a NaN written into an expression is an identifier Vega does not know
    # and would leave the tab showing a parse error instead of maps.
    #
    # A map with no finite bounds falls through to null, which is Vega's "take
    # the domain from the data" -- the same thing the box being unticked does.
    def bounds_expr(end, logged):
        """The `domainMin` or `domainMax` a map is pinned to under "Shared
        scale", as a Vega expression choosing on the map dropdown."""

        parts = []
        for key, pair in shared_ranges.items():
            value = float(pair[end])
            if not np.isfinite(value):
                continue
            if logged:
                # log1p is undefined at or below -1; the linear bound is not.
                if value <= -1:
                    continue
                value = float(np.log1p(value))
            parts.append(f"p_key === {json.dumps(key)} ? {value!r}")
        return " : ".join(parts + ["null"])

    color_scale = {
        "scheme": "viridis",
        "domainMin": {
            "expr": f"!p_shared ? null : (p_log ? ({bounds_expr(0, True)})"
                    f" : ({bounds_expr(0, False)}))"
        },
        "domainMax": {
            "expr": f"!p_shared ? null : (p_log ? ({bounds_expr(1, True)})"
                    f" : ({bounds_expr(1, False)}))"
        },
    }

    def block_spec(index, block):
        # The sweep's order, not the alphabetical one a nominal field defaults to.
        row_order = [
            label_of(LABELS, block.row_params, v) or " " for v in block.row_combos
        ]
        col_order = [
            label_of(LABELS, block.col_params, v) or " " for v in block.col_combos
        ]
        # Width of the order prefix on a row (see _rowx).
        digits = len(str(len(row_order) * len(GLOBAL_COMBOS)))

        # Ticking "Shared scale" pins every map to the range of the whole
        # environment; unticking it hands each cell back its own. Both are the
        # same scale with a different domain, so the box moves a signal and the
        # view recomputes only what depends on it -- where rewriting `resolve`
        # would mean compiling and embedding the spec again, which on a page this
        # size means re-reading every row.
        #
        # One bar per cell either way. Under a shared domain they all read the
        # same, which is a repetition; a single bar would need the blocks to share
        # a scale, and that is the `resolve` this avoids touching.
        legend = {"title": None}
        resolve = {"scale": {"color": "independent"}}

        return {
            "title": block.label or None,
            # Position, not title: an entry that pins nothing has no title, and two
            # of them would otherwise select each other's rows.
            "transform": [
                {"filter": f"datum._block === {index}"},
                {
                    "calculate":
                        f"pad('' + (max(indexof({json.dumps(row_order)}, "
                        f"datum._row), 0) * "
                        f"{len(GLOBAL_COMBOS)}{split_index}), {digits}, '0', "
                        f"'left') + '|' + datum._row{split_label}",
                    "as": "_rowx",
                },
            ],
            "facet": {
                # A row's label styling is config.headerRow's, so that flipping
                # rows and columns (a button on the page) swaps these two alone.
                "row": {
                    "field": "_rowx",
                    "type": "nominal",
                    "title": None,
                    "header": {
                        "labelExpr": f"slice(datum.value, {digits + 1})",
                    },
                },
                "column": {
                    "field": "_col",
                    "type": "nominal",
                    "title": None,
                    "sort": col_order,
                },
            },
            "spec": {
                "width": cell_px,
                "height": cell_px,
                "layer": [
                    {
                        # One invisible rect per cell, drawn from the placeholder
                        # row every cell has. Without it a cell whose run has not
                        # finished holds nothing a mark can draw -- its only datum
                        # has a NaN count, which the rect mark filters out as an
                        # invalid value -- and the facet drops the empty group
                        # instead of leaving a gap. This layer cannot be filtered
                        # away: it encodes no quantity, only a position, so the
                        # group always renders, bordered and headed like the rest.
                        "transform": [{"filter": "datum._placeholder === true"}],
                        "mark": {"type": "rect", "opacity": 0},
                        "encoding": {
                            "x": {
                                "field": "x",
                                "type": "ordinal",
                                "axis": None,
                                "scale": {"domain": list(range(grid[1]))},
                            },
                            "y": {
                                "field": "y",
                                "type": "ordinal",
                                "axis": None,
                                "scale": {"domain": list(range(grid[0]))},
                            },
                        },
                    },
                    {
                        "transform": [
                            {"filter": shown},
                            {"calculate": value_expr, "as": "_value"},
                        ],
                        "mark": {"type": "rect"},
                        "encoding": {
                            "x": {
                                "field": "x",
                                "type": "ordinal",
                                "axis": None,
                                "scale": {"domain": list(range(grid[1]))},
                            },
                            "y": {
                                "field": "y",
                                "type": "ordinal",
                                "axis": None,
                                "scale": {"domain": list(range(grid[0]))},
                            },
                            "color": {
                                "field": "_value",
                                "type": "quantitative",
                                "scale": color_scale,
                                "legend": legend,
                            },
                            # The tooltip reads the count, never its logarithm,
                            # and names the checkpoint the slider is on.
                            "tooltip": [
                                {"field": "count", "type": "quantitative"},
                                {
                                    "field": "_pct",
                                    "type": "quantitative",
                                    "title": "memory %",
                                },
                                {"field": "x", "type": "ordinal"},
                                {"field": "y", "type": "ordinal"},
                            ],
                        },
                    },
                ],
            },
            "resolve": resolve,
        }

    # Only the placeholder rows carry `_placeholder` and only they lack a count,
    # so every other row would be written with a NaN in each of those fields --
    # 0.8 MB of them here, and an object no JSON parser will read. An entry that
    # is not there means the same thing to Vega as one that is NaN: invalid, and
    # filtered out of the mark. NaN is the only value that fails `v == v`.
    values = [
        {k: v for k, v in record.items() if v == v}
        for record in long_df.to_dict(orient="records")
    ]

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": values},
        "params": params,
        "vconcat": [block_spec(i, block) for i, block in enumerate(blocks)],
        "config": {
            "view": {"stroke": "#888"},
            "axis": {"grid": False},
            "headerRow": {"labelAngle": 0, "labelAlign": "left"},
        },
    }


# --- Read the runs ---------------------------------------------------------

root = Path(args.folder)
if not root.is_dir():
    raise SystemExit(f"{root} is not a directory.")

keep_seeds = set(parse_seeds(args.rng_seed)) if args.rng_seed else None
if keep_seeds is not None:
    print(f"Drawing seed(s) {sorted(keep_seeds)} only.")

run_dirs, without_memory = resolve_runs(root, keep_seeds)
if without_memory:
    names = [p.name for p in without_memory]
    print(
        f"WARNING: {len(names)} configuration(s) have no memory.npz to draw and "
        f"are drawn empty: {names[:5]}{' ...' if len(names) > 5 else ''}. "
        f"memory.npz is only written when results.save_memory=True."
    )
if not run_dirs:
    raise SystemExit(
        f"No <config_id>/<rng_seed>/memory.npz under {root}"
        + (f" for seed(s) {sorted(keep_seeds)}." if keep_seeds else ".")
    )

NO_SEED = -1  # a configuration that has not saved a memory for any seed

flat_cfgs, run_cfgs, mem_files, run_seeds = {}, {}, {}, {}


def read_cfg(cfg_dir, name, mem_file, seed):
    """Put one entry in the frame. A `mem_file` of None is a configuration with
    nothing to draw yet, and it is here for its hyperparameters alone."""

    cfg_file = cfg_dir / "cfg.yaml"
    if not cfg_file.is_file():
        missing_cfg.append(cfg_dir.name)
        return
    run_cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    # Interpolations kept, as src.utils.id.composed_config keeps them: the two
    # are compared key by key below, and a key dropped from one side and not the
    # other is a key the comparison reads as missing from the run.
    flat_cfgs[name] = flatten_cfg(run_cfg, drop_interpolations=False)
    run_cfgs[name] = run_cfg
    mem_files[name] = mem_file
    run_seeds[name] = seed


for seed_dir in run_dirs:
    # cfg.yaml sits one level up, shared by every seed of the configuration.
    read_cfg(
        seed_dir.parent,
        f"{seed_dir.parent.name}/{seed_dir.name}",
        seed_dir / "memory.npz",
        int(seed_dir.name),
    )
# Configurations still running keep a place in the grid: no seed of theirs has a
# memory, so they are empty under every seed the dropdown offers.
for cfg_dir in without_memory:
    read_cfg(cfg_dir, f"{cfg_dir.name}/-", None, NO_SEED)

if missing_cfg:
    names = sorted(set(missing_cfg))
    print(
        f"WARNING: {len(names)} configuration(s) have a memory but no cfg.yaml "
        f"and are not drawn: {names[:5]}{' ...' if len(names) > 5 else ''}."
    )

if not flat_cfgs:
    raise SystemExit(
        f"Found {len(run_dirs)} memory.npz but no cfg.yaml beside any of them."
    )

df = pd.DataFrame.from_dict(flat_cfgs, orient="index")
df.columns = pd.MultiIndex.from_tuples([("hyperparameter", c) for c in df.columns])
df[SEED] = pd.Series(run_seeds)
vprint(f"\nRead {len(df)} run(s) from {root}")

global_sweep, sweep = load_sweep(args.sweep)
swept = swept_keys(global_sweep, sweep)

# Without the config YAMLs there is nothing to match against, and every run would
# be labelled the same -- one figure holding maps of different shapes on
# incomparable scales. The id the run recorded keeps them apart instead.
env_configs = load_config_group(args.environments) if args.environments else {}
if env_configs:
    df[ENV] = assign_configs(df, env_configs, swept)
elif ENV_ID in df.columns:
    df[ENV] = df[ENV_ID].map(value_str)
else:
    df[ENV] = "unknown"
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

if df.empty:
    raise SystemExit("--include/--exclude left no runs.")

# Only the environments the sweep declares, and under them only the runs it
# composes.
declared_envs = declared_environments(sweep, global_sweep)
if declared_envs is None:
    vprint(
        "\nAt least one sweep entry declares no environment, so every "
        "environment in the data is drawn."
    )
df = keep_declared_environments(df, declared_envs, "drawn")
df = keep_composed_runs(df, sweep, global_sweep, "drawn", vprint)

LABELS = hyperparameter_to_label(
    sorted(
        {c[1] for c in df.columns if c[0] == "hyperparameter"}
        | set(global_sweep)
        | {k for entry in sweep for k in entry}
    )
)

# After the filters, not before: a seed no remaining run has would still be
# offered by the dropdown, and the page opens on the first seed of the list, so
# it would open on empty cells.
#
# Plain ints: the seeds reach the Vega-Lite spec through json.dumps, which has no
# encoder for numpy scalars. NO_SEED is not one of them -- it belongs to no run.
SEEDS = sorted({int(s) for s in df[SEED].unique() if int(s) != NO_SEED})
if not SEEDS:
    raise SystemExit("No run with a saved memory is left after the filters.")
vprint(f"Seed(s) drawn: {SEEDS}")

# `global_sweep` keys with more than one value: one dropdown each, and one field
# per key on every row for the dropdown to filter on. A key the data does not
# record cannot be filtered, so it is left out rather than emptying the page.
# The environment splits the tabs and the seed has a dropdown of its own, so
# neither is one of these however the sweep declares them.
GLOBAL_KEYS = [
    k for k, declared in global_sweep.items()
    if len(declared) > 1
    and k not in ("environment", "experiment.rng_seed")
    and ("hyperparameter", k) in df.columns
]
GLOBAL_OPTIONS = {
    k: [value_str(v) for v in values_of(df, ("hyperparameter", k), global_sweep[k])]
    for k in GLOBAL_KEYS
}
# Every combination of those keys' values, as the fields a row carries: a cell
# gets a placeholder for each of them it was run with, so "All" has a row to open
# for every value it holds, even one whose run has not finished.
GLOBAL_COMBOS = [{}]
for k in GLOBAL_KEYS:
    GLOBAL_COMBOS = [
        {**c, vega_field_name(LABELS, k): v}
        for c in GLOBAL_COMBOS for v in GLOBAL_OPTIONS[k]
    ]
if GLOBAL_KEYS:
    vprint(f"Dropdowns: {[LABELS[k] for k in GLOBAL_KEYS]}")

# The grid, laid out the way interactive_curves.py lays it out. `curves=False`:
# a cell holds one map, so there is nothing to draw several of in one of them
# and the `global_sweep` keys become the dropdowns above instead of curve axes.
layout = SweepLayout(df, sweep, global_sweep, LABELS, curves=False, vprint=vprint)

# --- Load the memories -----------------------------------------------------

# Where each checkpoint sits in the list. The slider moves over these positions
# rather than over the percentages: a step that does not divide 1 leaves the
# appended full memory off the percentage grid, and a slider cannot land on a
# value its own step does not reach. The percentage is in the tooltip.
FRACTION_INDEX = {f: i for i, f in enumerate(FRACTIONS)}
vprint(f"Checkpoints: {FRACTIONS}")

maps = {}  # run name -> {fraction: {map key: 2D array}}
for name in df.index:
    # Nothing saved yet: it is in the frame to hold its cell, not to be drawn.
    if mem_files[name] is None:
        continue
    try:
        with np.load(mem_files[name]) as data:
            mem = {k: data[k] for k in data.files if k in NEEDED_MEM_KEYS}
    except Exception as e:
        print(f"WARNING: could not read {mem_files[name]}: {e}")
        continue
    try:
        per_fraction = compute_maps_all_fractions(run_cfgs[name], mem, FRACTIONS)
    except Exception as e:
        print(f"WARNING: could not compute maps for {name}: {e}")
        continue
    if per_fraction is None:
        vprint(f"{name}: environment has no tabular counter, skipping.")
        continue
    maps[name] = per_fraction

if not maps:
    raise SystemExit(
        "No run produced a map. The environments may have no tabular counter."
    )

MAP_KEYS = sorted({
    k for per_frac in maps.values() for per_key in per_frac.values() for k in per_key
})
vprint(f"Maps: {MAP_KEYS}, drawn for {len(maps)} run(s)")

# The same root every plotting script writes under, so one data directory holds
# all of their output side by side.
out_dir = os.path.join(args.folder, "plots")
os.makedirs(out_dir, exist_ok=True)
# Browser tab name: the data directory, then the sweep it was laid out from.
sweep_name = os.path.splitext(os.path.basename(args.sweep))[0]
folder_name = os.path.basename(os.path.normpath(args.folder))
page_title = f"{folder_name} ({sweep_name}) - heatmaps"
# File name: the sweep the page was laid out from, and what it draws.
out_path = os.path.join(out_dir, f"{sweep_name}_heatmaps.html")

panels = []
total_rows = 0

for env in sorted(df[ENV].unique(), key=str):
    # Named as the data writes it, overrides and all: the page has no plot config
    # to read display names from, and a tab has to be findable by the name the
    # runs were recorded under.
    env_name = str(env)
    df_env = df[df[ENV] == env]
    vprint(f"\n{'=' * 60}\nEnvironment: {env_name}\n{'=' * 60}")

    blocks, _curve_axes = layout.blocks(df_env)
    if not blocks:
        continue
    for i, block in enumerate(blocks):
        vprint(
            f"Block {block.label or i + 1}: "
            f"{len(block.row_combos)}x{len(block.col_combos)} cell(s), "
            f"columns {[c[1] for c in block.col_params]}, "
            f"rows {[c[1] for c in block.row_params]}"
        )

    # Pooled per map, not over all of them at once: "Shared scale" pins the
    # colour to the range of the map on screen, and a visit count and a validity
    # mask do not share a range.
    env_by_key = {}
    for name in df_env.index:
        if name in maps:
            for per_key in maps[name].values():
                for key, m in per_key.items():
                    env_by_key.setdefault(key, []).append(m)
    env_arrays = [m for arrays in env_by_key.values() for m in arrays]
    if not env_arrays:
        vprint(f"No map for {env_name}, skipping.")
        continue
    env_ranges = {k: pooled_range(arrays) for k, arrays in env_by_key.items()}

    rows = []
    for b_i, block in enumerate(blocks):
        for _, row_vals, _, col_vals, cell in iter_cells(block):
            cell_id = {
                "_block": b_i,
                "_row": label_of(LABELS, block.row_params, row_vals) or " ",
                "_col": label_of(LABELS, block.col_params, col_vals) or " ",
            }
            cell_runs = runs_of(cell)
            # The combinations this cell holds a run for, whether or not that run
            # has saved a memory yet. A combination it was never run with opens a
            # row under "All" that nothing can ever fill, so it gets no
            # placeholder; a cell holding no run at all is a configuration the
            # sweep expects and nothing has been read for, and it keeps one of
            # each rather than dropping out of the grid.
            ran = {
                tuple(values[vega_field_name(LABELS, k)] for k in GLOBAL_KEYS)
                for _, _, values in cell_runs
            }
            combos = [
                combo for combo in GLOBAL_COMBOS
                if tuple(combo[vega_field_name(LABELS, k)] for k in GLOBAL_KEYS) in ran
            ] or GLOBAL_COMBOS
            # One placeholder row per cell and combination, so that "All" finds
            # one in each of its rows, whatever ran for it. It is what
            # gives the facet a value to make a cell out of, and it passes the
            # filter inside the mark rather than being dropped by it, so the
            # cell is never left without a row: a configuration still running
            # keeps its place in the grid and draws as an empty map, bordered
            # and titled like the rest. Its count is NaN, which paints nothing
            # and stays out of the colour scale.
            rows.extend(
                {
                    **cell_id,
                    **combo,
                    "_placeholder": True,
                    "_seed": -1,
                    "_i": -1,
                    "_pct": -1,
                    "_key": "",
                    "x": 0,
                    "y": 0,
                    "count": float("nan"),
                }
                for combo in combos
            )

            for name, seed, values in cell_runs:
                if name not in maps:
                    continue
                base = {**cell_id, "_seed": seed, **values}
                for frac, per_key in maps[name].items():
                    # A checkpoint outside the ones asked for has no position on
                    # the slider and cannot be selected.
                    frac_i = FRACTION_INDEX.get(frac)
                    if frac_i is None:
                        continue
                    for key, arr in per_key.items():
                        m = np.atleast_2d(arr)
                        ys, xs = np.nonzero(~np.isnan(m))
                        rows.extend(
                            {
                                **base,
                                "_i": frac_i,
                                "_pct": round(frac * 100),
                                "_key": key,
                                "x": int(x),
                                "y": int(y),
                                "count": float(m[y, x]),
                            }
                            for y, x in zip(ys, xs)
                        )
    if not rows:
        vprint(f"Nothing to draw for {env_name}, skipping.")
        continue
    long_df = pd.DataFrame(rows)
    # One environment, one counter, so every map of this panel is the same
    # grid; the largest covers a run whose env config differs in some other way.
    shapes = [np.atleast_2d(m).shape for m in env_arrays]
    grid = (max(s[0] for s in shapes), max(s[1] for s in shapes))
    total_rows += len(long_df)
    vprint(f"  Panel: {env_name}  ({len(long_df)} rows, {grid[0]}x{grid[1]} bins)")
    panels.append({
        "title": env_name,
        "spec": build_spec(
            long_df,
            blocks,
            MAP_KEYS,
            FRACTIONS,
            SEEDS,
            grid,
            log_default=False,
            shared_default=False,
            shared_ranges={
                k: (float(lo), float(hi)) for k, (lo, hi) in env_ranges.items()
            },
            cell_px=int(SUBPLOT_SIZE * 60),
        ),
    })
if not panels:
    raise SystemExit("No environment produced a map, so nothing was written.")

if duplicate_runs:
    shown = [f"{name} (behind {kept})" for name, kept in duplicate_runs[:5]]
    print(
        f"WARNING: {len(duplicate_runs)} run(s) share a cell, a seed and the same "
        f"dropdown values with another, and only the first is drawn: {shown}"
        f"{' ...' if len(duplicate_runs) > 5 else ''}. "
        f"Use --include/--exclude to pick between them."
    )

html = html_page(page_title, panels, RENDER_JS)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

# Every checkpoint, map and seed is in the file, and one of each is on screen:
# the page is as big as everything the dropdowns can reach. Said out loud,
# because the way it goes wrong is a browser that stops responding rather than
# an error anyone can read.
size_mb = len(html.encode("utf-8")) / 1e6
print(
    f"\n{len(maps)} run(s), {len(FRACTIONS)} checkpoint(s), "
    f"{len(panels)} tab(s), {total_rows} row(s), {size_mb:.1f} MB -> {out_path}"
)
if size_mb > 50:
    print(
        "WARNING: a page this size is slow to open and may hang the browser. "
        "A coarser --progression_step, or --include/--exclude, cuts it down."
    )
