"""Unreachable goal selections over training, from the replay-memory dumps.

For every goal SELECTION stored in a run's `memory.npz`, this checks whether the
goal tile could still be reached from the tile the agent was standing on. Which
steps are a selection follows the actor's own flags: `per_step_selection` without
`goal_value_check` chooses again at every step; otherwise the choices are the
first step of an episode and every step the goal changes. A goal pursued for
50 steps is therefore one selection, not 50. Reachability is the forward closure of the
environment's own transition graph, enumerated with `set_state` / `step` over every
(state, action) pair, expanding only non-terminal transitions. Where the doorways
of a gridworld are one-way, an agent that has dropped into a room it cannot leave
reaches only that room's tiles, so every goal it picks outside the room is
unachievable until the episode resets.

Goals are compared on states only. A goal is a (state, action) pair, but every
action is available in every non-wall tile, so the state decides achievability.

A figure each for two metrics, both a percentage of the goal selections and
binned over the memory (`--points` bins):

    Unreachable Goals  the goal could not be reached from where the agent stood
    Perfect Goals      the goal was the best pick among the tiles within reach that
                       the agent had already put in its replay memory. `--goal_score`
                       chooses what best means: 1 / n, a least-visited tile
                       (default), or the actor's own V / n

Which configurations are drawn, what they are called and which environments they
are drawn over comes from a plot config, read exactly as plot_results.py reads it,
and a configuration keeps the colour it has there. Environments without a tabular
`set_state` interface have no transition graph to close over and are skipped.

Bins where no goal was selected are NaN, leaving a gap. --log_scale draws the same
figure on a log y axis instead, where bins at exactly 0% drop out too, a log scale
having no room for them.

Output goes to <data_dir>/<output>/goals/<plot config>/:

    unreachable_goals_curves.png  the unreachable-goal figure
    perfect_goals_curves.png      the perfect-goal figure
    <metric>_auc.png              under --auc_only, that metric's AUC bars alone,
                                  one figure per environment group
    <metric>_auc_all_envs.png     the same bars over every environment at once,
                                  written whatever --auc_only says
    legend_horizontal.png         the legend on its own, laid out by the config
    legend_vertical.png           the legend in a single column

Example

    python plot_unreachable_goals.py -f data_gcrl -p gcrl/all --with_auc -v
"""

import argparse
import hashlib
import json
import os
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.utils.plot import (
    GROUP_GAP_BAR,
    assign_groups,
    composed_mask,
    detect_precision,
    draw_bars,
    ensure_font,
    entry_config,
    env_groups,
    error_shade_plot,
    fitted_rows_cols,
    flatten_cfg,
    ignored_cfg_keys,
    load_config_group,
    load_plot_configs,
    plot_labels,
    rows_cols,
    save_figure,
    set_3_ticks,
    style_entries,
    summarize,
    tex_kwargs,
)
from src.utils.heatmaps import env_context

FONT_SIZE = 12
SUBPLOT_W = 2.6  # inches per subplot
SUBPLOT_H = 2.0

# Width in inches of an --auc_only panel. Narrower than a curves subplot, since its
# scale is drawn inside the panel and needs no margin.
AUC_ONLY_WIDTH = 2.5
# Thickness of an --auc_only bar, as a fraction of the slot it sits in, and the
# space to the next one.
AUC_ONLY_BAR_THICKNESS = 0.65
AUC_ONLY_BAR_GAP = 0.03
# Inches of panel height per unit of that layout, so a bar keeps its thickness on
# the page and the panel shrinks with the gaps rather than the bars growing.
AUC_ONLY_UNIT_INCHES = 0.25
# Small enough for every configuration's label to fit beside its own bar.
AUC_ONLY_LABEL_SIZE = 7

CACHE_DIRNAME = ".goal_curves_cache"

METRICS = ("unreachable", "perfect")
METRIC_LABELS = {
    "unreachable": "Unreachable Goals (%)",
    "perfect": "Perfect Goals (%)",
}
METRIC_FILES = {
    "unreachable": "unreachable_goals",
    "perfect": "perfect_goals",
}
# What the bars are titled when they stand on their own, with no curves beside
# them to say what is being averaged.
METRIC_AUC_TITLES = {
    "unreachable": "Avg. Unreachable Goal %",
    "perfect": "Avg. Perfect Goal %",
}

# The actor clips candidate counts to this before dividing (src/actor.py), which is
# what keeps a never-visited tile finite rather than infinitely attractive.
COUNT_FLOOR = 1e-3

# A tile has to have been visited more than this to count as an available goal: the
# actor draws its candidates from the replay memory, so a tile that never reached
# it was never on offer and does not belong in the best-available score.
CANDIDATE_MIN_COUNT = 1

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--folder",
    required=True,
    metavar="DIR",
    help="Data directory holding the runs, each a <config_id>/<rng_seed> with a "
         "memory.npz in it.",
)
parser.add_argument(
    "-p", "--plot_config",
    required=True,
    metavar="NAME",
    help="Plot config: which configurations to draw, and over which environments. "
         "A name without .yaml is looked for in configs/plots, or give a path; a "
         "directory is a set of configs and every one of them is drawn.",
)
parser.add_argument(
    "-o", "--output",
    default="plots",
    help="Output root, under the data directory. Figures go in <root>/goals/<plot "
         "config>, so one data directory holds the output of every plotting script "
         "without one overwriting another.",
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
    "--points",
    type=int,
    default=None,
    help="Bins along the x axis. Default: the run's experiment.testing_points.",
)
parser.add_argument(
    "--smoothing_window",
    type=int,
    default=0,
    help="Points either side to average a curve over. 0 draws it as recorded.",
)
parser.add_argument(
    "--goal_score",
    default="novelty",
    choices=["novelty", "ratio"],
    help="Which score a perfect goal is the argmax of, over the tiles in reach: "
         "'novelty' is 1 / n, i.e. a least-visited tile; 'ratio' is the actor's own "
         "V / n, with V the discounted reachability gamma ** (steps to reach it).",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="Discount the 'ratio' score values a goal with: a goal d steps away is "
         "worth gamma ** d. Unused by 'novelty'.",
)
parser.add_argument(
    "--log_scale",
    action="store_true",
    help="Draw the curves on a log y axis, in place of the linear figure rather "
         "than alongside it.",
)
parser.add_argument(
    "--with_auc",
    action="store_true",
    help="Append an AUC subplot to the figures: one bar per configuration, its "
         "rate averaged over the environments of the group.",
)
parser.add_argument(
    "--auc_only",
    action="store_true",
    help="Write only the AUC bars, one figure per metric, with no curves beside "
         "them.",
)
parser.add_argument(
    "--annotate_auc",
    action="store_true",
    help="Print the value beside each bar of the AUC figures.",
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
parser.add_argument(
    "-j", "--jobs",
    type=int,
    default=0,
    help="Dumps to read at once. 0 picks one per CPU, capped at 8; 1 reads them "
         "one at a time.",
)
parser.add_argument(
    "--no_cache",
    action="store_true",
    help="Recompute every dump rather than reusing cached per-seed curves.",
)
parser.add_argument(
    "--cache_dir",
    default=None,
    metavar="DIR",
    help=f"Where per-seed curves are cached. Default: <folder>/{CACHE_DIRNAME}",
)
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()


def vprint(*a):
    """Print only under --verbose."""

    if args.verbose:
        print(*a)


# -----------------------------------------------------------------------------
# ----- Reachability ----------------------------------------------------------
# -----------------------------------------------------------------------------

_dist_cache: dict = {}


def build_distances(env_cfg):
    """(n_states, n_states) float matrix: `dist[s, g]` is the fewest steps from
    state `s` to state `g`, and `inf` when `g` cannot be got to from `s` at all.

    The transition graph is read off the environment itself -- `set_state(s)`, step
    every action, record `get_state()` -- so one-way doorways, absorbing tiles and
    any other layout quirk are captured without hard-coding the geometry. Returns
    None for environments without a tabular `set_state` / `get_state` interface,
    which is what makes this metric a gridworld one.

    Building an environment costs seconds and many runs share one, so the result is
    cached on the config it was built from."""

    key = json.dumps(env_cfg, sort_keys=True, default=str)
    if key in _dist_cache:
        return _dist_cache[key]

    from src.wrappers import gym_wrappers

    env = gym_wrappers.make_gym_env(**env_cfg)
    u = env.unwrapped
    try:
        if not all(hasattr(u, k) for k in ("set_state", "get_state", "grid_reachable")):
            _dist_cache[key] = None
            return None
        # Action noise would make the enumeration below random. It cannot change
        # which states are reachable (a noisy action is resampled from the same
        # action set), so zeroing it out gives the same graph, deterministically.
        for attr in ("random_action_prob", "slippery_prob", "random_reset_prob"):
            if hasattr(u, attr):
                setattr(u, attr, 0.0)
        env.reset()

        valid = np.asarray(u.grid_reachable).reshape(-1).astype(bool)
        n_states = int(valid.size)
        n_act = int(u.action_space.n)
        succ = np.full((n_states, n_act), -1, dtype=np.int64)
        term = np.zeros((n_states, n_act), dtype=bool)
        for s in np.flatnonzero(valid):
            for a in range(n_act):
                u.set_state(int(s))
                _, _, terminated, truncated, _ = env.step(a)
                succ[s, a] = int(u.get_state())
                term[s, a] = bool(terminated)
                if terminated or truncated:
                    env.reset()
    finally:
        env.close()

    # Breadth-first from every tile, so the result is the number of steps to the
    # goal and not merely whether it can be got to: the discounted reachability the
    # actor scores with is gamma raised to that distance. All edges cost one step,
    # so the queue leaves each tile at its shortest distance. A tile found only
    # through a terminal transition keeps its distance but is not expanded -- the
    # episode ends on arrival, so nothing beyond it is still available.
    dist = np.full((n_states, n_states), np.inf)
    for s in np.flatnonzero(valid):
        d = np.full(n_states, np.inf)
        expanded = np.zeros(n_states, dtype=bool)
        d[s] = 0.0
        queue = deque([int(s)])
        while queue:
            x = queue.popleft()
            if expanded[x]:
                continue
            expanded[x] = True
            for a in range(n_act):
                y = int(succ[x, a])
                if y < 0:
                    continue
                if d[x] + 1 < d[y]:
                    d[y] = d[x] + 1
                if not term[x, a] and not expanded[y]:
                    queue.append(y)
        dist[s] = d

    _dist_cache[key] = dist
    return dist


# -----------------------------------------------------------------------------
# ----- Metrics ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def goal_curves(
    dist,
    counter,
    mem,
    warmup,
    n_points,
    gamma,
    score_mode,
    per_step_selection,
    goal_value_check,
):
    """Two per-bin percentages of the goal SELECTIONS, and their totals.

    unreachable  the goal tile could not be reached from the tile the agent was on
    perfect      the goal maximised `score_mode` among the tiles in reach

    Both scores divide by `n`, how often a tile appears in `obs` strictly before the
    step -- the true count these runs select on, as it stood then rather than at the
    end -- and differ in the numerator:

        novelty  1 / n, so the best goal is simply a least-visited tile in reach
        ratio    the actor's own `V / n` (src/actor.py), where V is discounted
                 reachability: a goal `d` steps away is worth `gamma ** d`

    Whether V is `gamma ** d` or `gamma ** (d - 1)` makes no difference -- a
    constant factor across every candidate cannot move the argmax.

    The best score is taken over tiles visited more than CANDIDATE_MIN_COUNT times,
    since goal candidates are drawn from the replay memory and a tile the agent has
    not put there cannot be picked. The goal itself is scored whatever its own
    count, so selecting a barely-seen tile still gets credit. `n` is floored at
    COUNT_FLOOR the way the actor floors it. A goal is perfect when it ties the best
    score available, since tiles often share it and any of them is an equally good
    pick; an unreachable goal never is.

    Both metrics share the same denominator (every goal selection), so the two
    read against each other. A choice counts once however long the agent then
    acts on it, so an actor that picks a goal per episode and one that picks per
    step are both measured on the decisions they made. `per_step_selection` and
    `goal_value_check` are the actor's own flags and decide which steps those
    are. Bins with no selection stay NaN, leaving a gap rather than reading as
    0%."""

    reach = np.isfinite(dist)
    obs = mem.get("obs")
    goal_obs = mem.get("goal_obs")
    goal_valid = mem.get("goal_valid")
    if obs is None or goal_obs is None or goal_valid is None:
        return None
    # Where an episode ended, so the step after it is a fresh selection. A dump
    # without them is read as one long episode, which only costs the selections
    # that repeat a tile across a reset.
    done = None
    if mem.get("term") is not None and mem.get("trunc") is not None:
        done = (
            np.asarray(mem["term"]).reshape(len(obs)).astype(bool)
            | np.asarray(mem["trunc"]).reshape(len(obs)).astype(bool)
        )
    size = len(obs)
    if warmup >= size:
        return None

    state = np.asarray(counter.bin_index(obs)).reshape(size)
    goal = np.asarray(counter.bin_index(goal_obs)).reshape(size)
    carried = np.asarray(goal_valid).reshape(size, -1).astype(bool).any(axis=-1)

    # The steps that are a selection, which is what these rates are over: a
    # choice counts once, however long the agent then acts on it.
    #
    # Which steps those are follows the actor's own two flags (see
    # Actor.draw_action in src/actor.py). `per_step_selection` without
    # `goal_value_check` re-selects at every step, so every carrying step is a
    # choice whether or not it lands on the same tile again. With the check, and
    # without per-step selection, the actor keeps its goal until something drops
    # it: the choices are the first step of an episode and every step the goal
    # changes under it. A step whose predecessor carried no goal is a choice
    # either way, and so is the first step of the dump.
    if per_step_selection and not goal_value_check:
        active = carried
    else:
        starts = np.zeros(size, dtype=bool)
        starts[0] = True
        if done is not None:
            # A step after a terminated or truncated one begins an episode, and
            # the goal is picked again there even where it is the same tile.
            starts[1:] |= done[:-1]
        changed = np.ones(size, dtype=bool)
        changed[1:] = (goal[1:] != goal[:-1]) | ~carried[:-1]
        active = carried & (starts | changed)

    unreachable = np.zeros(size, dtype=bool)
    idx = np.flatnonzero(active)
    unreachable[idx] = ~reach[state[idx], goal[idx]]

    # Which tiles are within reach depends only on which room the agent is in, so
    # `reach` has a handful of distinct rows. Grouping them turns the per-step
    # maximum into one lookup over one index array instead of a mask over all tiles.
    rows, row_of = np.unique(reach, axis=0, return_inverse=True)
    members = [np.flatnonzero(r) for r in rows]

    # The numerator of the score, per (state, goal). Out of reach is 0 either way --
    # gamma ** inf is 0 -- so such a tile scores 0 and can never win.
    if score_mode == "ratio":
        with np.errstate(over="ignore"):
            value = np.power(gamma, dist)
    else:
        value = reach.astype(np.float64)

    perfect = np.zeros(size, dtype=bool)
    counts = np.zeros(reach.shape[0], dtype=np.float64)
    for t in range(size):
        s = int(state[t])
        if active[t]:
            g = int(goal[t])
            within = members[row_of[s]]
            within = within[counts[within] > CANDIDATE_MIN_COUNT]
            if reach[s, g] and within.size:
                scores = value[s, within] / np.maximum(counts[within], COUNT_FLOOR)
                best = scores.max()
                score_g = value[s, g] / max(counts[g], COUNT_FLOOR)
                perfect[t] = score_g >= best * (1.0 - 1e-12)
        counts[s] += 1.0

    edges = np.linspace(warmup, size, n_points + 1).astype(int)
    flags = {"unreachable": unreachable, "perfect": perfect}
    curves = {}
    for name, flag in flags.items():
        y = np.full(n_points, np.nan)
        for i in range(n_points):
            lo, hi = edges[i], edges[i + 1]
            n_sel = int(active[lo:hi].sum())
            if n_sel > 0:
                y[i] = 100.0 * flag[lo:hi].sum() / n_sel
        curves[name] = y
    totals = {name: int(flag[warmup:].sum()) for name, flag in flags.items()}
    return curves, totals, int(active[warmup:].sum()), size


# -----------------------------------------------------------------------------
# ----- Per-seed cache --------------------------------------------------------
# -----------------------------------------------------------------------------
#
# One dump reduces to two short curves and three integers, while producing them
# costs a multi-hundred-MB read and a pass over every step. Nothing about that
# depends on how the figure is styled, so a re-run that only changes the plot
# config, the legend or --auc_only can reuse it. The key covers the dump's identity
# and every parameter that changes the numbers, so a changed --goal_score or a
# re-synced dump misses rather than returning something stale.

cache_dir = None
if not args.no_cache:
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir
        else Path(args.folder) / CACHE_DIRNAME
    )
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"WARNING: cannot use cache directory {cache_dir}: {e}")
        cache_dir = None


def cache_path(
    mem_file,
    env_cfg,
    warmup,
    n_points,
    per_step_selection,
    goal_value_check,
):
    """Where one dump's reduced curves are kept, named by everything that decides
    what they hold."""

    st = mem_file.stat()
    payload = json.dumps({
        "file": str(mem_file.resolve()),
        "mtime_ns": st.st_mtime_ns,
        "size": st.st_size,
        "warmup": warmup,
        "points": n_points,
        "per_step_selection": bool(per_step_selection),
        "goal_value_check": bool(goal_value_check),
        # What a rate is over. Bumped when that changes, so curves reduced under
        # the old definition are missed rather than read back.
        "denominator": "selections",
        "gamma": args.gamma,
        "score": args.goal_score,
        "min_count": CANDIDATE_MIN_COUNT,
        "floor": COUNT_FLOOR,
        "env": env_cfg,
    }, sort_keys=True, default=str)
    return cache_dir / f"{hashlib.sha1(payload.encode()).hexdigest()}.npz"


def cache_read(path):
    """What was cached for one dump, or None when nothing usable is there."""

    try:
        with np.load(path) as d:
            return ({m: d[m] for m in METRICS},
                    {m: int(d[f"total_{m}"]) for m in METRICS},
                    int(d["selected"]), int(d["size"]))
    except Exception:
        return None


def cache_write(path, result):
    """Keep one dump's curves. Written beside the target and renamed, so a run
    interrupted mid-write cannot leave a half-file that later reads as a hit."""

    curves, totals, n_sel, size = result
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(
        tmp,
        selected=n_sel,
        size=size,
        **{m: curves[m] for m in METRICS},
        **{f"total_{m}": totals[m] for m in METRICS},
    )
    os.replace(tmp, path)


def curves_for_file(task):
    """One dump's curves, from the cache when it is there. Returns (result, hit)."""

    mem_file, env_cfg, dist, counter, warmup, n_points, per_step, value_check = task
    path = (
        cache_path(mem_file, env_cfg, warmup, n_points, per_step, value_check)
        if cache_dir else None
    )
    if path is not None:
        hit = cache_read(path)
        if hit is not None:
            return hit, True
    try:
        with np.load(mem_file) as data:
            mem = {
                k: data[k] for k in data.files
                if k in ("obs", "goal_obs", "goal_valid", "term", "trunc")
            }
    except Exception as e:
        vprint(f"  WARNING: skipping unreadable {mem_file}: {e}")
        return None, False
    result = goal_curves(
        dist,
        counter,
        mem,
        warmup,
        n_points,
        args.gamma,
        args.goal_score,
        per_step,
        value_check,
    )
    if result is None:
        vprint(f"  WARNING: {mem_file} stores no goals, skipping")
        return None, False
    if path is not None:
        try:
            cache_write(path, result)
        except OSError as e:
            vprint(f"  WARNING: could not cache {mem_file}: {e}")
    return result, False


# -----------------------------------------------------------------------------
# ----- The runs --------------------------------------------------------------
# -----------------------------------------------------------------------------

plot_cfgs = load_plot_configs(args.plot_config)
vprint(f"\n{len(plot_cfgs)} plot config(s): {[n for n, _, _ in plot_cfgs]}")

root = Path(args.folder)
if not root.is_dir():
    raise SystemExit(f"Not a directory: {root}")

# A run is a <data_dir>/<config_id>/<rng_seed> with a memory.npz in it, and every
# seed of a configuration is read: the curves are an average over them.
run_dirs = sorted(p for p in root.glob("*/*") if (p / "memory.npz").is_file())
if not run_dirs:
    raise SystemExit(
        f"No <config_id>/<rng_seed>/memory.npz under {root}. memory.npz is only "
        f"written when results.save_memory=True."
    )

flat_cfgs: dict = {}   # "<config_id>/<seed>" -> flat cfg
run_cfgs: dict = {}    # "<config_id>/<seed>" -> nested cfg
mem_files: dict = {}   # "<config_id>/<seed>" -> path to memory.npz
no_cfg_dirs: list = []

for seed_dir in tqdm(
    run_dirs, desc="Reading configs", unit="run", disable=not args.verbose
):
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
        f"any of them. -f takes a data directory."
    )

df = pd.DataFrame.from_dict(flat_cfgs, orient="index")
df.columns = pd.MultiIndex.from_tuples([("hyperparameter", c) for c in df.columns])

# Neither the environment nor the algorithm is recorded by a run: both are matched
# back from their config YAMLs, the file a run's hyperparameters agree with.
env_configs = load_config_group(args.environments) if args.environments else {}
algo_configs = load_config_group(args.algorithms) if args.algorithms else {}

# The answers `assign_groups` has already given, by the exemptions that produced
# them: configs declaring the same ones -- usually every one of them -- get the
# same answer, and the frame is compared against every YAML to arrive at it.
_assigned: dict = {}


ensure_font()
sns.set_context("paper")
sns.set_style("darkgrid", {"legend.frameon": True})
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.axisbelow"] = False
plt.rcParams["grid.linestyle"] = "--"


def runs_of(env, entry, ignored, matched):
    """Every run of one environment that an entry selects, one per seed: the ones
    whose whole recorded configuration is the one the entry composes to there.

    The comparison covers every key, the ones the entry leaves out included: a
    run launched with another value for one of those is a different
    configuration, and belongs to whichever entry composes to it.

    `ignored` and `matched` are the pair `assign_groups` returned for this plot
    config, and have to be that pair: a mask built under one set of exemptions
    says nothing about a comparison made under another."""

    # The shared reading, not one of its own: a configuration then means the same
    # runs here as in every other figure, and it stays that way when it changes.
    flat = entry_config(env, entry["selector"], entry["label"])
    if flat is None:
        return []
    # Over the whole frame, as plot_results.py selects: the environment is
    # composed into `flat`, so its own keys already keep the runs of another one
    # out, and the two scripts read one configuration the same way.
    #
    # The same exemption assign_configs was given above: a key a launch set for
    # every run alike is not part of what tells two configurations apart, and
    # holding a run to the YAML's value for it would match nothing.
    return list(df.index[composed_mask(df, flat, ignore=ignored) & matched])


# -----------------------------------------------------------------------------
# ----- What the plot configs ask for -----------------------------------------
# -----------------------------------------------------------------------------
#
# Settled before any memory is read, and read once for all of them: a dump costs a
# decompression and a pass over every step whether one figure uses it or five.

# config name -> (entries, {group: envs}, {(entry index, env): [runs]})
layout: dict = {}
for config_name, _path, cfg in plot_cfgs:
    entries = style_entries(cfg)
    if not entries:
        print(f"WARNING: {config_name} declares no configurations to plot, skipping.")
        continue
    # This config's own exemptions, not every config's: exempting a key changes
    # which YAML a run matches, and one config's `ignored_cfg_keys` says nothing
    # about the runs another one draws.
    ignored_keys = ignored_cfg_keys(cfg)
    if ignored_keys:
        vprint(f"{config_name} exempts from config matching: {sorted(ignored_keys)}")
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
        algo_decides_usable=False,
        cache=_assigned,
    )
    # Without the "unknown" stem, so a config that declares no environment group
    # of its own does not get one holding every run that matched no YAML.
    envs_found = sorted(set(env_stem[matched]), key=str)
    groups = env_groups(cfg) or {"": envs_found}
    selected = {}
    for group_envs in groups.values():
        for env in group_envs:
            if env not in envs_found:
                continue
            for i, entry in enumerate(entries):
                names = runs_of(env, entry, ignored_keys, matched)
                if names:
                    selected[(i, env)] = names
    layout[config_name] = (entries, groups, selected)

needed = sorted({
    n for _e, _g, sel in layout.values() for ns in sel.values() for n in ns
})
if not needed:
    raise SystemExit("No run matches any configuration of the plot config(s).")
vprint(f"{len(needed)} run(s) to read, of {len(df)} found")

# --- Everything a dump needs before it can be reduced ---
tasks: list = []          # (name, task tuple for curves_for_file)
x_span: dict = {}         # name -> (warmup, size), filled once the dump is read
warmup_of: dict = {}      # name -> warmup
no_counter: list = []     # runs whose environment has no transition graph to close

for name in needed:
    run_cfg = run_cfgs[name]
    env_cfg = run_cfg.get("environment") or {}
    dist = build_distances(env_cfg)
    counter = env_context(env_cfg)[0] if dist is not None else None
    if dist is None or counter is None:
        no_counter.append(name)
        continue

    experiment = run_cfg.get("experiment") or {}
    replay = experiment.get("replay_memory") or {}
    warmup = int(replay.get("min_size") or 0)
    n_points = args.points or int(experiment.get("testing_points") or 100)
    warmup_of[name] = warmup
    # How the actor was configured to choose, which decides what counts as a
    # selection: every carrying step, or the first of an episode and every
    # reselection after it.
    actor = ((run_cfg.get("agent") or {}).get("actor") or {})
    per_step = bool(actor.get("per_step_selection"))
    value_check = bool(actor.get("goal_value_check"))
    tasks.append((
        name,
        (
            mem_files[name],
            env_cfg,
            dist,
            counter,
            warmup,
            n_points,
            per_step,
            value_check,
        ),
    ))

if not tasks:
    raise SystemExit(
        "No run has a tabular state space with a set_state interface, so no "
        "transition graph can be closed over and nothing can be drawn."
    )

# --- Read the dumps ---
# Threads, not processes: the pass over the steps holds the GIL, but the read and
# the array work that dominate a cold run release it, and a warm cache turns the
# whole pass into small reads. Processes would need this script wrapped in a main()
# guard, since spawn re-imports it.
n_jobs = args.jobs if args.jobs > 0 else min(8, (os.cpu_count() or 1))
n_jobs = max(1, min(n_jobs, len(tasks)))

curves_of: dict = {}   # name -> {metric: curve}
totals_of: dict = {}   # name -> {metric: count, "selected": count}
cache_hits = 0


def keep(name, result):
    """File one dump's reduction under the run it came from."""

    curves, totals, n_sel, size = result
    curves_of[name] = curves
    totals_of[name] = dict(totals, selected=n_sel)
    x_span[name] = (warmup_of[name], size)


with tqdm(
    total=len(tasks),
    desc="Reading memories",
    unit="dump",
    disable=not args.verbose,
) as pbar:
    if n_jobs == 1:
        for name, task in tasks:
            result, hit = curves_for_file(task)
            cache_hits += hit
            if result is not None:
                keep(name, result)
            pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = {
                pool.submit(curves_for_file, task): name for name, task in tasks
            }
            for future in as_completed(futures):
                name = futures[future]
                result, hit = future.result()
                cache_hits += hit
                if result is not None:
                    keep(name, result)
                pbar.update(1)

print(f"Runs found                 : {len(run_dirs)}")
print(
    f"Memory dumps               : {len(tasks)}"
    f"  ({cache_hits} from cache, {len(tasks) - cache_hits} read,"
    f" {n_jobs} at a time)"
)
if no_counter:
    print(
        f"Skipped                    : {len(no_counter)}"
        f"  (no tabular state space / no set_state interface)"
    )
if not curves_of:
    raise SystemExit("No data to plot.")


# -----------------------------------------------------------------------------
# ----- Figures ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def seed_curves(names, metric):
    """The runs of one (entry, environment) stacked one per row, and the span they
    cover, or None when none of them was reduced."""

    rows = [curves_of[n][metric] for n in names if n in curves_of]
    if not rows:
        return None
    spans = [x_span[n] for n in names if n in curves_of]
    return np.vstack(rows), (min(w for w, _ in spans), max(s for _, s in spans))


def pooled(selected, envs, metric, index):
    """One configuration's rate averaged over the environments of a group, as
    (mean, interval).

    The two are averaged separately: the interval is what the seeds of an
    environment disagreed by, averaged over environments, and not the spread
    between environments -- those differ in difficulty, not in what the
    configuration did."""

    values = []
    for env in envs:
        names = selected.get((index, env))
        if not names:
            continue
        # nanmean warns on a curve whose every bin is empty, which is a run that
        # selected no goal at all. Scoped here rather than filtered for the
        # process: every other RuntimeWarning this run raises is worth seeing.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            per_seed = [
                float(np.nanmean(curves_of[n][metric]))
                for n in names if n in curves_of
            ]
        value = summarize(per_seed)
        if value is not None and np.isfinite(value[0]):
            values.append(value)
    if not values:
        return None
    return (
        float(np.mean([m for m, _ in values])),
        float(np.mean([e for _, e in values])),
    )


def hbar_positions(drawn):
    """Centre of each --auc_only bar, top to bottom, and the slot one bar takes: its
    thickness plus the gap to the next, with GROUP_GAP_BAR added between groups."""

    slot = AUC_ONLY_BAR_THICKNESS + AUC_ONLY_BAR_GAP
    ys = [0.0]
    for prev, cur in zip(drawn, drawn[1:]):
        gap = slot + (GROUP_GAP_BAR if cur["group"] != prev["group"] else 0.0)
        ys.append(ys[-1] + gap)
    return ys, slot


def draw_hbars(ax, items):
    """draw_bars on its side: one horizontal bar per configuration, stacked top to
    bottom in the order given.

    Same group gaps, colours, hatches and error bars as draw_bars, on the tighter
    slot hbar_positions lays out. Every bar stands from one floor below the lowest
    interval, and the axis starts at that floor.

    Each bar is named on the y axis in unrotated text, and its value is written
    past the end of its error bar. The value axis carries no tick labels."""

    drawn = [entry for entry, _ in items]
    ys, slot = hbar_positions(drawn)
    lows = [m - e for _, v in items if v is not None for m, e in [v] if np.isfinite(m)]
    floor = min(lows) - 0.1 * abs(min(lows)) if lows else 0.0

    ends = []
    for y, (entry, value) in zip(ys, items):
        if value is None:
            continue
        mean, err = value
        ax.barh(
            y,
            mean - floor,
            left=floor,
            height=AUC_ONLY_BAR_THICKNESS,
            xerr=err,
            color=entry["color"],
            hatch=entry["hatch"],
            edgecolor="black",
            linewidth=0.6,
            capsize=2,
            error_kw={"linewidth": 0.8, "ecolor": "black"},
        )
        if np.isfinite(mean):
            ends.append((y, mean + (err if np.isfinite(err) else 0.0), mean))

    if ys:
        # Inverted, so the first configuration is on top.
        ax.set_ylim(ys[-1] + slot / 2, ys[0] - slot / 2)
    ax.set_yticks(ys)
    ax.set_yticklabels([e["label"] for e in drawn], fontsize=AUC_ONLY_LABEL_SIZE)
    for text in ax.get_yticklabels():
        text.set(**tex_kwargs(text.get_text()))
    ax.tick_params(axis="y", length=0, pad=2)
    # darkgrid would otherwise draw a line through every bar at its tick.
    ax.yaxis.grid(False)
    ax.set_xticklabels([])

    means = [v[0] for _, v in items if v is not None and np.isfinite(v[0])]
    if means:
        # Left at the floor the bars stand from, so none of them starts away from
        # the axis; the right keeps its pad, which the value labels sit in.
        lo, hi = min(means), max(means)
        ax.set_xlim(floor, hi + 0.1 * (hi - lo))
    if ends:
        xlo, xhi = ax.get_xlim()
        ax.set_xlim(xlo, xhi + 0.08 * (xhi - xlo))
        offset = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        prec = detect_precision([m for _, _, m in ends], min_prec=2, max_prec=6)
        for y, end, mean in ends:
            ax.text(
                end + offset,
                y,
                f"{mean:.{prec}f}",
                ha="left",
                va="center",
                fontsize=FONT_SIZE - 4,
                clip_on=False,
            )


def pad_ylim(ax, items):
    """Pad the top of the axis above the tallest bar and put three ticks on it.

    The bottom stays at the floor draw_bars set, which already sits below the
    lowest interval. The top is only raised, so it keeps the headroom draw_bars
    made for an annotation.
    """

    means = [v[0] for _, v in items if v is not None and np.isfinite(v[0])]
    if means:
        lo, hi = min(means), max(means)
        ax.set_ylim(top=max(ax.get_ylim()[1], hi + 0.1 * (hi - lo)))
    set_3_ticks(ax, which="y")


def legend_on(fig, cfg, handles, offset=None, top=None):
    """Name the configurations drawn, once for the whole figure and below it.

    `--no_legend` leaves it off; the standalone legend figure is written either way,
    for a figure that has to carry one elsewhere. `top` pins the legend's top edge to
    that figure-fraction height instead of dropping it by `offset`, for a figure
    whose labels hang down by an amount a fixed drop cannot know."""

    if args.no_legend or not handles:
        return
    n_cols = rows_cols(cfg.get("legend_rows_cols"), len(handles))[1]
    if top is None:
        rows = -(-len(handles) // n_cols)
        loc, anchor = "lower center", (0.5, offset - 0.14 * (rows - 1))
    else:
        loc, anchor = "upper center", (0.5, top)
    legend = fig.legend(
        handles,
        [h.get_label() for h in handles],
        fontsize=FONT_SIZE - 2,
        frameon=True,
        loc=loc,
        ncol=n_cols,
        bbox_to_anchor=anchor,
    )
    for text in legend.get_texts():
        text.set(**tex_kwargs(text.get_text()))


def write_legend(cfg, entries, output_dir):
    """The legend on its own, for a figure that has to carry one elsewhere.

    Both shapes are written every time, since which one fits depends on the document
    the figure goes into rather than on the data. The horizontal is laid out by
    `legend_rows_cols`; the vertical is always one entry per row."""

    handles = [mpatches.Patch(facecolor=e["color"], label=e["label"]) for e in entries]
    if not handles:
        return
    n_cols = rows_cols(cfg.get("legend_rows_cols"), len(handles))[1]

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
            handlelength=2.6,
            handleheight=1.3,
        )
        for text in legend.get_texts():
            text.set(**tex_kwargs(text.get_text()))
        # The canvas is cut to the legend rather than guessed at.
        fig_legend.canvas.draw()
        box = legend.get_window_extent().transformed(
            fig_legend.dpi_scale_trans.inverted())
        fig_legend.set_size_inches(box.width, box.height)
        vprint(f"  Saved: {save_figure(fig_legend, output_dir, name)}")
        plt.close(fig_legend)


def draw_curves(
    config_name,
    cfg,
    entries,
    selected,
    group_name,
    envs,
    metric,
    output_dir,
):
    """One metric over one environment group: a subplot per environment, a curve per
    configuration, seeds averaged into the mean and its confidence band.

    The two metrics get a figure each rather than two rows of one, so either can be
    placed on its own."""

    ylabel = METRIC_LABELS[metric]
    present = [e for e in envs if any((i, e) in selected for i in range(len(entries)))]
    if not present:
        return 0
    drawn = sorted({i for i, e in selected if e in present})

    n_cells = len(present) + (1 if args.with_auc else 0)
    n_rows, n_cols = fitted_rows_cols(
        config_name,
        "plots_rows_cols",
        n_cells,
        cfg.get("plots_rows_cols"),
    )
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(SUBPLOT_W * n_cols, SUBPLOT_H * n_rows),
        squeeze=False,
    )
    fig.subplots_adjust(wspace=0.3)

    shown: list = []
    for k, env in enumerate(present):
        ax = axs[k // n_cols][k % n_cols]
        title = env_labels.get(str(env), str(env))
        ax.set_title(title, fontsize=FONT_SIZE, **tex_kwargs(title))

        x_lo, x_hi = float("inf"), 0.0
        for i in drawn:
            names = selected.get((i, env))
            if not names:
                continue
            stacked = seed_curves(names, metric)
            if stacked is None:
                continue
            data, (warmup, size) = stacked
            entry = entries[i]
            error_shade_plot(
                ax, data,
                stepsize=(size - warmup) / data.shape[1],
                smoothing_window=args.smoothing_window,
                color=entry["color"], linestyle=entry["linestyle"],
                x_start=warmup, linewidth=2.0,
            )
            if i not in shown:
                shown.append(i)
            x_lo = min(x_lo, warmup)
            x_hi = max(x_hi, size)

        ax.tick_params(axis="x", labelsize=FONT_SIZE - 2, pad=-2)
        ax.tick_params(axis="y", labelsize=FONT_SIZE - 2, pad=1)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(3, 3))
        ax.xaxis.offsetText.set_visible(False)
        ax.set_ylabel(
            ylabel if k % n_cols == 0 else "",
            fontsize=FONT_SIZE,
            **tex_kwargs(ylabel),
        )
        # set_3_ticks assumes linear spacing, and a log axis has no room for the
        # bins that sit at exactly 0% -- matplotlib drops them either way.
        if args.log_scale:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
            set_3_ticks(ax, which="y")
        if x_hi > 0:
            ax.set_xlim(x_lo, x_hi)
            ax.set_xticks([x_lo, (x_lo + x_hi) / 2, x_hi])
            ticks = ax.xaxis.get_major_ticks()
            if len(ticks) >= 1:
                ticks[0].label1.set_horizontalalignment("left")
            if len(ticks) >= 3:
                ticks[-1].label1.set_horizontalalignment("right")

    if args.with_auc:
        k = len(present)
        ax = axs[k // n_cols][k % n_cols]
        ax.set_title("Avg. Area-Under-Curve", fontsize=FONT_SIZE)
        items = [(entries[i], pooled(selected, present, metric, i)) for i in drawn]
        draw_bars(
            ax,
            items,
            ylabel="",
            bar_width=args.bar_width,
            font_size=FONT_SIZE,
            annotate=args.annotate_auc,
        )
        pad_ylim(ax, items)

    # The cells of the arrangement that nothing landed in.
    for k in range(n_cells, n_rows * n_cols):
        axs[k // n_cols][k % n_cols].axis("off")

    legend_on(
        fig,
        cfg,
        [
            plt.Line2D(
                [], [],
                color=entries[i]["color"],
                linestyle=entries[i]["linestyle"],
                linewidth=2.0,
                label=entries[i]["label"],
            )
            for i in shown
        ],
        offset=-0.15,
    )

    name = METRIC_FILES[metric] + (f"_{group_name}" if group_name else "") + "_curves"
    vprint(f"  Saved: {save_figure(fig, output_dir, name)}")
    plt.close(fig)
    return 1


def draw_auc(cfg, entries, selected, group_name, envs, metric, output_dir):
    """One metric's AUC bars on their own, with no curves beside them.

    Horizontal bars, one per configuration top to bottom, each named in unrotated
    text beside it. One panel of AUC_ONLY_WIDTH, its value scale drawn inside, and
    the title naming the quantity."""

    present = [e for e in envs if any((i, e) in selected for i in range(len(entries)))]
    if not present:
        return 0
    drawn = sorted({i for i, e in selected if e in present})
    items = [(entries[i], pooled(selected, present, metric, i)) for i in drawn]
    if not any(v is not None for _, v in items):
        return 0

    ys, slot = hbar_positions([entry for entry, _ in items])
    fig, axs = plt.subplots(
        1, 1,
        figsize=(AUC_ONLY_WIDTH, (ys[-1] - ys[0] + slot) * AUC_ONLY_UNIT_INCHES),
        squeeze=False,
    )
    ax = axs[0][0]
    title = METRIC_AUC_TITLES[metric]
    ax.set_title(title, fontsize=FONT_SIZE, **tex_kwargs(title))
    draw_hbars(ax, items)

    # The legend goes under where the panel and its labels actually end, measured
    # rather than a fixed drop.
    fig.canvas.draw()
    bottom = ax.get_tightbbox(fig.canvas.get_renderer()).y0
    top = fig.transFigure.inverted().transform((0, bottom))[1] - 0.02
    legend_on(
        fig,
        cfg,
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
        top=top,
    )

    name = METRIC_FILES[metric] + (f"_{group_name}" if group_name else "") + "_auc"
    vprint(f"  Saved: {save_figure(fig, output_dir, name)}")
    plt.close(fig)
    return 1


def draw_auc_all(cfg, entries, selected, envs, metric, output_dir):
    """One metric's AUC bars over every environment at once, drawn as draw_auc
    draws a group's.

    The figures above are one per environment group. This pools every environment
    the plot config names, group or no group, into one number per configuration.
    Written whatever --auc_only says."""

    present = [e for e in envs if any((i, e) in selected for i in range(len(entries)))]
    if not present:
        return 0
    drawn = sorted({i for i, e in selected if e in present})
    items = [(entries[i], pooled(selected, present, metric, i)) for i in drawn]
    if not any(v is not None for _, v in items):
        return 0

    ys, slot = hbar_positions([entry for entry, _ in items])
    fig, axs = plt.subplots(
        1, 1,
        figsize=(AUC_ONLY_WIDTH, (ys[-1] - ys[0] + slot) * AUC_ONLY_UNIT_INCHES),
        squeeze=False,
    )
    ax = axs[0][0]
    title = METRIC_AUC_TITLES[metric]
    ax.set_title(title, fontsize=FONT_SIZE, **tex_kwargs(title))
    draw_hbars(ax, items)

    name = METRIC_FILES[metric] + "_auc_all_envs"
    vprint(f"  Saved: {save_figure(fig, output_dir, name)}")
    plt.close(fig)
    return 1


def recap(config_name, entries, selected, envs):
    """What share of the goals each configuration selected were unreachable, and how
    many of them were the best pick available, over the whole post-warm-up memory."""

    rows = []
    for env in envs:
        for i, entry in enumerate(entries):
            names = [n for n in selected.get((i, env), []) if n in totals_of]
            if not names:
                continue
            n_sel = sum(totals_of[n]["selected"] for n in names)
            if not n_sel:
                continue
            rows.append((
                env_labels.get(str(env), str(env)),
                entry["label"],
                n_sel,
                {
                    m: 100.0 * sum(totals_of[n][m] for n in names) / n_sel
                    for m in METRICS
                },
                len(names),
            ))
    if not rows:
        return
    env_w = max(len(r[0]) for r in rows)
    label_w = max(len(r[1]) for r in rows)
    print(f"\nGoal selections over the whole post-warm-up memory -- {config_name}")
    for env, label, n_sel, pct, n_seeds in rows:
        print(
            f"  {env:<{env_w}}  {label:<{label_w}}  {n_sel:>9d} goals"
            f"  unreachable {pct['unreachable']:6.2f}%"
            f"  perfect {pct['perfect']:6.2f}%"
            f"  ({n_seeds} seed(s))"
        )


written = 0
for config_name, _path, cfg in plot_cfgs:
    if config_name not in layout:
        continue
    vprint(f"\n{'#' * 60}\nPlot config: {config_name}\n{'#' * 60}")
    entries, groups, selected = layout[config_name]
    # Only the runs that actually reduced: an environment with no transition graph
    # drops out here rather than leaving an empty subplot.
    selected = {k: [n for n in v if n in curves_of] for k, v in selected.items()}
    selected = {k: v for k, v in selected.items() if v}
    if not selected:
        print(
            f"WARNING: nothing {config_name} asks for has a reducible dump, "
            f"skipping."
        )
        continue

    _stat_labels, env_labels, _ylims = plot_labels(cfg)
    output_dir = os.path.join(args.folder, args.output, "goals", config_name)
    os.makedirs(output_dir, exist_ok=True)

    for group_name, group_envs in groups.items():
        vprint(
            f"\n{'=' * 60}\n"
            f"Environment group: {group_name or '(unnamed)'}\n"
            f"{'=' * 60}"
        )
        absent = [
            e for e in group_envs
            if not any((i, e) in selected for i in range(len(entries)))
        ]
        if absent:
            print(
                f"WARNING: environments in {config_name} with nothing to draw: "
                f"{absent}"
            )
        for metric in METRICS:
            if args.auc_only:
                written += draw_auc(
                    cfg, entries, selected, group_name, group_envs, metric,
                    output_dir,
                )
            else:
                written += draw_curves(
                    config_name,
                    cfg,
                    entries,
                    selected,
                    group_name,
                    group_envs,
                    metric,
                    output_dir,
                )

    # One figure per metric over every environment the config names, after the
    # per-group ones: the groups split the figures, and this is the number that
    # does not belong to any one of them.
    all_envs = [e for envs in groups.values() for e in envs]
    for metric in METRICS:
        written += draw_auc_all(cfg, entries, selected, all_envs, metric, output_dir)

    write_legend(cfg, entries, output_dir)
    recap(config_name, entries, selected, all_envs)
    print(
        f"\n{config_name}: {len(entries)} configuration(s), "
        f"{len(groups)} environment group(s) -> {output_dir}"
    )

print(
    f"\n{len(layout)} plot config(s), {written} figure(s) -> "
    f"{os.path.join(args.folder, args.output, 'goals')}"
)
