"""Visitation and value heatmaps, from a live agent or from a saved memory.

Both sources end up as the same thing -- a 2D map over the environment's tabular
bins -- so everything after that point is shared: the counter that owns the
binning, the orientation, the colour scale and the drawing.

    heatmaps_from_agent(actor, critic, env, savepath)   # during a run
    heatmaps_from_memory(memory, savepath, env=env)     # from memory.npz

Both draw a figure and save it. The scripts that lay many runs out on one grid
want the arrays instead, and call `agent_maps` / `memory_maps` directly.

The counter decides everything about shape: `_map_shape` is the grid a map is
reshaped to, and the counter's own binning turns an observation into a flat
index, so nothing here needs to know how an environment encodes its states.
Non-gridworld maps are transposed and flipped, since their coordinates are
Cartesian while an image's are not.
"""

import contextlib
import json
import sys
import warnings

import matplotlib
# No GUI, safe for background threads -- the training path draws from one. Chosen
# here only where nothing has chosen yet: importing pyplot is what settles the
# backend, so a caller that has already imported it keeps the one it picked, and
# importing this module does not change how that caller draws.
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Zero is drawn as "never seen" rather than as the bottom of the colour scale,
# and NaN renders black, so an empty bin is visibly empty.
EMPTY = np.nan


def viridis_black_bad():
    """Return a fresh viridis colormap with the bad-value colour set to black, so
    NaN cells (empty bins) render solid black rather than transparent."""

    cm = plt.get_cmap("viridis").copy()
    cm.set_bad(color="black")
    return cm


CMAP = viridis_black_bad()


def add_border(ax, lw=1.5):
    """Ensure all four axes spines are visible, black, and `lw` thick."""

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(lw)


# ----- Counters ---------------------------------------------------------------
# The counter maps an observation to a bin, and it is the only source of the
# grid's shape. It comes either from a live env, or from an env config a saved
# run was launched with.

def counter_of(env):
    """Return the environment's tabular counter, or None when it has none (a
    continuous environment with no binning, say)."""

    from src.pseudocount import tabular_count

    try:
        return tabular_count(env)
    except Exception:
        return None


def is_cartesian(env):
    """Return whether the environment's coordinates are Cartesian, i.e. y grows
    upward and a map has to be flipped to be drawn as an image. Gridworlds already
    index like an image and are left alone."""

    spec = getattr(getattr(env, "unwrapped", env), "spec", None)
    return "Gym-Gridworlds" not in (str(spec.id) if spec is not None else "")


_env_ctx_cache: dict = {}


def env_context(env_cfg: dict):
    """Create the environment a saved run was launched with and return its
    (counter, cartesian, reachable bins), caching the result.

    `env_cfg` is the `environment` section of that run's config, as written to
    cfg.yaml, and is what the environment is rebuilt from. Creation is often
    0.5-3s and dominates when many runs share one environment, so identical
    configs are only built once. The counter is None when the environment has no
    tabular counter, and the reachable bins are None when it keeps no account of
    them."""

    from src.wrappers import gym_wrappers

    key = json.dumps(env_cfg, sort_keys=True, default=str)
    if key in _env_ctx_cache:
        return _env_ctx_cache[key]
    env = gym_wrappers.make_gym_env(**env_cfg)
    try:
        ctx = (counter_of(env), is_cartesian(env), grid_reachable(env))
    finally:
        env.close()
    _env_ctx_cache[key] = ctx
    return ctx


def map_shape(counter):
    """Return the grid a flat map reshapes to: a gridworld's `(H, W)`, a binned
    counter's `tuple(n_bins)`, or a flat `(n,)` when it has neither."""

    if counter is None:
        return None
    shape = getattr(counter, "_map_shape", None)
    if shape is not None:
        return tuple(shape)
    shape = getattr(counter, "shape", None)
    return (int(np.prod(shape[:-1])),) if shape is not None else None


def scale_key(key):
    """Return the colour-scale family a map belongs to.

    Maps of one family are pooled into a single (vmin, vmax), so the figure of one
    is read against the colours of the other -- a pseudocount against the visit
    count it approximates. A key with no sibling is a family of its own, and two
    families never share a range: quantities measuring unrelated things pooled
    together flatten whichever is smaller into a single colour."""

    if key in ("visit_count", "pseudocount"):
        return "visit"
    if key in ("goal_selected_count", "goal_reached_count"):
        return "goal"
    return key


def orient(values, shape, cartesian, nan_mask=None, zero_empty=True):
    """Turn a map into an image and return it.

    `values` is one number per bin, flat or already 2D, and `shape` is the grid it
    belongs on. `cartesian` flips it (y grows upward in the environment, downward
    in an image).

    `nan_mask` is a boolean over the same bins, in the same layout as `values`
    before either is turned into an image: true keeps a bin, false sets it to
    NaN. It is ignored unless its size matches the map's.

    `zero_empty` turns zeros into NaN, which the colormap draws as empty. That is
    what a count map wants, a bin counted zero times being a bin never visited.
    A map whose values are not counts must pass False: zero is one of its values,
    and drawing it as empty reports it as a bin with no data."""

    m = np.asarray(values, dtype=float)
    if shape is not None and m.size == int(np.prod(shape)):
        m = m.reshape(shape)
    if nan_mask is not None:
        keep = np.asarray(nan_mask, dtype=bool)
        if keep.size == m.size:
            m = np.where(keep.reshape(m.shape), m, EMPTY)
    if cartesian and m.ndim == 2:
        m = np.flipud(m.T)
    return np.where(m == 0, EMPTY, m) if zero_empty else m


# ----- Maps -------------------------------------------------------------------

def grid_reachable(env):
    """Read off `env` which bins an agent can occupy and return them as a flat
    boolean array, or None where the environment keeps no such account."""

    grid = getattr(getattr(env, "unwrapped", env), "grid_reachable", None)
    if grid is None:
        return None
    return np.asarray(grid).astype(bool).ravel()


def agent_maps(actor, critic, env):
    """Build the maps of a live agent and return them as (map, title) pairs: the
    visit counts the critic keeps, the goal counts the actor keeps, and its
    value functions where they can be evaluated on the grid.

    The list is empty when the environment has no tabular counter."""

    visit_count = getattr(critic, "visit_count", None)
    counter = visit_count if visit_count is not None else counter_of(env)
    shape = map_shape(counter)
    if shape is None or len(shape) != 2:
        return []

    cartesian = is_cartesian(env)
    panels = []

    nan_mask = grid_reachable(env)

    if visit_count is not None:
        counts = orient(np.asarray(visit_count()).sum(-1), shape, cartesian)
        panels.append((counts, "Visits"))

    # Kept by goal-conditioned actors: which state-action pairs were picked as
    # goals, and which of those were reached.
    for attr, title in (
        ("goal_selected_count", "Goals Selected"),
        ("goal_reached_count", "Goals Reached"),
    ):
        goals = getattr(actor, attr, None)
        if goals is not None:
            panels.append(
                (orient(np.asarray(goals()).sum(-1), shape, cartesian), title)
            )

    values = value_map(critic, counter_of(env), env, nan_mask)
    if values is not None:
        panels.append((values, "Value"))

    visit_values = visit_value_map(critic, counter_of(env), env, nan_mask)
    if visit_values is not None:
        panels.append((visit_values, "Visit-Value"))

    return panels


def value_map(critic, counter, env, nan_mask=None):
    """Compute V(s) = max_a Q(s, a) over the grid and return it, or None when the
    critic cannot be evaluated on it.

    A tabular critic hands over its whole table; a network is queried at the bin
    centres, which only `counter` knows -- without them there is no grid to plot.
    `env` decides the orientation only.

    `nan_mask` is true for the bins to keep and false for the ones to set to
    NaN. See orient."""

    shape = map_shape(counter)
    if shape is None or len(shape) != 2:
        return None
    n_states = int(np.prod(shape))

    q = None
    if getattr(critic, "q", None) is not None and callable(critic.q):
        try:
            q = np.asarray(critic.q())
        except Exception:
            q = None

    if q is None and callable(critic):
        if not hasattr(counter, "bin_centers_raw"):
            return None
        states = counter.bin_centers_raw()
        if hasattr(critic, "eval"):
            critic.eval()
        try:
            q = np.asarray(critic(states))
        except Exception:
            return None
        finally:
            if hasattr(critic, "train"):
                critic.train()

    if q is None or q.size % n_states != 0:
        return None
    # A value is not a count: V(s) = 0 is a value the critic holds, and drawing
    # it as an empty bin would report the state as never visited. `nan_mask` is
    # what says which bins have no data here.
    return orient(
        q.max(-1).reshape(shape),
        shape,
        is_cartesian(env),
        nan_mask,
        zero_empty=False,
    )


def _flatten_and_split(x, shape):
    """Reshape a 4D grid (h, w, h, w) into a 2D visualization: rows index outer
    state s, cols index outer goal g, and each cell holds the (h, w) inner grid.
    Inserts NaN separator rows/cols between blocks for visual clarity."""

    a, b, c, d = shape
    y = x.reshape(shape).transpose(2, 0, 3, 1).reshape(c * a, d * b)
    row_sep = np.full((2, d * b), np.nan)
    row_blocks = np.split(y, c)
    y_row = np.vstack([
        block if i == c - 1 else np.vstack([block, row_sep])
        for i, block in enumerate(row_blocks)
    ])
    col_sep = np.full((y_row.shape[0], 2), np.nan)
    col_blocks = np.split(y_row, d, axis=1)
    y_sep = np.hstack([
        block if i == d - 1 else np.hstack([block, col_sep])
        for i, block in enumerate(col_blocks)
    ])
    return y_sep


def visit_value_map(critic, counter, env, nan_mask=None):
    """Compute V(s, g) = max_a Q_visit(s, a, g) over the grid and return it as a
    nested map -- one inner map over goals inside every outer cell over states --
    or None when the critic keeps no visit Q-function to evaluate on it.

    The two kinds of critic disagree on what a goal is, so they reduce
    differently:

      - a tabular critic hands over its whole table, of shape
        (n_states, n_act, n_states * n_act). Its goals are (state, action) pairs,
        so the goal's action is reduced away as well as the action taken.
      - a network cannot be read out at once, so it is queried at every pair of
        bin centres, which only `counter` knows. Its goals are states already
        (`critic(obs, goal=...)` slices out what it needs), so only the action
        taken is reduced away.

    `env` decides the orientation only.

    `nan_mask` is true for the bins to keep and false for the ones to set to
    NaN. A cell here is a (state, goal) pair, so it is kept only where both ends
    are. It is applied on the nested grid, before the orientation and the
    separators."""

    shape = map_shape(counter)
    if shape is None or len(shape) != 2:
        return None
    n_states = int(np.prod(shape))

    q_visit = getattr(critic, "q_visit", None)
    if q_visit is None:
        return None

    v = None
    if callable(q_visit):
        try:
            table = np.asarray(q_visit())
        except Exception:
            table = None
        if (
            table is not None
            and table.ndim == 3
            and table.shape[0] == n_states
            and table.shape[2] == n_states * table.shape[1]
        ):
            n_act = table.shape[1]
            v = table.max(1).reshape(n_states, n_states, n_act).max(-1)

    if v is None and callable(critic):
        if not hasattr(counter, "bin_centers_raw"):
            return None
        states = counter.bin_centers_raw()
        n, dim = states.shape
        if hasattr(critic, "eval"):
            critic.eval()
        try:
            q = np.asarray(
                critic(
                    obs=np.broadcast_to(states[:, None, :], (n, n, dim)),
                    goal=np.broadcast_to(states[None, :, :], (n, n, dim)),
                )
            )
        except Exception:
            return None
        finally:
            if hasattr(critic, "train"):
                critic.train()
        # GCRL QTable dims are (s, a, s*a). QNetwork is (s, s, a, a).
        if q.ndim not in (3, 4) or q.shape[:2] != (n_states, n_states):
            return None
        v = q.max(-1)
        if q.ndim == 4:
            v = v.max(-1)

    if v is None:
        return None

    v = v.reshape(shape + shape)
    if nan_mask is not None:
        keep = np.asarray(nan_mask, dtype=bool)
        if keep.size == n_states:
            keep = keep.reshape(shape)
            v = np.where(
                keep[:, :, None, None] & keep[None, None, :, :],
                v,
                EMPTY,
            )
    if is_cartesian(env):
        # The flipud(m.T) orient() applies to a flat map, done here on the outer
        # state grid and on the inner goal grid: both are over the environment's
        # coordinates, so both need it.
        v = np.flip(v.transpose(1, 0, 3, 2), axis=(0, 2))
    return _flatten_and_split(v, v.shape)


def memory_maps(memory, env=None, env_cfg=None, fractions=(1.0,)):
    """Count what a replay memory holds and return {fraction: {map name: map}},
    in one pass.

    Each value in `fractions` is a prefix of the memory: 0.25 counts its first
    quarter, 1.0 all of it. The maps of one run are therefore nested -- each is
    the one before it plus the entries in between -- which is what makes a row of
    them read as the visitation building up over training.

    `memory` is anything with an "obs" array: a loaded memory.npz, or the live
    replay memory. What else it holds decides which maps come back: a "count"
    array adds "pseudocount"; "goal_obs" and "goal_valid" add
    "goal_selected_count"; "act" and "goal_act" beside them add
    "goal_reached_count".
    Pass either a live `env` or the `env_cfg` a saved run was launched with.
    Returns None when the environment has no tabular counter."""

    if env is not None:
        counter, cartesian = counter_of(env), is_cartesian(env)
    elif env_cfg is not None:
        counter, cartesian, _reachable = env_context(env_cfg)
    else:
        raise ValueError("memory_maps needs either env or env_cfg")
    if counter is None:
        return None

    shape = map_shape(counter)
    n_flat = int(np.prod(shape))
    obs = np.asarray(memory["obs"])
    size = len(obs)
    obs_flat = counter.bin_index(obs)

    entry_count = None
    if "count" in memory:
        entry_count = np.asarray(memory["count"]).sum(-1).astype(float)

    goal_flat = goal_valid = goal_reached = None
    if "goal_obs" in memory and "goal_valid" in memory:
        goal_valid = np.asarray(memory["goal_valid"]).astype(bool)
        goal_flat = counter.bin_index(np.asarray(memory["goal_obs"]))
        if "act" in memory and "goal_act" in memory:
            from src.pseudocount import is_neighbor_binned

            # The rule the actor counts a goal reached by, in one call: the step
            # is within a bin width of the goal and takes the action the goal
            # names. `is_neighbor` compares element-wise, so passing the two
            # (n, obs_dim) arrays tests each step against its own goal.
            goal_reached = is_neighbor_binned(
                np.asarray(memory["obs"]), np.asarray(memory["goal_obs"]), counter
            ) & (
                np.asarray(memory["act"]).ravel()
                == np.asarray(memory["goal_act"]).ravel()
            )

    # The prefixes are nested, so each one is the one before it plus the entries
    # in between. Walked in increasing order and accumulated, every entry is
    # counted once across all the fractions rather than once per fraction --
    # the same saving pseudocount_prefixes makes over calling pseudocount per
    # (radius, prefix). `np.maximum.at` is the slowest of the counts and gains
    # the most: it runs over each entry once instead of over every prefix.
    prefixes = [max(1, int(round(size * fraction))) for fraction in fractions]
    visit   = np.zeros(n_flat, dtype=np.int64)
    pseudo  = np.zeros(n_flat, dtype=float) if entry_count is not None else None
    picked  = np.zeros(n_flat, dtype=np.int64) if goal_flat is not None else None
    hit     = np.zeros(n_flat, dtype=np.int64) if goal_reached is not None else None

    maps = {}
    start = 0
    for i in sorted(range(len(fractions)), key=lambda j: prefixes[j]):
        end = prefixes[i]
        if end > start:
            visit += np.bincount(obs_flat[start:end], minlength=n_flat)
            if pseudo is not None:
                np.maximum.at(pseudo, obs_flat[start:end], entry_count[start:end])
            if picked is not None:
                valid = goal_valid[start:end]
                picked += np.bincount(
                    goal_flat[start:end][valid], minlength=n_flat
                )
                if hit is not None:
                    # Counted in the goal's bin, like the selections above: the
                    # two maps answer how often a bin was aimed at and how often
                    # it was hit, so they have to count in the same place.
                    hit += np.bincount(
                        goal_flat[start:end][valid & goal_reached[start:end]],
                        minlength=n_flat,
                    )
            start = end
        # orient() builds a new array, so the accumulators can go on growing.
        per_key = {"visit_count": orient(visit, shape, cartesian)}
        if pseudo is not None:
            per_key["pseudocount"] = orient(pseudo, shape, cartesian)
        if picked is not None:
            per_key["goal_selected_count"] = orient(picked, shape, cartesian)
        if hit is not None:
            per_key["goal_reached_count"] = orient(hit, shape, cartesian)
        maps[fractions[i]] = per_key
    # Back in the order they were asked for: the walk above is in prefix order,
    # and callers lay the fractions out in the order they passed them.
    return {fraction: maps[fraction] for fraction in fractions}


def compute_maps_all_fractions(run_cfg, mem, fractions):
    """Count what a saved run's replay memory holds and return
    {fraction: {map name: map}}: `memory_maps` keyed off the run's own config.

    `run_cfg` is a whole cfg.yaml, so the environment is rebuilt from its
    `environment` section rather than passed in live. `mem` is the loaded
    memory.npz. Returns None when that environment has no tabular counter."""

    return memory_maps(mem, env_cfg=run_cfg["environment"], fractions=fractions)


# ----- Colour scale -----------------------------------------------------------
# A map with nothing in it raises RuntimeWarnings that are ordinary here: a map
# no run visited is all NaN, and a nanmin or nanmax over it warns about an empty
# slice, while a log scale over a zero or a negative bound warns about the
# logarithm of it. All of them come back as NaN, which is drawn as the "no data"
# colour. Silenced inside the functions that raise them rather than at their
# callers: every script that draws a map goes through these, and a filter set for
# the process would also hide an overflow in code that has nothing to do with
# heatmaps.

@contextlib.contextmanager
def quiet_empty_maps():
    """Silence the RuntimeWarnings an empty map raises (see above)."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def arr_range(arr):
    """Return (vmin, vmax) from a single array, falling back to (0.0, 1.0) if it
    is unusable (None, empty, or non-finite bounds)."""

    # np.any(isfinite) before nanmin: an all-NaN map is a map whose every cell
    # is empty, which happens before anything has been counted, and nanmin warns
    # on it rather than just returning nan. That guard is why this function needs
    # no quiet_empty_maps of its own -- a full nanmin warns only where every
    # value is NaN, and it has already returned by then.
    if arr is None or arr.size == 0 or not np.any(np.isfinite(arr)):
        return 0.0, 1.0
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 0.0, 1.0
    return lo, hi


def pooled_range(arrays, default=(0.0, 1.0)):
    """Return (vmin, vmax) over several arrays, skipping the ones that are None or
    empty. `default` is returned when none of them is usable."""

    valid = [a for a in arrays if a is not None and a.size]
    if not valid:
        return default
    with quiet_empty_maps():
        lo = float(np.nanmin([np.nanmin(a) for a in valid]))
        hi = float(np.nanmax([np.nanmax(a) for a in valid]))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return default
    return lo, hi


def scale_map(arr, vmin, vmax, log_scale=False):
    """Compress a map and its bounds together and return (arr, vmin, vmax), so the
    colour scale keeps meaning what it says.

    log1p, not log: a bin visited once stays apart from one never visited, and an
    empty bin lands at the bottom of the scale rather than at minus infinity."""

    if log_scale:
        with quiet_empty_maps():
            return np.log1p(arr), float(np.log1p(vmin)), float(np.log1p(vmax))
    return arr, vmin, vmax


def auto_dpi(arrays, max_dpi=1000, pixels_per_cell=15, subplot_size=1.5):
    """Return a DPI that gives each cell of a heatmap about `pixels_per_cell`
    output pixels, capped at `max_dpi`.

    `arrays` are the maps that will be drawn, and the largest of them sets the
    scale; `subplot_size` is the width in inches one map is drawn at."""

    valid = [a for a in arrays if a is not None]
    if not valid:
        return max_dpi
    max_dim = max(max(a.shape) for a in valid)
    return int(np.clip(max_dim * pixels_per_cell / subplot_size, 72, max_dpi))


# ----- Colour bars ------------------------------------------------------------
# A panel of a grid is too small to hold a colour bar, so one is drawn outside
# the panels it applies to, in the gap under them. A bar belongs to exactly the
# panels that were drawn on one range and spans them, so which panels it
# describes is read off the figure rather than from a caption.

CBAR_SUBPLOT_SIZE = 2.2    # inches per panel, once a bar has to fit under it
CBAR_THICKNESS    = 0.07   # inches, the short side of a bar
CBAR_PAD          = 0.06   # inches between a panel and its bar
CBAR_HSPACE       = 0.22   # row gap, as a fraction of a panel, that a bar sits in


def scale_funcs(log_scale=False, exp_scale=None):
    """The transform the maps were drawn under and its inverse, as
    (forward, inverse): a colour bar is normalised in the units the colours came
    from and ticked in the units the map holds.

    `exp_scale` wins over `log_scale`, as it does where the maps are drawn.

    Both are quiet on a bound the transform has no answer for -- a log scale over
    a bound at or below -1 -- and return NaN for it, which is the reading
    add_colorbar draws nothing on."""

    def quiet(f):
        def transform(v):
            # Through numpy whatever it was handed, which is what makes the NaN
            # above true of every input. A bare Python float takes Python's own
            # power semantics, where a negative base under a fractional exponent
            # is a complex number rather than NaN -- and float() of a complex
            # raises, so the exp_scale inverse of a bound below -1 would come
            # back as a TypeError out of add_colorbar rather than as a bar it
            # declines to draw.
            with quiet_empty_maps():
                return f(np.asarray(v))

        return transform

    if exp_scale is not None:
        a = float(exp_scale)
        return (
            quiet(lambda v: (1.0 + v) ** a),
            quiet(lambda v: v ** (1.0 / a) - 1.0),
        )
    if log_scale:
        return quiet(np.log1p), quiet(np.expm1)
    return (lambda v: v), (lambda v: v)


def fmt_tick(v):
    """A colour-bar tick label: values of 100 or more are written without
    decimals, smaller ones to three significant figures."""

    v = float(v)
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}"


def cbar_groups(cell_keys, cell_ranges):
    """{(row, col): key} and {(row, col): (vmin, vmax)} -> [((vmin, vmax), cells)],
    one entry per set of cells that was drawn on a shared range -- that is, one
    entry per colour bar.

    A cell with no range never held a map, so it joins no set, and a key whose
    every cell was empty drops out rather than captioning blank panels."""

    groups: dict = {}
    for cell, key in cell_keys.items():
        if cell in cell_ranges:
            groups.setdefault(key, [cell_ranges[cell], []])[1].append(cell)
    return [(rng, cells) for rng, cells in groups.values()]


def cbar_hspace(cell_keys, nrows, hspace=CBAR_HSPACE, default=0.05):
    """The row gap a grid of `nrows` needs to hold its colour bars.

    A bar sits under the lowest row of the cells it covers. One covering a whole
    column ends below the grid and needs no room made for it; one that stops
    short -- a bar per panel, or per block of rows -- lands between two rows, and
    only then is a gap opened."""

    lowest: dict = {}
    for (r, _c), key in cell_keys.items():
        lowest[key] = max(lowest.get(key, -1), r)
    return hspace if any(r != nrows - 1 for r in lowest.values()) else default


def add_colorbar(
    fig,
    axes,
    vmin,
    vmax,
    cmap=CMAP,
    inverse=None,
    fontsize=7,
    n_ticks=3,
    thickness=CBAR_THICKNESS,
    pad=CBAR_PAD,
):
    """A horizontal colour bar under `axes`, spanning them, ticked in the units of
    the map rather than of the colours.

    `vmin`/`vmax` are the bounds the panels were drawn with, already transformed
    if they were, and `inverse` (see scale_funcs) turns a tick back into the value
    it stands for. Where the bar goes is taken from where the panels actually sit,
    and its thickness is set in inches, so it is the same bar whatever the shape
    of the grid.

    A flat or unusable range draws nothing: every cell came out the same colour,
    so there is no scale to read."""

    lo, hi = float(vmin), float(vmax)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None

    boxes = [ax.get_position() for ax in axes]
    x0, x1 = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    thick = thickness / fig.get_figheight()
    rect = [
        x0 + 0.05 * (x1 - x0),
        min(b.y0 for b in boxes) - pad / fig.get_figheight() - thick,
        0.9 * (x1 - x0),
        thick,
    ]

    cb = fig.colorbar(
        ScalarMappable(norm=Normalize(lo, hi), cmap=cmap),
        cax=fig.add_axes(rect),
        orientation="horizontal",
    )
    ticks = np.linspace(lo, hi, n_ticks)
    cb.set_ticks(ticks)
    cb.set_ticklabels(
        [fmt_tick(v) for v in (ticks if inverse is None else inverse(ticks))]
    )
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=fontsize, length=1.5, width=0.4, pad=1)
    return cb


# ----- Drawing ----------------------------------------------------------------

def plot_map(ax, arr, vmin=None, vmax=None, cmap=CMAP, log_scale=False):
    """Draw one map on `ax` as a bare grid cell: no ticks, bordered, black behind
    the empty bins.

    The counterpart of `draw_map` for a figure built out of many small panels. It
    uses `imshow` rather than a seaborn heatmap, since a cell a few pixels across
    fits no annotations or colour bar and a sweep draws hundreds of them.

    `arr` of None leaves an empty bordered panel, which is how a configuration
    with no run shows up. `vmin`/`vmax` default to the array's own range, so a
    lone map fills the scale; pass a pooled range to make several maps
    comparable. `log_scale` compresses the colour scale (see scale_map)."""

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("black")
    if arr is None:
        add_border(ax)
        return

    if vmin is None or vmax is None:
        auto_min, auto_max = arr_range(arr)
        vmin = auto_min if vmin is None else vmin
        vmax = auto_max if vmax is None else vmax
    scaled, vmin, vmax = scale_map(arr, vmin, vmax, log_scale)

    ax.imshow(
        scaled,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
    )
    add_border(ax)


def draw_map(
    ax,
    arr,
    vmin=None,
    vmax=None,
    annotate=False,
    log_scale=False,
    cmap=CMAP,
    cbar=False,
    fmt="{:.2f}",
    fontsize=None,
):
    """Draw one map on `ax`, ticks off and bordered.

    `arr` of None leaves an empty bordered panel, which is how a configuration
    with no run shows up. `vmin`/`vmax` default to the array's own range, so a
    lone map fills the scale; pass a pooled range to make several maps
    comparable. `log_scale` compresses the colour scale (see scale_map). With
    `annotate`, each cell's value is written in it, formatted
    with `fmt` at `fontsize` -- seaborn picks the text colour per cell, so it
    stays readable at both ends of the scale. `cbar` adds a horizontal colour
    bar."""

    ax.set_facecolor("black")
    if arr is None:
        ax.set_xticks([])
        ax.set_yticks([])
        add_border(ax)
        return

    if vmin is None or vmax is None:
        auto_min, auto_max = arr_range(arr)
        vmin = auto_min if vmin is None else vmin
        vmax = auto_max if vmax is None else vmax
    scaled, vmin, vmax = scale_map(arr, vmin, vmax, log_scale)

    # The annotations show the map's own values, not the compressed ones: what is
    # drawn is a colour, what is read is a count.
    labels = False
    if annotate:
        labels = np.where(
            np.isnan(arr), "", np.vectorize(fmt.format)(np.nan_to_num(arr))
        )

    sns.heatmap(
        scaled,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        cbar=cbar,
        cbar_kws={"orientation": "horizontal", "pad": 0.08} if cbar else None,
        annot=labels,
        fmt="",
        annot_kws={"fontsize": fontsize} if fontsize else None,
        xticklabels=False,
        yticklabels=False,
    )
    add_border(ax)


def draw_panels(
    panels,
    subplot_size=4.0,
    annotate=False,
    log_scale=False,
    shared_range=False,
    cbar=False,
    ranges=None,
):
    """Draw the given (map, title) pairs side by side and return the figure, each
    panel as wide as its map is and `subplot_size` inches tall.

    `shared_range` pools vmin/vmax over the panels, which is what you want when
    they measure the same thing and not when they do not -- counts and values on
    one scale say nothing.

    `ranges` gives one (vmin, vmax) per panel instead, for a figure whose panels
    fall into groups measuring different things: which panels pool together is
    then the caller's to decide. It takes precedence over `shared_range`.

    `annotate`, `log_scale` and `cbar` are passed to each panel (see draw_map).
    Returns None when there are no panels."""

    if not panels:
        return None

    widths = [m.shape[1] / float(m.shape[0]) for m, _ in panels]
    fig, axs = plt.subplots(
        1, len(panels),
        figsize=(
            sum(w * subplot_size for w in widths) + 0.5 * (len(panels) - 1),
            subplot_size,
        ),
        gridspec_kw={"width_ratios": widths},
        squeeze=False,
    )

    fontsize = float(np.clip(12.0 / (len(panels) ** 0.5), 6, 14))
    bounds = pooled_range([m for m, _ in panels]) if shared_range else None
    per_panel = list(ranges) if ranges is not None else [bounds] * len(panels)

    for ax, (m, title), bound in zip(axs[0], panels, per_panel):
        vmin, vmax = bound if bound is not None else (None, None)
        finite = m[~np.isnan(m)]
        draw_map(
            ax,
            m,
            vmin,
            vmax,
            annotate,
            log_scale,
            cbar=cbar,
            # Counts are whole numbers and read better without a decimal point.
            fmt="{:.0f}" if finite.size and np.allclose(finite % 1, 0) else "{:.2f}",
            fontsize=fontsize,
        )
        ax.set_title(title, fontsize=fontsize * 1.3)

    fig.subplots_adjust(wspace=0.15)
    return fig


def save_panels(
    fig,
    savepath,
    name,
    arrays=(),
    dpi=None,
    max_dpi=1000,
    pixels_per_cell=15,
):
    """Save the figure to `savepath/name.png`, close it and return the path.

    Without a `dpi`, one is chosen from `arrays` -- the maps that were drawn -- so
    that each of their cells gets about `pixels_per_cell` pixels, capped at
    `max_dpi`. When the file cannot be written the figure is still closed, a
    warning is raised and None is returned instead of the path."""

    import os

    os.makedirs(savepath, exist_ok=True)
    path = os.path.join(savepath, f"{name}.png")
    if dpi is None:
        dpi = auto_dpi(list(arrays), max_dpi, pixels_per_cell)
    try:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
    except OSError as err:
        warnings.warn(f"could not write {path}: {err}", RuntimeWarning)
        path = None
    finally:
        plt.close(fig)
    return path


# ----- The two entry points ---------------------------------------------------

def heatmaps_from_agent(
    actor,
    critic,
    env,
    savepath,
    tot_steps=-1,
    annotate=False,
    log_scale=False,
    cbar=False,
    dpi=200,
):
    """Draw the visits and value of a live agent, save them as
    `savepath/<tot_steps>.png` and return the path.

    `tot_steps` is the training step the maps describe, and only names the file.
    `annotate` writes each cell's value in it, `log_scale` compresses the colour
    scale (see scale_map), `cbar` adds a colour bar and `dpi`
    is the resolution to save at. Returns None when the environment has no
    tabular counter and there is nothing to draw."""

    panels = agent_maps(actor, critic, env)
    fig = draw_panels(
        panels,
        annotate=annotate,
        log_scale=log_scale,
        cbar=cbar,
    )
    if fig is None:
        return None
    return save_panels(
        fig,
        savepath,
        str(tot_steps),
        [m for m, _ in panels],
        dpi=dpi,
    )


def heatmaps_from_memory(
    memory,
    savepath,
    env=None,
    env_cfg=None,
    tot_steps=-1,
    fractions=(1.0,),
    annotate=False,
    log_scale=False,
    cbar=False,
    dpi=None,
):
    """Draw the maps a replay memory holds, save them as
    `savepath/<tot_steps>.png` and return the path.

    Each value in `fractions` is a prefix of the memory: 0.25 counts its first
    quarter, 1.0 all of it. One panel is drawn per (map, fraction). The panels of
    one colour-scale family share a range across the maps in it and across the
    fractions, so a sequence of fractions reads as the visitation filling up
    rather than each prefix renormalizing to itself, and a pseudocount reads
    against the visit count it approximates; two families never share a range (see
    scale_key).

    Pass either a live `env` or the `env_cfg` a saved run was launched with.
    `tot_steps` names the file. `annotate`, `log_scale`, `cbar` and
    `dpi` are as in heatmaps_from_agent. Returns None when the environment has no
    tabular counter and there is nothing to draw."""

    maps = memory_maps(memory, env=env, env_cfg=env_cfg, fractions=fractions)
    if not maps:
        return None

    # One range per colour-scale family, pooled over every map in it and every
    # fraction it was counted at.
    per_family: dict = {}
    for per_key in maps.values():
        for key, arr in per_key.items():
            per_family.setdefault(scale_key(key), []).append(arr)
    range_of = {family: pooled_range(arrs) for family, arrs in per_family.items()}

    panels, ranges = [], []
    for fraction, per_key in maps.items():
        for key, arr in per_key.items():
            panels.append(
                (arr, key if len(fractions) == 1 else f"{key} {fraction:.0%}")
            )
            ranges.append(range_of[scale_key(key)])
    fig = draw_panels(
        panels,
        annotate=annotate,
        log_scale=log_scale,
        ranges=ranges,
        cbar=cbar,
    )
    if fig is None:
        return None
    return save_panels(
        fig,
        savepath,
        str(tot_steps),
        [m for m, _ in panels],
        dpi=dpi,
    )
