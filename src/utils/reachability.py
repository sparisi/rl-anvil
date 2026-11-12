"""Shortest distances between the states of a tabular environment, read off its
own transition graph, and the goal selections of a replay-memory dump whose
goal could not be reached from the state it was selected at or was the best
pick among the goals in reach."""

import json
from collections import deque

import numpy as np

# The actor clips candidate counts to this before dividing (src/actor.py), which is
# what keeps a never-visited tile finite rather than infinitely attractive.
COUNT_FLOOR = 1e-3

# A tile has to have been visited more than this to count as an available goal: the
# actor draws its candidates from the replay memory, so a tile that never reached
# it was never on offer and does not belong in the best-available score.
CANDIDATE_MIN_COUNT = 1

_dist_cache: dict = {}


def build_distances(env_cfg):
    """Return `(dist, n_act)`, where `dist` is an (n_states, n_states) float
    matrix holding in `dist[s, g]` the fewest steps from state `s` to state `g`,
    `inf` when `g` cannot be got to from `s` at all, and `n_act` is the number of
    actions.

    The transition graph is read off the environment itself -- `set_state(s)`, step
    every action, record `get_state()` -- so one-way doorways, absorbing tiles and
    any other layout quirk are captured without hard-coding the geometry. Returns
    (None, None) for environments without a tabular `set_state` / `get_state` interface,
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
            _dist_cache[key] = (None, None)
            return None, None
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

    _dist_cache[key] = (dist, n_act)
    return dist, n_act


def unreachable_selections(dist, counter, mem):
    """Return `(state, goal, selected, unreachable)` over the steps of the
    replay-memory dump `mem`, or None when it stores no goals.

    `state` and `goal` are the counter bins of `obs` and `goal_obs`, `selected`
    marks the steps on which the actor selected a goal (`goal_selection`), and
    `unreachable` marks the selections whose goal tile cannot be reached from the
    tile the agent was on, by the distances `dist` of build_distances. Goals are
    compared on states only, since every action is available in every non-wall
    tile."""

    if any(mem.get(k) is None for k in ("obs", "goal_obs", "goal_selection")):
        return None
    size = len(mem["obs"])
    state = np.asarray(counter.bin_index(mem["obs"])).reshape(size)
    goal = np.asarray(counter.bin_index(mem["goal_obs"])).reshape(size)
    selected = np.asarray(mem["goal_selection"]).reshape(size).astype(bool)
    unreachable = np.zeros(size, dtype=bool)
    idx = np.flatnonzero(selected)
    unreachable[idx] = ~np.isfinite(dist[state[idx], goal[idx]])
    return state, goal, selected, unreachable


def perfect_selections(
    dist,
    n_act,
    state,
    goal,
    act,
    goal_act,
    selected,
    score_mode,
    gamma,
):
    """Return a bool array over the steps of a replay-memory dump marking the
    goal selections whose goal maximised the score among the (tile, action)
    pairs whose tile is in reach.

    `state`, `goal` and `selected` are as unreachable_selections returns them,
    and `act` and `goal_act` the actions and goal actions of the steps. Both
    scores divide by `n`, how often the pair appears in (`state`, `act`)
    strictly before the step, and differ in the numerator:

        novelty  1 / n, so the best goal is a least-tried action of a tile in reach
        ratio    the actor's own `V / n`, where V is discounted reachability: a
                 goal `d` steps away is worth `gamma ** d`

    The best score is taken over the tiles visited more than CANDIDATE_MIN_COUNT
    times, since goal candidates are drawn from the replay memory. The goal is
    scored with the count of its own goal action, and `n` is floored at
    COUNT_FLOOR as the actor floors it. A goal is perfect when it ties the best
    score; an unreachable goal never is."""

    reach = np.isfinite(dist)
    # Which tiles are within reach depends only on which room the agent is in, so
    # `reach` has a handful of distinct rows. Grouping them turns the per-step
    # maximum into one lookup over one index array instead of a mask over all tiles.
    rows, row_of = np.unique(reach, axis=0, return_inverse=True)
    row_of = np.asarray(row_of).reshape(-1)
    members = [np.flatnonzero(r) for r in rows]

    # The numerator of the score, per (state, goal). Out of reach is 0 either way --
    # gamma ** inf is 0 -- so such a tile scores 0 and can never win.
    if score_mode == "ratio":
        with np.errstate(over="ignore"):
            value = np.power(gamma, dist)
    else:
        value = reach.astype(np.float64)

    size = len(state)
    perfect = np.zeros(size, dtype=bool)
    counts = np.zeros((reach.shape[0], n_act), dtype=np.float64)
    for t in range(size):
        s = int(state[t])
        if selected[t]:
            g = int(goal[t])
            within = members[row_of[s]]
            within = within[counts[within].sum(1) > CANDIDATE_MIN_COUNT]
            if reach[s, g] and within.size:
                scores = value[s, within, None] / np.maximum(counts[within], COUNT_FLOOR)
                best = scores.max()
                score_g = value[s, g] / max(counts[g, goal_act[t]], COUNT_FLOOR)
                perfect[t] = score_g >= best * (1.0 - 1e-12)
        counts[s, act[t]] += 1.0
    return perfect
