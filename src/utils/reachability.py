"""Goal selections of a goal-conditioned actor read from a replay-memory dump:
the goal candidates the actor scored at each selection, replayed from the seeds
of the run, the selections whose goal was the best pick among those candidates,
and the cause of each goal reselection under the true visit Q-function.

The arrays over the steps of a dump are `state` and `goal`, the counter bins of
its `obs` and `goal_obs`, and `act`, `goal_act`, `has_goal` and `act_source` as
stored, with `selected` marking the steps stored with `goal_selection` and
`done` the steps that ended an episode (`term` or `trunc`)."""

import numpy as np

from src.actor import ACT_RANDOM

# The actor clips candidate counts to this before dividing (src/actor.py), which is
# what keeps a never-visited tile finite rather than infinitely attractive.
COUNT_FLOOR = 1e-3

# Tolerance of the comparisons between values of the true visit Q-function
# (src.oracle.TrueVisitQ), passed to reselection_causes as `tol`.
ORACLE_TOL = 1e-9

# Entries of the (selection, candidate[, action]) arrays replay_candidates and
# perfect_selections compute at once, which bounds their memory.
BLOCK = 2_000_000

# Causes of a goal reselection, in the order reselection_causes tests them.
RESELECTION_CAUSES = (
    "unreachable_at_selection",
    "became_unreachable",
    "random_action",
    "suboptimal_action",
    "env_noise",
    "critic_error",
)


def replay_candidates(
    state,
    act,
    goal,
    goal_act,
    has_goal,
    act_source,
    selected,
    done,
    rng_seed,
    batch_size,
    n_act,
    n_critics,
    per_step_selection,
    goal_value_check,
    max_size,
):
    """Return `(sel_steps, cand, missing)`: the goal selection steps of a
    replay-memory dump, in order, the states of the goal candidates the actor
    scored at each of them, an int32 array with a row per selection, and the
    number of selections whose goal state is not among their candidates.

    The dump holds every step of the run in order, as Experiment.savedata
    merges it, and the replay memory held at most `max_size` steps, at position
    step % `max_size`.

    The candidates are drawn by `ReplayMemory.get` with the actor's random
    generator, and the actions that generator draws do not depend on any value
    the critic computes, so its draws are replayed here in the order the run made
    them. Experiment.train seeds it at every episode from
    `SeedSequence([rng_seed, 0, episode])`, and in an episode it draws:
    - at the reset, the ensemble member acting in the episode when the critic is
      an ensemble of `n_critics` members (`n_critics` None otherwise),
    - at every step, one uniform for the epsilon test, then
      - for a random action (`act_source` ACT_RANDOM), the action,
      - otherwise, at a goal selection `batch_size` memory positions below the
        memory size, min(step, `max_size`), and the tie-break of the goal, and
        then the tie-break of the action.
    A tie-break is `random_argmax` over a 1-D array, which draws one uniform
    through `random_choice`. A step with neither a random action nor a goal
    selection thus draws two uniforms, and a run of k such steps is replayed as
    one `rng.random(2 * k)`: the generator draws each uniform from one 64-bit
    output, as a run of single calls does, and leaves the 32-bit half that
    `integers` keeps buffered between its draws as it was. The loop visits only
    the episode starts, the random actions and the goal selections.

    A position p drawn at step t holds the last step before t stored there,
    p + `max_size` * ((t - 1 - p) // `max_size`). With `per_step_selection` and
    without `goal_value_check`, the goal held before the selection is a
    candidate too, as the last column: the goal of the previous step in the
    episode, unless that step reached it, i.e. its state is the goal's and its
    action the goal action; a selection without one holds -1 there."""

    size = len(state)
    sel_steps = np.flatnonzero(selected)
    events = np.flatnonzero(selected | (act_source == ACT_RANDOM))
    ends = np.flatnonzero(done) + 1
    starts = np.concatenate(([0], ends[ends < size]))
    episode_of = np.searchsorted(starts, events, side="right") - 1

    draws = np.empty((len(sel_steps), batch_size), dtype=np.int32)
    i_sel, episode, cursor, rng = 0, -1, 0, None
    for t, ep in zip(events.tolist(), episode_of.tolist()):
        if ep != episode:
            episode = ep
            seed_seq = np.random.SeedSequence([rng_seed, 0, episode])
            _env_seed, actor_seed, _critic_seed = (int(s) for s in seed_seq.generate_state(3))
            rng = np.random.default_rng(seed=actor_seed)
            if n_critics is not None:
                rng.integers(n_critics)
            cursor = int(starts[episode])
        if t > cursor:
            rng.random(2 * (t - cursor))
        rng.random()
        if selected[t]:
            draws[i_sel] = rng.integers(min(t, max_size), size=batch_size)
            i_sel += 1
            rng.random(2)
        else:
            rng.integers(n_act)
        cursor = t + 1

    # Positions to states in place, BLOCK entries at a time.
    rows = max(1, BLOCK // batch_size)
    for lo in range(0, len(sel_steps), rows):
        t = sel_steps[lo:lo + rows, None]
        pos = draws[lo:lo + rows]
        draws[lo:lo + rows] = state[pos + max_size * ((t - 1 - pos) // max_size)]
    cand = draws
    if per_step_selection and not goal_value_check:
        p = np.maximum(sel_steps - 1, 0)
        first = sel_steps == starts[np.searchsorted(starts, sel_steps, side="right") - 1]
        reached = (state[p] == goal[p]) & (act[p] == goal_act[p])
        valid = ~first & has_goal[p] & ~reached
        cand = np.concatenate((cand, np.where(valid, goal[p], -1).astype(np.int32)[:, None]), axis=1)
    missing = int((~(cand == goal[sel_steps][:, None]).any(axis=1)).sum())
    return sel_steps, cand, missing


def perfect_selections(
    dist,
    n_act,
    state,
    goal,
    act,
    goal_act,
    sel_steps,
    cand,
    score_mode,
    gamma,
):
    """Return a bool array over the steps of a replay-memory dump marking the
    goal selections whose goal maximised the score among the (candidate, action)
    pairs the actor scored.

    `dist` holds in `dist[s, g]` the fewest steps from state `s` to state `g`,
    `inf` when `g` cannot be reached, and `sel_steps` and `cand` are the goal
    selection steps and the states of their candidates, as replay_candidates
    returns them (-1 for no candidate). Both scores divide by `n`, how often the
    pair appears in (`state`, `act`) strictly before the step, and differ in the
    numerator:

        novelty  1 / n, so the best goal is a least-tried action of a candidate
                 in reach
        ratio    the actor's own `V / n`, where V is discounted reachability: a
                 goal `d` steps away is worth `gamma ** d`

    A candidate out of reach scores 0. The goal is scored with the count of its
    own goal action, and `n` is floored at COUNT_FLOOR as the actor floors it. A
    goal is perfect when it ties the best score; an unreachable goal never is.

    Every count is looked up at once: with every step keyed by its pair, pair *
    size + step, and the keys sorted, the count of pair q before step t is the
    number of keys in [q * size, q * size + t), found by binary search. The
    scores are computed for BLOCK (selection, candidate, action) entries at a
    time."""

    reach = np.isfinite(dist)
    # The numerator of the score, per (state, goal). Out of reach is 0 either way --
    # gamma ** inf is 0 -- so such a tile scores 0 and can never win.
    if score_mode == "ratio":
        with np.errstate(over="ignore"):
            value = np.power(gamma, dist)
    else:
        value = reach.astype(np.float64)

    size = len(state)
    perfect = np.zeros(size, dtype=bool)
    if len(sel_steps) == 0:
        return perfect

    n_pairs = reach.shape[0] * n_act
    keys = np.sort((state.astype(np.int64) * n_act + act) * size + np.arange(size))
    first = np.searchsorted(keys, np.arange(n_pairs, dtype=np.int64) * size)

    def count_before(pair, step):
        return np.searchsorted(keys, pair * size + step) - first[pair]

    s = state[sel_steps].astype(np.int64)
    g = goal[sel_steps].astype(np.int64)
    score_g = value[s, g] / np.maximum(count_before(g * n_act + goal_act[sel_steps], sel_steps), COUNT_FLOOR)

    best = np.empty(len(sel_steps))
    actions = np.arange(n_act)
    rows = max(1, BLOCK // (cand.shape[1] * n_act))
    for lo in range(0, len(sel_steps), rows):
        hi = lo + rows
        valid = cand[lo:hi] >= 0
        c = np.where(valid, cand[lo:hi], 0).astype(np.int64)
        n = count_before(c[:, :, None] * n_act + actions, sel_steps[lo:hi, None, None])
        scores = value[s[lo:hi, None], c][:, :, None] / np.maximum(n, COUNT_FLOOR)
        best[lo:hi] = np.where(valid[:, :, None], scores, -np.inf).max(axis=(1, 2))

    perfect[sel_steps] = reach[s, g] & (score_g >= best * (1.0 - 1e-12))
    return perfect


def reselection_causes(
    true_visit,
    state,
    goal,
    act,
    goal_act,
    has_goal,
    act_source,
    selected,
    done,
    tol,
):
    """Return an int array over the steps of a replay-memory dump holding, at
    every goal reselection, the index in RESELECTION_CAUSES of its cause, and -1
    at every other step.

    A reselection is a goal selection whose previous step, in the same episode,
    held a goal it did not reach and that selects a different goal; a step
    reaches its goal when its state is the goal's and its action the goal
    action. The goal replaced, g with goal action a_g, was selected at the last
    goal selection before the reselection, at state s_sel, and is replaced at
    state s. With the true visit Q-function `true_visit` (src.oracle.TrueVisitQ),
    its reachability and V*(x, g) = max_a Q*(x, a | g, a_g), the cause is the
    first that holds of:
    - unreachable_at_selection: g is not reachable from s_sel,
    - became_unreachable: g is not reachable from s,
    - random_action, suboptimal_action, env_noise: V*(s, g) < V*(s_sel, g) - tol,
      attributed to the first step k since the selection that lost true value,
      r_miss + gamma * V*(s_{k+1}, g) < V*(s_k, g) - tol, or to the step that
      lost the most when no step loses more than `tol`: a random action
      (`act_source` ACT_RANDOM), an action not optimal under Q*, or else the
      transition,
    - critic_error: the true value did not drop."""

    q, reach = true_visit.q, true_visit.reach
    size = len(state)
    causes = np.full(size, -1, dtype=np.int64)
    last_sel = -1
    for t in range(size):
        new_episode = t == 0 or done[t - 1]
        if new_episode:
            last_sel = -1
        if not selected[t]:
            continue
        p = t - 1
        replaced = (
            not new_episode
            and last_sel >= 0
            and has_goal[p]
            and not (state[p] == goal[p] and act[p] == goal_act[p])
            and (goal[t] != goal[p] or goal_act[t] != goal_act[p])
        )
        if replaced:
            g, a_g = int(goal[p]), int(goal_act[p])
            s_sel, s_now = int(state[last_sel]), int(state[t])
            if not reach[s_sel, g]:
                causes[t] = 0
            elif not reach[s_now, g]:
                causes[t] = 1
            else:
                v = q[:, g, :, a_g].max(axis=1)
                if v[s_now] >= v[s_sel] - tol:
                    causes[t] = 5
                else:
                    ks = np.arange(last_sel, t)
                    loss = v[state[ks]] - (true_visit.reward_miss + true_visit.gamma * v[state[ks + 1]])
                    faults = np.flatnonzero(loss > tol)
                    k = int(ks[faults[0]] if faults.size else ks[np.argmax(loss)])
                    if act_source[k] == ACT_RANDOM:
                        causes[t] = 2
                    elif q[state[k], g, act[k], a_g] < v[state[k]] - tol:
                        causes[t] = 3
                    else:
                        causes[t] = 4
        last_sel = t
    return causes
