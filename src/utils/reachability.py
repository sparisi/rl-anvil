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
from src.utils.misc import random_argmax

# The actor clips candidate counts to this before dividing (src/actor.py), which is
# what keeps a never-visited tile finite rather than infinitely attractive.
COUNT_FLOOR = 1e-3

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
    """Return `(candidates, missing)`, where `candidates` maps every goal
    selection step of a replay-memory dump to the states of the goal candidates
    the actor scored there, and `missing` counts the selections whose goal state
    is not among them.

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
    A position p drawn at step t holds the last step before t stored there,
    p + `max_size` * ((t - 1 - p) // `max_size`). With `per_step_selection` and
    without `goal_value_check`, the goal held before the selection is a
    candidate too: the goal of the previous step, unless that step reached it,
    i.e. its state is the goal's and its action the goal action."""

    candidates = {}
    missing = 0
    size = len(state)
    tiebreak = np.zeros(1)
    episode = 0
    t = 0
    while t < size:
        seed_seq = np.random.SeedSequence([rng_seed, 0, episode])
        _env_seed, actor_seed, _critic_seed = (int(s) for s in seed_seq.generate_state(3))
        rng = np.random.default_rng(seed=actor_seed)
        if n_critics is not None:
            rng.integers(n_critics)
        start = t
        while t < size:
            rng.random()
            if act_source[t] == ACT_RANDOM:
                rng.integers(n_act)
            else:
                if selected[t]:
                    pos = rng.integers(min(t, max_size), size=batch_size)
                    cand = state[pos + max_size * ((t - 1 - pos) // max_size)]
                    held = (
                        per_step_selection
                        and not goal_value_check
                        and t > start
                        and has_goal[t - 1]
                        and not (state[t - 1] == goal[t - 1] and act[t - 1] == goal_act[t - 1])
                    )
                    if held:
                        cand = np.append(cand, goal[t - 1])
                    candidates[t] = cand
                    missing += not np.any(cand == goal[t])
                    random_argmax(tiebreak, rng, axis=-1)
                random_argmax(tiebreak, rng, axis=-1)
            t += 1
            if done[t - 1]:
                break
        episode += 1
    return candidates, missing


def perfect_selections(
    dist,
    n_act,
    state,
    goal,
    act,
    goal_act,
    candidates,
    score_mode,
    gamma,
):
    """Return a bool array over the steps of a replay-memory dump marking the
    goal selections whose goal maximised the score among the (candidate, action)
    pairs the actor scored.

    `dist` holds in `dist[s, g]` the fewest steps from state `s` to state `g`,
    `inf` when `g` cannot be reached, and `candidates` maps every goal selection
    step to the states of its candidates (see replay_candidates). Both scores
    divide by `n`, how often the pair appears in (`state`, `act`) strictly before
    the step, and differ in the numerator:

        novelty  1 / n, so the best goal is a least-tried action of a candidate
                 in reach
        ratio    the actor's own `V / n`, where V is discounted reachability: a
                 goal `d` steps away is worth `gamma ** d`

    A candidate out of reach scores 0. The goal is scored with the count of its
    own goal action, and `n` is floored at COUNT_FLOOR as the actor floors it. A
    goal is perfect when it ties the best score; an unreachable goal never is."""

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
    counts = np.zeros((reach.shape[0], n_act), dtype=np.float64)
    for t in range(size):
        s = int(state[t])
        cand = candidates.get(t)
        if cand is not None:
            g = int(goal[t])
            if reach[s, g]:
                scores = value[s, cand, None] / np.maximum(counts[cand], COUNT_FLOOR)
                best = scores.max()
                score_g = value[s, g] / max(counts[g, goal_act[t]], COUNT_FLOOR)
                perfect[t] = score_g >= best * (1.0 - 1e-12)
        counts[s, act[t]] += 1.0
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
