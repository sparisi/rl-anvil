"""Goal selections of a goal-conditioned actor read from a replay-memory dump:
the cause of each goal reselection under the true visit Q-function, the true
progress toward the goal replaced at the reselections attributed to
approximation error, and whether each goal unreachable from the state of its
last selection was replaced by a reselection.

The arrays over the steps of a dump are `state` and `goal`, the counter bins of
its `obs` and `goal_obs`, `start_idx`, the index of the starting position of the
next reset at every step (see src.oracle.TrueVisitQ), as stored or zeros when the
dump does not store it, and `act`, `goal_act`, `has_goal` and `act_source` as
stored, with `selected` marking the steps stored with `goal_selection` and
`done` the steps that ended an episode (`term` or `trunc`). The true visit
Q-function and its reachability are read at the state and the start_idx of a
step."""

import numpy as np

from src.actor import ACT_RANDOM, ORACLE_TOL

# Causes of a goal reselection, in the order reselection_causes tests them.
RESELECTION_CAUSES = (
    "unreachable_at_selection",
    "became_unreachable",
    "random_action",
    "suboptimal_action",
    "env_noise",
    "critic_error",
)


def reselection_causes(
    true_visit,
    state,
    start_idx,
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
            i_sel, i_now = int(start_idx[last_sel]), int(start_idx[t])
            if not reach[s_sel, i_sel, g]:
                causes[t] = 0
            elif not reach[s_now, i_now, g]:
                causes[t] = 1
            else:
                v = q[:, :, g, :, a_g].max(axis=-1)
                if v[s_now, i_now] >= v[s_sel, i_sel] - tol:
                    causes[t] = 5
                else:
                    ks = np.arange(last_sel, t)
                    loss = v[state[ks], start_idx[ks]] - (
                        true_visit.reward_miss
                        + true_visit.gamma * v[state[ks + 1], start_idx[ks + 1]]
                    )
                    faults = np.flatnonzero(loss > tol)
                    k = int(ks[faults[0]] if faults.size else ks[np.argmax(loss)])
                    if act_source[k] == ACT_RANDOM:
                        causes[t] = 2
                    elif q[state[k], start_idx[k], g, act[k], a_g] < v[state[k], start_idx[k]] - tol:
                        causes[t] = 3
                    else:
                        causes[t] = 4
        last_sel = t
    return causes


def unreachable_selections(true_visit, state, start_idx, goal, act, goal_act, has_goal, selected, done):
    """Return `(fates, held)`, an int array and a float array over the steps of a
    replay-memory dump, for every goal held that is not reachable from the state
    of its last selection, under the reachability of the true visit Q-function
    `true_visit` (src.oracle.TrueVisitQ).

    A goal is held from the selection that picks it while the previous step, in
    the same episode, held no goal, a different one, or reached it, through the
    selections that pick it again, until a selection picks another goal, the
    goal is reached, or the episode or the dump ends. Its last selection is the
    state reselection_causes tests for unreachable_at_selection. At the step of
    its first selection, `fates` holds 1 when a reselection (see
    reselection_causes) replaces it, and 0 otherwise; `held` holds the steps
    from its first selection to the reselection, and NaN when no reselection
    replaces it. Every other step holds -1 in `fates` and NaN in `held`."""

    reach = true_visit.reach
    size = len(state)
    fates = np.full(size, -1, dtype=np.int64)
    held = np.full(size, np.nan)
    first = last = -1  # first and last selection of the goal held, or -1
    for t in range(size):
        new_episode = t == 0 or done[t - 1]
        if new_episode and first >= 0:
            if not reach[int(state[last]), int(start_idx[last]), int(goal[last])]:
                fates[first] = 0
            first = last = -1
        if not selected[t]:
            continue
        p = t - 1
        held_before = not new_episode and first >= 0 and has_goal[p]
        reached = held_before and state[p] == goal[p] and act[p] == goal_act[p]
        same = held_before and goal[t] == goal[p] and goal_act[t] == goal_act[p]
        if same and not reached:
            last = t
            continue
        if first >= 0 and not reach[int(state[last]), int(start_idx[last]), int(goal[last])]:
            replaced = held_before and not reached
            fates[first] = int(replaced)
            if replaced:
                held[first] = t - first
        first = last = t
    if first >= 0 and not reach[int(state[last]), int(start_idx[last]), int(goal[last])]:
        fates[first] = 0
    return fates, held


def critic_error_progress(true_visit, state, start_idx, goal, goal_act, selected, causes):
    """Return a float array over the steps of a replay-memory dump holding, at
    every reselection `causes` (as reselection_causes returns them) attributes to
    critic_error, the steps of true progress toward the goal replaced since its
    selection, and NaN at every other step.

    The goal replaced, g with goal action a_g, was selected at the last goal
    selection before the reselection, at state s_sel, and is replaced at state
    s. With r_hit, r_miss, gamma and V*(x, g) = max_a Q*(x, a | g, a_g) of the
    true visit Q-function `true_visit` (src.oracle.TrueVisitQ), d(x) solves

        V*(x, g) = r_miss / (1 - gamma) + (r_hit - r_miss / (1 - gamma)) gamma^d(x)

    and the progress is d(s_sel) - d(s). Under deterministic dynamics d(x) is
    the number of steps from x to the hit of the goal, and the progress an
    integer."""

    q, gamma = true_visit.q, true_visit.gamma
    floor = true_visit.reward_miss / (1.0 - gamma)
    progress = np.full(len(state), np.nan)
    sel = np.flatnonzero(selected)
    for t in np.flatnonzero(causes == RESELECTION_CAUSES.index("critic_error")):
        t_sel = sel[np.searchsorted(sel, t) - 1]
        g, a_g = int(goal[t - 1]), int(goal_act[t - 1])
        v = q[state[[t_sel, t]], start_idx[[t_sel, t]], g, :, a_g].max(axis=1)
        d = np.log((v - floor) / (true_visit.reward_hit - floor)) / np.log(gamma)
        progress[t] = d[0] - d[1]
    return progress
