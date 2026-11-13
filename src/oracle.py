"""
Exact visit Q-function of a Gym-Gridworlds environment, solved from the
environment's transition model. Goal-conditioned actors and QVisitNetwork use
it to compare their goal selection and their learned visit Q-function against
the true ones.
"""

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import StickyAction
from gym_gridworlds.gridworld import (
    ACTION_TO_VEC,
    GOOD,
    GOOD_SMALL,
    NOISY_TILE,
    PIT,
    QCKSND,
    STAY,
    WALL,
    Gridworld,
    _move,
)

from src.pseudocount import GridworldCount


class TrueVisitQ:
    """
    Hold the optimal visit Q-function Q*(s, a | g, a_g) of a base Gridworld,
    solved by value iteration with the Bellman operator QVisitNetwork learns:

        Q*(s, a | g, a_g) = r_hit                                             if s = g and a = a_g
        Q*(s, a | g, a_g) = r_miss + gamma * sum_s' C(s' | s, a) max_a' Q*(s', a' | g, a_g)   otherwise

    r_hit is 1 and r_miss is 0, both lowered by 1 with negative visit rewards.
    C(s' | s, a) is the probability that the step from s with action a lands in
    s' and does not terminate the episode. Truncation does not stop the
    bootstrap, as in the critic.

    The model reproduces Gridworld._step and Gridworld.step: random actions
    (random_action_prob and noisy tiles), quicksand, one-directional tiles,
    walls, pits, positive tiles terminating with STAY (or any action with
    no_stay), infinite_horizon, slippery_prob, action_to_terminate and
    random_reset_prob. Subclasses of Gridworld define their own dynamics and
    are rejected, as are random goals, which change the grid at every reset,
    and observation noise, which makes the observation differ from the state.

    A gymnasium StickyAction wrapper is modelled too. At every step outside a
    series of repeats, with probability p, it starts a series of d steps, d
    uniform in [d_min, d_max], that execute the last executed action in place
    of the agent's. The draw does not depend on the agent's action, so the
    Bellman operator becomes

        Q*(s, a | g, a_g) = r_miss + gamma * sum_s' C(s' | s, a) U(s', a)
        U(s, a) = (1 - p) V*(s) + p * mean_d F_d(s, a)
        F_0(s, a) = U(s, a)
        F_k(s, a) = r_hit                                                  if s = g
        F_k(s, a) = r_miss + gamma * sum_s' C(s' | s, a) F_{k-1}(s', a)    otherwise
        V*(s) = max_a Q*(s, a | g, a_g)

    with Q*(g, a_g | g, a_g) = r_hit as above. U(s, a) is the value of cell s
    after executing a, before the draw of the next step, and F_k(s, a) the
    value of cell s with k repeats of a left. A repeat in the goal's cell hits,
    since the visit reward is given for the agent's action. Q* and V* are the
    values of a cell when the agent's action is executed.
    """

    def __init__(
        self,
        env,
        counter: GridworldCount,
        gamma: float,
        negative_reward: bool,
        tol: float = 1e-10,
        max_iterations: int = 100_000,
    ):
        """
        Args:
            env (gymnasium.Env): the training environment, a base Gridworld
                observed through MatrixWrapper, with at most one StickyAction
                wrapper,
            counter (GridworldCount): one-hot counter of the environment, which
                maps observations to cells,
            gamma (float): discount factor of the visit Q-function,
            negative_reward (bool): if True, rewards are encoded as 0 on hit
                and -1 otherwise, as with `negative_visit_reward`,
            tol (float): value iteration stops when no value changes by more
                than this,
            max_iterations (int): value iteration raises an error if it does
                not converge within this many iterations.
        """

        unwrapped = env.unwrapped
        if type(unwrapped) is not Gridworld:
            raise ValueError(
                "TrueVisitQ supports only the base Gridworld class, got "
                f"{type(unwrapped).__name__}."
            )
        if unwrapped.random_goals:
            raise ValueError("TrueVisitQ does not support random_goals=True.")
        if unwrapped.observation_noise > 0.0:
            raise ValueError("TrueVisitQ does not support observation_noise > 0.")
        if not isinstance(counter, GridworldCount) or not counter._one_hot:
            raise ValueError("TrueVisitQ needs the one-hot GridworldCount of a MatrixWrapper observation.")

        self._counter = counter
        self.gamma = gamma
        self.reward_hit = 0.0 if negative_reward else 1.0
        self.reward_miss = -1.0 if negative_reward else 0.0
        self.sticky_prob, self.sticky_durations = self._sticky(env)
        cont = self._transitions(unwrapped)
        self.q = self._solve(cont, tol, max_iterations)
        self.reach = self._reachability(cont)

    def __call__(self, obs, goal):
        """
        Return Q*(obs, a | goal, a_g) with shape (..., n_actions, n_actions):
        agent actions on the second-to-last axis and goal actions on the last,
        as the critic returns them. `obs` and `goal` are observations with the
        same batch shape.
        """

        return self.q[self._counter.bin_index(obs), self._counter.bin_index(goal)]

    def reachable(self, obs, goal):
        """
        Return whether the cell of `goal` can be reached from the cell of `obs`
        with the batch shape of `obs` and `goal`. A reachable goal can be hit
        with any goal action, since the hit is the goal action taken in the
        goal's cell.
        """

        return self.reach[self._counter.bin_index(obs), self._counter.bin_index(goal)]

    @staticmethod
    def _sticky(env):
        """
        Return the repeat probability and the (min, max) repeat duration of the
        StickyAction wrapper of `env`, or (0.0, (1, 1)) without one.
        """

        found = []
        while isinstance(env, gym.Wrapper):
            if isinstance(env, StickyAction):
                found.append(env)
            env = env.env
        if not found:
            return 0.0, (1, 1)
        if len(found) > 1:
            raise ValueError("TrueVisitQ supports at most one StickyAction wrapper.")
        low, high = found[0].repeat_action_duration_range
        return float(found[0].repeat_action_probability), (int(low), int(high))

    @staticmethod
    def _start_distribution(env):
        """
        Return the probability of every cell at reset: uniform over the starting
        positions (with repetitions), or over the cells the agent can spawn in
        when there are none.
        """

        grid = env.original_grid
        start = np.zeros(grid.size)
        if env.start_pos is None:
            allowed = ~np.isin(grid, (WALL, PIT, GOOD, GOOD_SMALL))
            start[allowed.ravel()] = 1.0
        else:
            for pos in env.start_pos:
                start[np.ravel_multi_index(pos, grid.shape)] += 1.0
        return start / start.sum()

    @staticmethod
    def _executed(env, pos, act):
        """
        Return the outcomes of executing action `act` from cell `pos`, as a
        list of (probability, next cell, terminated).
        """

        grid = env.original_grid
        if env.action_to_terminate and act == env.terminate_action_id:
            return [(1.0, pos, True)]

        term = grid[pos] in (GOOD, GOOD_SMALL) and (act == STAY or env.no_stay)
        outcomes = []
        p_move = 1.0
        if grid[pos] == QCKSND:
            outcomes.append((0.9, pos, term))
            p_move = 0.1

        if grid[pos] in ACTION_TO_VEC and grid[pos] != act:
            first = pos
        else:
            first = _move(pos, act, grid)
        first_term = term
        if grid[first] == PIT:
            first_term = True
        elif grid[first] == WALL:
            first = pos

        if env.slippery_prob > 0.0:
            second = _move(first, act, grid)
            second_term = first_term
            if grid[second] == PIT:
                second_term = True
            elif grid[second] == WALL:
                second = first
            outcomes.append((p_move * env.slippery_prob, second, second_term))
        outcomes.append((p_move * (1.0 - env.slippery_prob), first, first_term))

        if env.infinite_horizon:
            outcomes = [(p, cell, False) for p, cell, _ in outcomes]
        return outcomes

    def _transitions(self, env):
        """
        Return C with shape (n_cells, n_actions, n_cells), where C[s, a, s'] is
        the probability that the step from s with action a lands in s' and does
        not terminate the episode.
        """

        grid = env.original_grid
        n_cells = grid.size
        n_actions = env.action_space.n
        start = self._start_distribution(env)
        reset_p = env.random_reset_prob
        cont = np.zeros((n_cells, n_actions, n_cells))
        for s in range(n_cells):
            pos = tuple(int(x) for x in np.unravel_index(s, grid.shape))
            p_random = env.random_action_prob
            if grid[pos] == NOISY_TILE:
                p_random += (1.0 - p_random) * 0.5
            for a in range(n_actions):
                executed = [(1.0 - p_random, a)] + [(p_random / n_actions, b) for b in range(n_actions)]
                for p_act, b in executed:
                    if p_act == 0.0:
                        continue
                    for p, cell, term in self._executed(env, pos, b):
                        if term:
                            continue
                        mass = p_act * p
                        cont[s, a, np.ravel_multi_index(cell, grid.shape)] += (1.0 - reset_p) * mass
                        cont[s, a] += reset_p * mass * start
        return cont

    def _solve(self, cont, tol, max_iterations):
        """
        Return Q* with shape (n_cells, n_cells, n_actions, n_actions), indexed
        as [s, g, a, a_g], by value iteration on the continuation probabilities
        `cont` for all goals at once. Every iteration updates Q* and U (see the
        class docstring) from the U of the previous one. Without a
        StickyAction wrapper, U(s, a) is V*(s).
        """

        n_cells, n_actions, _ = cont.shape
        n_goals = n_cells * n_actions  # one column per (g, a_g), with index g * n_actions + a_g
        goal = np.arange(n_goals)
        hit_cell = goal // n_actions
        hit_act = goal % n_actions
        by_action = np.ascontiguousarray(cont.transpose(1, 0, 2))
        p = self.sticky_prob
        low, high = self.sticky_durations

        def backup(y):
            # r_miss + gamma * sum_s' C(s' | s, a) y[s', a], for every (s, a).
            return self.reward_miss + self.gamma * np.matmul(by_action, y.transpose(1, 0, 2)).transpose(1, 0, 2)

        q = np.zeros((n_cells, n_actions, n_goals))
        u = np.zeros_like(q)
        for _ in range(max_iterations):
            q_new = backup(u)
            q_new[hit_cell, hit_act, goal] = self.reward_hit
            u_new = np.broadcast_to(q_new.max(axis=1, keepdims=True), q.shape)
            if p > 0.0:
                forced = u
                repeats = np.zeros_like(q)
                for d in range(1, high + 1):
                    forced = backup(forced)
                    forced[hit_cell, :, goal] = self.reward_hit
                    if d >= low:
                        repeats += forced
                u_new = (1.0 - p) * u_new + p * repeats / (high - low + 1)
            delta = max(np.abs(q_new - q).max(), np.abs(u_new - u).max())
            q, u = q_new, u_new
            if delta < tol:
                break
        else:
            raise RuntimeError(f"Value iteration did not converge within {max_iterations} iterations.")

        return q.reshape(n_cells, n_actions, n_cells, n_actions).transpose(0, 2, 1, 3).copy()

    @staticmethod
    def _reachability(cont):
        """
        Return a boolean array with shape (n_cells, n_cells), True at [s, g]
        when g can be reached from s through steps that do not terminate the
        episode (`cont` > 0), with s reaching itself.
        """

        reach = np.eye(cont.shape[0], dtype=bool) | (cont.max(axis=1) > 0.0)
        while True:
            # Paths of up to twice the current length, so the loop ends after
            # about log2(n_cells) passes.
            new = reach | ((reach.astype(np.int64) @ reach.astype(np.int64)) > 0)
            if np.array_equal(new, reach):
                return reach
            reach = new
