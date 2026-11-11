import numpy as np
from abc import ABC, abstractmethod
from omegaconf import DictConfig

from src.utils.misc import StatsTracker, random_argmax, random_argmin, random_choice
import src.approximator
import src.parameter
from src.critic import Critic
from src.replay_memory import ReplayMemory
from src.pseudocount import is_neighbor_binned

# Where an action comes from, stored for every step in the replay memory under
# `act_source`.
ACT_RANDOM = 0  # uniformly random
ACT_GCRL = 1    # greedy on the goal-conditioned critic, toward the current goal
ACT_GREEDY = 2  # greedy on the critic's Q

# Tolerance of the comparisons between values of the true visit Q-function
# (src.oracle.TrueVisitQ).
ORACLE_TOL = 1e-9


def eps_greedy(x: np.array, eps: float, axis: int = -1):
    i = x == x.max(axis=axis, keepdims=True)
    r = eps / x.shape[axis]
    return np.where(i, (1.0 - eps) / i.sum(axis=axis, keepdims=True) + r, r)

def softmax(x: np.array, eps: float = 1.0, axis: int = -1):
    x_max = x.max(axis=axis, keepdims=True)
    x_exp = np.exp((x - x_max) / max(eps, 1e-12))
    return x_exp / x_exp.sum(axis=axis, keepdims=True)

def resmax(x: np.array, eps: float = 1.0, axis: int = -1):
    x_max = x.max(axis=axis, keepdims=True)
    p_non_max = 1.0 / (x.shape[axis] + 1.0 / max(eps, 1e-12) * (x_max - x))
    max_i = x == x_max
    non_max_i = np.logical_not(max_i)
    p_non_max *= non_max_i
    p_max = 1.0 / max_i.sum(axis=axis, keepdims=True) * (1.0 - p_non_max.sum(axis=axis, keepdims=True))
    return np.where(max_i, p_max, p_non_max)


class Actor(ABC):
    def __init__(
        self,
        critic: Critic,
        pi: DictConfig = None,
        seed: int = None,
        **kwargs,
    ):
        """
        Args:
            critic (Critic): the critic,
            pi (DictConfig): configuration to initialize the actor's own approximator
                (e.g., a neural network). The actor acts on its outputs, and
                `update` trains it with the critic's output (e.g., as in PPO).
                If None, the actor acts on the estimates of the critic directly
                (as in DQN), and `update` does nothing,
            seed (int): seed to initialize the network for reproducibility,
        """

        self._critic = critic
        self._train = True
        self._pi = None
        if pi is not None:
            self._pi = getattr(src.approximator, pi.id)(
                critic.obs_shape,
                critic.n_actions,
                **pi,
                seed=seed,
            )
        self.reset(seed=seed)

    def __call__(self, obs, rng_generator, **kwargs):
        """
        Draw one action in one state. Not vectorized in training mode, when
        the action comes from `explore`. In evaluation mode, the action comes
        from `greedy`, which is vectorized.
        """

        # Always draw action in eval mode.
        self._critic.eval()
        if self._pi is not None:
            self._pi.eval()

        if self._train:
            self.t += 1
            return self.explore(obs, rng_generator, **kwargs)
        else:
            return self.greedy(obs, rng_generator, **kwargs)

    def _values(self, obs):
        """
        Return the values the actor acts on: the outputs of its own network or
        the estimates of the critic. If normalization is enabled in the critic,
        observations are normalized.
        """

        if self._pi is None:
            return self._critic(obs)  # calls self._critic._normalize_obs
        return self._pi(self._critic._normalize_obs(obs))

    def greedy(self, obs, rng_generator, **kwargs):
        """
        Return the greedy action. Vectorized.
        """

        return random_argmax(self._values(obs), rng_generator, axis=-1)

    @abstractmethod
    def explore(self, obs, rng_generator, **kwargs):
        pass

    @abstractmethod
    def info(self):
        pass

    def rng_generator(self, seed=None):
        return np.random.default_rng(seed=seed)

    def update(self, replay_memory, rng_generator=None, **kwargs):
        """
        Train the actor's own network with the advantages of the critic on the
        transitions collected since the last update (OnlineMemory), and return
        a dictionary of statistics. Without a network, it does nothing and
        returns an empty dictionary.
        The network update assumes that the transitions were sampled from the
        policy of the network, as Softmax does with eps = 1 and a PolicyNetwork.
        """

        if self._pi is None:
            return {}

        self._pi.train()
        self._critic.train()

        batch = replay_memory.get()
        advantages = self._critic.advantages(batch)

        # Every transition is a sample for the minibatches.
        n = advantages.size
        obs = self._critic._normalize_obs(batch["obs"])
        return self._pi.update(
            obs.reshape(n, *obs.shape[2:]),
            batch["act"].reshape(n),
            advantages.reshape(n),
            rng_generator=rng_generator,
        )

    def post_update(self, *args, **kwargs):
        pass

    def post_step(self, *args, **kwargs):
        pass

    def reset(self, seed=None):
        self.t = 0
        if self._pi is not None:
            self._pi.reset(seed=seed)

    def reset_episode(self, seed=None, *args, **kwargs):
        pass

    def eval(self):
        self._train = False

    def train(self):
        self._train = True

    def copy_from(self, source):
        self.t = source.t
        self._train = source._train
        self._critic.copy_from(source._critic)
        if self._pi is not None:
            self._pi.copy_from(source._pi)


class EpsilonGreedy(Actor):
    def __init__(self, *args, eps: DictConfig, **kwargs):
        """
        Args:
            eps (DictConfig): configuration to initialize the exploration coefficient
                epsilon,
        """

        self._eps = getattr(src.parameter, eps.id)(**eps)
        Actor.__init__(self, *args, **kwargs)

    def copy_from(self, source):
        Actor.copy_from(self, source)
        self._eps.copy_from(source._eps)

    def explore(self, obs, rng_generator, **kwargs):
        if rng_generator.random() < self._eps.value:
            return rng_generator.integers(self._critic.n_actions)
        else:
            return random_argmax(self._values(obs), rng_generator, axis=-1)

    def post_step(self, *args, **kwargs):
        self._eps.step()

    def reset(self, seed=None):
        Actor.reset(self, seed=seed)
        self._eps.reset()

    def info(self):
        return {"eps": self._eps.value}


class Resmax(Actor):
    def __init__(self, *args, eps: DictConfig, **kwargs):
        """
        Args:
            eps (DictConfig): configuration to initialize the exploration coefficient
                epsilon,
        """

        self._eps = getattr(src.parameter, eps.id)(**eps)
        Actor.__init__(self, *args, **kwargs)

    def copy_from(self, source):
        Actor.copy_from(self, source)
        self._eps.copy_from(source._eps)

    def explore(self, obs, rng_generator, **kwargs):
        p = resmax(self._values(obs), self._eps.value)
        return random_choice(p, rng_generator, axis=-1)

    def post_step(self, *args, **kwargs):
        self._eps.step()

    def reset(self, seed=None):
        Actor.reset(self, seed=seed)
        self._eps.reset()

    def info(self):
        return {"eps": self._eps.value}


class Softmax(Actor):
    def __init__(self, *args, eps: DictConfig, **kwargs):
        """
        Args:
            eps (DictConfig): configuration to initialize the exploration coefficient
                epsilon (eps → 0 makes the distribution greedy; eps → ∞ makes it
                uniform; eps = 1 recovers vanilla categorical / softmax sampling),
        """

        self._eps = getattr(src.parameter, eps.id)(**eps)
        Actor.__init__(self, *args, **kwargs)

    def copy_from(self, source):
        Actor.copy_from(self, source)
        self._eps.copy_from(source._eps)

    def explore(self, obs, rng_generator, **kwargs):
        p = softmax(self._values(obs), self._eps.value)
        return random_choice(p, rng_generator, axis=-1)

    def post_step(self, *args, **kwargs):
        self._eps.step()

    def reset(self, seed=None):
        Actor.reset(self, seed=seed)
        self._eps.reset()

    def info(self):
        return {"eps": self._eps.value}


# ------------------------------------------------------------------------------
# -------------------------------- GCRL ----------------------------------------
# ------------------------------------------------------------------------------

class GoalConditioned(EpsilonGreedy):
    """
    Common base class for goal-conditioned actors.
    Use together with Q-Visit critics.

    The main differences between the algorithms are the following (beside how
    they score goal candidates).
    - Once the goal is reached, all algorithms reset it to None. If
      `random_tail=True`, exploration then continues with random actions
      until the end of the episode; otherwise a new goal is selected at
      the next policy step. Per-step configurations typically set
      `random_tail=False` in yaml — the code does not enforce this,
      the two flags are independent.
    - When `per_step_selection=True`, every policy step is an opportunity
      to (re)select a goal. The candidate pool is drawn fresh each time
      from the replay buffer, and the current goal (if any) is included as
      a candidate so it can be kept.
    """

    def __init__(
        self,
        critic: Critic,
        batch_size_goal: int,
        random_tail: bool,
        per_step_selection: bool,
        goal_value_check: bool,
        **kwargs,
    ):
        """
        Args:
            critic (Critic): the critic providing estimates of state-action values,
            batch_size_goal (int): number of samples from the replay buffer
                used as goal candidates. Their counts come from the replay
                memory, which decides whether they are true binned counts or
                pseudocounts (see its rho),
            random_tail (bool): if True, switch to fully-random actions for the
                rest of the episode after the current goal is reached; otherwise
                a new goal is selected,
            per_step_selection (bool): if True, re-select the goal at every step;
                if False, only select when there is no current goal (i.e. on
                episode start or after the previous goal was reached),
            goal_value_check (bool): only relevant when `per_step_selection=True`.
                If True, the observation at goal-selection time is stored as
                `_sel_obs`. At every subsequent policy step both V(s_current, G)
                and V(s_sel, G) are re-evaluated through the CURRENT critic
                parameters — this drift-cancels any global shift in the critic
                between selection and now. The goal is forgotten (and a new one
                selected) if V(s_current, G) < V(s_sel, G).
        """

        self.batch_size_goal = batch_size_goal
        self._random_tail = random_tail
        self._per_step_selection = per_step_selection
        self._goal_value_check = goal_value_check
        self.goal_selected_count = None  # for debugging
        self.goal_reached_count = None
        # The true visit Q-function (src.oracle.TrueVisitQ), set by the experiment
        # when results.true_visit_check is True. Every diagnostic that reads it
        # runs only when it is set, and none of them draws random numbers.
        self.true_visit = None
        super().__init__(critic, **kwargs)

    def copy_from(self, source):
        super().copy_from(source)
        self.batch_size_goal = source.batch_size_goal
        self._random_tail = source._random_tail
        self._previous_was_random = False
        self._per_step_selection = source._per_step_selection
        self._goal_value_check = source._goal_value_check
        self._episode_critic_idx = source._episode_critic_idx
        self._sel_obs = None if source._sel_obs is None else source._sel_obs.copy()
        self._random_exploration = source._random_exploration
        self._steps_to_goal_current = source._steps_to_goal_current
        self._steps_to_goal_sum = source._steps_to_goal_sum
        self._goal_reached = source._goal_reached
        self._goal_selections = source._goal_selections
        self._new_goal_selections = source._new_goal_selections
        self._goal_reselections = source._goal_reselections
        self._self_goal_selections = source._self_goal_selections
        self._policy_steps = source._policy_steps
        self._random_tail_steps = source._random_tail_steps
        self.goal = None if source.goal is None else source.goal.copy()
        self._goal_used_last_step = None
        self._act_source_last_step = ACT_RANDOM
        self._goal_selection_last_step = False
        if source.goal_selected_count is not None:
            self.goal_selected_count.copy_from(source.goal_selected_count)
        if source.goal_reached_count is not None:
            self.goal_reached_count.copy_from(source.goal_reached_count)

    def info(self):
        return super().info() | {
            "goal_success_%": np.nan if self._new_goal_selections == 0 else self._goal_reached / self._new_goal_selections,
            "self_goal_%": np.nan if self._goal_selections == 0 else self._self_goal_selections / self._goal_selections,
            "goal_reselections_%": np.nan if self._policy_steps == 0 else self._goal_reselections / self._policy_steps,
            "goal_selection_rate": np.nan if self._policy_steps == 0 else self._goal_selections / self._policy_steps,
            "avg_steps_to_goal": np.nan if self._goal_reached == 0 else self._steps_to_goal_sum / self._goal_reached,
            "random_tail_steps_%": np.nan if (self.t - self._policy_steps) == 0 else self._random_tail_steps / (self.t - self._policy_steps),
        } | self._oracle_stats.get()

    def reset_goal(self):
        self.goal = None
        self._steps_to_goal_current = 0
        self._sel_obs = None

    def reset_episode(self, seed=None, rng_generator=None, *args, **kwargs):
        self.reset_goal()
        self._random_exploration = False
        self._pick_episode_critic(rng_generator)

    @property
    def _acting_critic(self):
        """If the critic is an ensemble, only one critic selects actions throughout
        an episode.
        """

        critics = getattr(self._critic, "critics", None)
        if critics is None:
            return self._critic
        return critics[self._episode_critic_idx]

    def _pick_episode_critic(self, rng_generator):
        """If the critic is an ensemble, draw the ensemble member that selects
        actions for this episode.
        Goal selection is unaffected: it reads the spread across the ensemble
        and still queries every member.
        """

        critics = getattr(self._critic, "critics", None)
        if critics is None:
            self._episode_critic_idx = 0
            return
        self._episode_critic_idx = int(rng_generator.integers(len(critics)))

    def _values(self, obs, goal=None):
        """
        Return the values the actor acts on toward `goal`: the outputs of its own
        network, or the estimates of the acting critic for the goal action. Without
        a goal, return the values of `Actor._values`.
        """

        if goal is None:
            return Actor._values(self, obs)
        if self._pi is None:
            return self._acting_critic(obs=obs, goal=goal["obs"])[..., goal["act"]]
        return self._pi(
            self._critic._normalize_obs(obs),
            self._critic._normalize_obs(goal["obs"])[..., self._critic.goal_idx],
            goal["act"],
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        # `_goal_used_last_step` is needed because `self.goal` is reset to None
        #     inside `_goal_reached_check` (called during exploration), and
        #     outside functions would miss it.
        # `_goal_selection_last_step` tells whether the step selected a goal,
        #     and it is also read from outside.
        # `_act_source_last_step` describes how the action was selected (randomly,
        #     or greedy on the goal-conditioned critic toward the current goal).
        self.goal = None
        self._goal_used_last_step = None
        self._act_source_last_step = ACT_RANDOM
        self._goal_selection_last_step = False
        self._random_exploration = False
        self._previous_was_random = False
        self._sel_obs = None
        self._episode_critic_idx = 0
        self._goal_selections = 0
        self._new_goal_selections = 0
        self._goal_reselections = 0
        self._goal_reached = 0
        self._self_goal_selections = 0
        self._policy_steps = 0
        self._random_tail_steps = 0
        self._steps_to_goal_current = 0
        self._steps_to_goal_sum = 0
        # Diagnostics on the true visit Q-function (see `true_visit`).
        # `_oracle_prev` holds (obs, act, was_random, goal) of the last step.
        # `_oracle_sel_obs` is the obs where the goal held was selected, and the
        #     `_since_sel_*` flags record how the steps since then changed its
        #     true value (see `_oracle_track_step`).
        # `_oracle_kept_old` tells whether the true scores of the last goal
        #     selection keep the goal held before it, or None if that goal was
        #     not a candidate.
        # `_oracle_stats` averages every diagnostic since the last reset.
        self._oracle_prev = None
        self._oracle_sel_obs = None
        self._oracle_kept_old = None
        self._since_sel_random = False
        self._since_sel_suboptimal = False
        self._since_sel_bad_landing = False
        self._oracle_stats = StatsTracker()
        if self.goal_selected_count is not None:
            self.goal_selected_count.reset()
        if self.goal_reached_count is not None:
            self.goal_reached_count.reset()

    def _in_goal_bin(self, obs, goal_obs, replay_memory):
        """Return whether `obs` is within one bin width of `goal_obs`."""

        if replay_memory.counter is None:
            raise ValueError(
                "Goal checks need the environment's binned counter, but this "
                "environment has none."
            )

        # The goal is reached if |s_t - g_t| < η, where η is the bin width of true counts.
        # this is equivalent to using an oracle, aligned with how GCRL environments are implemented.
        return bool(is_neighbor_binned(
            obs,
            goal_obs[None],
            replay_memory.counter,
        )[0])

    def _goal_reached_check(self, obs, act, replay_memory):
        if self.goal is None:
            return False

        is_reached = (
            self._in_goal_bin(obs, self.goal["obs"], replay_memory)
            and act == self.goal["act"]
        )

        if is_reached:
            self._goal_reached += 1
            self._steps_to_goal_sum += self._steps_to_goal_current
            if self.goal_reached_count is not None:
                self.goal_reached_count.update(self.goal["obs"], self.goal["act"])
            if self._random_tail:
                self._random_exploration = True
            self.reset_goal()

        return is_reached

    def _get_candidates(self, obs, rng_generator, replay_memory):
        candidates = replay_memory.get(
            batch_size=self.batch_size_goal,
            sequence_length=1,
            rng_generator=rng_generator,
            keys=["obs", "count"],
            priority_key=None,
        )
        candidate_goal_obs = candidates["obs"][:, 0]  # remove time dimension
        candidate_goal_n = candidates["count"][:, 0]

        extra_goals = []
        if self.goal is not None and self._per_step_selection:
            extra_goals.append(self.goal["obs"])

        if extra_goals:
            extra_goals = np.stack(extra_goals, axis=0)
            candidate_goal_obs = np.concatenate(
                (candidate_goal_obs, extra_goals),
                axis=0,
            )
            candidate_goal_n = np.concatenate(
                (candidate_goal_n, replay_memory.counts(extra_goals)),
                axis=0,
            )

        candidate_goal_n = np.clip(candidate_goal_n, 1e-3, None)
        return candidate_goal_obs, candidate_goal_n

    def _select_goal(self, obs, rng_generator, replay_memory):
        goal_obs, n = self._get_candidates(obs, rng_generator, replay_memory)

        q_visit = self._critic(
            obs=np.broadcast_to(obs[None], goal_obs.shape),
            goal=goal_obs,
        )
        score = self._score(q_visit, n)
        best = random_argmax(score.flatten(), rng_generator, axis=-1)
        goal_obs_idx, goal_act = np.unravel_index(best, score.shape)

        if self.true_visit is not None:
            self._oracle_score(obs, goal_obs, n, goal_obs_idx, goal_act)

        self.goal = {
            "obs": goal_obs[goal_obs_idx],
            "act": int(goal_act),
        }
        self._goal_selections += 1

    def _oracle_score(self, obs, goal_obs, n, goal_obs_idx, goal_act):
        """
        Score the goal candidates with `_score` on the true visit Q-function
        and record whether the selected goal also has the best true score, the
        true score it loses to the best one, the true values of both goals
        from `obs`, and whether the goal held before this selection, when it
        is a candidate, would have been kept.
        """

        q_true = self.true_visit(np.broadcast_to(obs[None], goal_obs.shape), goal_obs)
        score_true = self._score(q_true, n)
        best = score_true.max()
        oracle_idx, oracle_act = np.unravel_index(score_true.argmax(), score_true.shape)
        self._oracle_stats.update(**{
            "oracle_agree_%": score_true[goal_obs_idx, goal_act] >= best - ORACLE_TOL,
            "oracle_regret": best - score_true[goal_obs_idx, goal_act],
            "true_q_sel": q_true[goal_obs_idx, :, goal_act].max(),
            "true_q_oracle": q_true[oracle_idx, :, oracle_act].max(),
        })

        # `_get_candidates` appends the goal held as the last candidate, under
        # the same condition.
        self._oracle_kept_old = None
        if self.goal is not None and self._per_step_selection:
            self._oracle_kept_old = bool(score_true[-1, self.goal["act"]] >= best - ORACLE_TOL)

    def _oracle_track_step(self, obs):
        """
        Flag how the last step, taken toward the goal still held, changed the
        true value of that goal, given that `obs` is where it landed:
        - `_since_sel_random`: the action was random,
        - `_since_sel_suboptimal`: the action was not optimal under the true
          visit Q-function,
        - `_since_sel_bad_landing`: the action was optimal, but the transition
          landed in a state whose true value is below what the action expects.
        """

        if self._oracle_prev is None:
            return
        prev_obs, prev_act, prev_random, prev_goal = self._oracle_prev
        self._oracle_prev = None
        # Compared by content: per-step selection builds a new goal dict at every
        # selection, also when it keeps the same goal.
        if (
            prev_goal is None
            or self.goal is None
            or not np.array_equal(prev_goal["obs"], self.goal["obs"])
            or prev_goal["act"] != self.goal["act"]
        ):
            return

        if prev_random:
            self._since_sel_random = True
            return
        q_prev = self.true_visit(prev_obs, prev_goal["obs"])[:, prev_goal["act"]]
        if q_prev[prev_act] < q_prev.max() - ORACLE_TOL:
            self._since_sel_suboptimal = True
            return
        v_curr = self.true_visit(obs, prev_goal["obs"])[:, prev_goal["act"]].max()
        expected = q_prev[prev_act]
        if self.true_visit.reward_miss + self.true_visit.gamma * v_curr < expected - ORACLE_TOL:
            self._since_sel_bad_landing = True

    def _oracle_new_goal(self, obs):
        """Store `obs` as the obs where the new goal was selected and clear the
        flags of `_oracle_track_step`."""

        self._oracle_sel_obs = obs.copy()
        self._since_sel_random = False
        self._since_sel_suboptimal = False
        self._since_sel_bad_landing = False

    def _oracle_classify_reselection(self, old_goal, obs):
        """
        Record why `old_goal` was replaced at `obs`. If its true value at `obs`
        is below its true value where it was selected, every flag of
        `_oracle_track_step` raised since then counts as a cause; otherwise,
        the reselection counts as an error of the learned visit Q-function.
        With per-step selection, also record whether the true scores would
        have kept `old_goal`.
        """

        g, a_g = old_goal["obs"], old_goal["act"]
        v_curr = self.true_visit(obs, g)[:, a_g].max()
        v_sel = self.true_visit(self._oracle_sel_obs, g)[:, a_g].max()
        dropped = v_curr < v_sel - ORACLE_TOL
        stats = {
            "resel_random_%": dropped and self._since_sel_random,
            "resel_suboptimal_%": dropped and self._since_sel_suboptimal,
            "resel_stochastic_%": dropped and self._since_sel_bad_landing,
            "resel_value_error_%": not dropped,
        }
        if self._oracle_kept_old is not None:
            stats["resel_oracle_keep_%"] = self._oracle_kept_old
        self._oracle_stats.update(**stats)

    def explore(self, obs, rng_generator, replay_memory, **kwargs):
        if self.true_visit is not None:
            self._oracle_track_step(obs)

        if (
            rng_generator.random() < self._eps.value or
            self._random_exploration
        ):
            self._previous_was_random = True
            if self._random_exploration:
                self._random_tail_steps += 1
            self._act_source_last_step = ACT_RANDOM
            self._goal_selection_last_step = False
            act = rng_generator.integers(self._critic.n_actions)

        else:
            self._critic.eval()
            self._policy_steps += 1

            old_goal = self.goal

            # Value check. V is evaluated at the current obs and at the goal's
            # selection obs, both under the CURRENT critic (to prevent errors due
            # to the critic updates). If V(current) < V(selection) the actor is
            # moving away from the goal, i.e., it does not know how to reach it.
            # Or it has entered a state (e.g., due to environment stochasticity)
            # from where the goal is not reachable anymore.
            # In both cases, the goal should be reselected.
            should_select = False
            if self._per_step_selection:
                if self._goal_value_check:
                    if not self._previous_was_random and self.goal is not None and self._sel_obs is not None:
                        q_curr = self._critic(obs=obs, goal=self.goal["obs"])
                        q_sel = self._critic(obs=self._sel_obs, goal=self.goal["obs"])
                        curr_value = q_curr[..., self.goal["act"]].max()
                        sel_value = q_sel[..., self.goal["act"]].max()
                        if curr_value < sel_value:
                            self.reset_goal()  # Sets goal to None, triggering `should_select` below
                else:
                    should_select = True

            should_select |= self.goal is None

            if should_select:
                self._select_goal(obs, rng_generator, replay_memory)
                self._sel_obs = obs.copy()
                if self._in_goal_bin(obs, self.goal["obs"], replay_memory):
                    self._self_goal_selections += 1
                goal_changed = old_goal is not None and (
                    not np.array_equal(old_goal["obs"], self.goal["obs"])
                    or old_goal["act"] != self.goal["act"]
                )
                if old_goal is None or goal_changed:
                    self._new_goal_selections += 1
                if goal_changed:
                    self._goal_reselections += 1
                    # Steps to goal are counted from the selection of the goal now held.
                    self._steps_to_goal_current = 0
                    if self.true_visit is not None:
                        self._oracle_classify_reselection(old_goal, obs)
                if self.true_visit is not None and (old_goal is None or goal_changed):
                    self._oracle_new_goal(obs)

                if self.goal_selected_count is not None:
                    self.goal_selected_count.update(self.goal["obs"], self.goal["act"])

            act = random_argmax(self._values(obs, goal=self.goal), rng_generator, axis=-1)
            self._previous_was_random = False
            self._act_source_last_step = ACT_GCRL
            self._goal_selection_last_step = should_select

        if self.goal is not None:
            self._steps_to_goal_current += 1
        # Snapshot the goal before _goal_reached_check may reset self.goal
        self._goal_used_last_step = self.goal
        if self.true_visit is not None:
            self._oracle_prev = (obs.copy(), int(act), self._act_source_last_step == ACT_RANDOM, self.goal)
        self._goal_reached_check(obs, act, replay_memory)
        return act

    def _score(self, *args, **kwargs):
        raise NotImplementedError


class SUN_Ratio(GoalConditioned):
    def _score(self, q_visit, n):
        return q_visit.max(-2) / n


class SUN_Novelty(GoalConditioned):
    def _score(self, q_visit, n):
        return 1.0 / n


class SUN_Reachability(GoalConditioned):
    def _score(self, q_visit, n):
        return q_visit.max(-2)


class SUN_UCB(GoalConditioned):
    def __init__(
        self,
        *args,
        beta_novelty: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta_novelty = beta_novelty

    def _score(self, q_visit, n):
        return q_visit.max(-2) + self.beta_novelty * np.sqrt(np.log(self.t) / n)


class DISCOVER(GoalConditioned):
    """
    The "achievability + novelty" version of DISCOVER, where there is no
    environment goal. Instead, the actor uses the goal-conditioned critic to
    enhance exploration of the environment.
    The goal is selected according to

        g_t = argmax_{g in memory}  mu(s_0, g) + beta_novelty * sigma(s_0, g)

    where mu and sigma are the mean and standard deviation of the
    goal-conditioned critic ensemble.

    Goal candidates are sampled from the main replay memory rather than from a
    dedicated achieved-goals buffer. For pure exploration, this is arguably the
    better candidate pool, as it includes everything the random-tail phase has
    stumbled into, which is where the actually novel states live.
    """

    def __init__(
        self,
        *args,
        beta_novelty: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta_novelty = beta_novelty
        self.q_std = np.nan

    def copy_from(self, source):
        super().copy_from(source)
        self.beta_novelty = source.beta_novelty
        self.q_std = source.q_std

    def _select_goal(self, obs, rng_generator, replay_memory):
        goal_obs = replay_memory.get(
            batch_size=self.batch_size_goal,
            sequence_length=1,
            rng_generator=rng_generator,
            keys=["obs"],
            priority_key=None,
        )["obs"][:, 0]  # remove time dimension

        obs_batched = np.broadcast_to(obs[None], goal_obs.shape)
        values = np.stack([c(obs=obs_batched, goal=goal_obs) for c in self._critic.critics], axis=0)
        mu = values.mean(axis=0).max(axis=-2)
        sigma = values.std(axis=0).max(axis=-2)
        score = mu + self.beta_novelty * sigma
        self.q_std = sigma.mean()
        flat_idx = random_argmax(score.flatten(), rng_generator, axis=-1)
        goal_obs_idx, goal_act = np.unravel_index(flat_idx, score.shape)

        self.goal = {
            "obs": goal_obs[goal_obs_idx],
            "act": int(goal_act),
        }
        self._goal_selections += 1

    def info(self):
        return super().info() | {
            "q_std": self.q_std,
        }


class AdaGoal(GoalConditioned):
    """
    The goal is selected according to

        g_t = argmax_{g in memory}  sigma(s_0, g),

    where sigma is the standard deviation of the goal-conditioned critic ensemble.
    As in the original AdaGoal, goal candidates are sampled from the replay memory.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.q_std = np.nan

    def copy_from(self, source):
        super().copy_from(source)
        self.q_std = source.q_std

    def _select_goal(self, obs, rng_generator, replay_memory):
        goal_obs = replay_memory.get(
            batch_size=self.batch_size_goal,
            sequence_length=1,
            rng_generator=rng_generator,
            keys=["obs"],
            priority_key=None,
        )["obs"][:, 0]  # remove time dimension

        obs_batched = np.broadcast_to(obs[None], goal_obs.shape)
        values = np.stack([c(obs=obs_batched, goal=goal_obs) for c in self._critic.critics], axis=0)
        sigma = values.std(axis=0).max(axis=-2)
        self.q_std = sigma.mean()
        flat_idx = random_argmax(sigma.flatten(), rng_generator, axis=-1)
        goal_obs_idx, goal_act = np.unravel_index(flat_idx, sigma.shape)

        self.goal = {
            "obs": goal_obs[goal_obs_idx],
            "act": int(goal_act),
        }
        self._goal_selections += 1

    def info(self):
        return super().info() | {
            "q_std": self.q_std,
        }
