import warnings
import numpy as np
from abc import ABC, abstractmethod
from omegaconf import DictConfig

from src.utils.misc import random_argmax, random_argmin, random_choice
import src.parameter
from src.critic import Critic
from src.replay_memory import ReplayMemory
from src.pseudocount import is_neighbor_binned


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
    def __init__(self, critic: Critic, seed: int = None):
        self._critic = critic
        self._train = True
        self.reset(seed=seed)

    def __call__(self, obs, rng_generator, **kwargs):
        """
        Draw one action in one state. Not vectorized.
        """

        if self._train:
            self.t += 1
            self._critic.eval()
            return self.explore(obs, rng_generator, **kwargs)
        else:
            return random_argmax(self._critic(obs), rng_generator, axis=-1)

    @abstractmethod
    def explore(self, obs, rng_generator, **kwargs):
        pass

    @abstractmethod
    def info(self):
        pass

    def rng_generator(self, seed=None):
        return np.random.default_rng(seed=seed)

    def post_update(self, *args, **kwargs):
        pass

    def post_step(self, *args, **kwargs):
        pass

    def reset(self, seed=None):
        self.t = 0

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


class EpsilonGreedy(Actor):
    def __init__(self, critic: Critic, eps: DictConfig, **kwargs):
        """
        Args:
            critic (Critic): the critic providing estimates of state-action values,
            eps (DictConfig): configuration to initialize the exploration coefficient
                epsilon,
        """

        self._eps = getattr(src.parameter, eps.id)(**eps)
        Actor.__init__(self, critic)

    def copy_from(self, source):
        Actor.copy_from(self, source)
        self._eps.copy_from(source._eps)

    def explore(self, obs, rng_generator, **kwargs):
        if rng_generator.random() < self._eps.value:
            return rng_generator.integers(self._critic.n_actions)
        else:
            self._critic.eval()
            return random_argmax(self._critic(obs), rng_generator, axis=-1)

    def post_step(self, *args, **kwargs):
        self._eps.step()

    def reset(self, seed=None):
        Actor.reset(self, seed=seed)
        self._eps.reset()

    def info(self):
        return {"eps": self._eps.value}


class Resmax(Actor):
    def __init__(self, critic: Critic, eps: DictConfig, **kwargs):
        """
        Args:
            critic (Critic): the critic providing estimates of state-action values,
            eps (DictConfig): configuration to initialize the exploration coefficient
                epsilon,
        """

        self._eps = getattr(src.parameter, eps.id)(**eps)
        Actor.__init__(self, critic)

    def copy_from(self, source):
        Actor.copy_from(self, source)
        self._eps.copy_from(source._eps)

    def explore(self, obs, rng_generator, **kwargs):
        p = resmax(self._critic(obs), self._eps.value)
        return random_choice(p, rng_generator, axis=-1)

    def post_step(self, *args, **kwargs):
        self._eps.step()

    def reset(self, seed=None):
        Actor.reset(self, seed=seed)
        self._eps.reset()

    def info(self):
        return {"eps": self._eps.value}


class Softmax(Actor):
    def __init__(self, critic: Critic, eps: DictConfig, **kwargs):
        """
        Args:
            critic (Critic): the critic providing estimates of state-action values,
            eps (DictConfig): configuration to initialize the exploration coefficient
                epsilon (eps → 0 makes the distribution greedy; eps → ∞ makes it
                uniform; eps = 1 recovers vanilla categorical / softmax sampling),
        """

        self._eps = getattr(src.parameter, eps.id)(**eps)
        Actor.__init__(self, critic)

    def copy_from(self, source):
        Actor.copy_from(self, source)
        self._eps.copy_from(source._eps)

    def explore(self, obs, rng_generator, **kwargs):
        p = softmax(self._critic(obs), self._eps.value)
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
        self._goal_reselections = source._goal_reselections
        self._self_goal_selections = source._self_goal_selections
        self._policy_steps = source._policy_steps
        self._random_tail_steps = source._random_tail_steps
        self.goal = None if source.goal is None else source.goal.copy()
        self._goal_used_last_step = None
        if source.goal_selected_count is not None:
            self.goal_selected_count.copy_from(source.goal_selected_count)
        if source.goal_reached_count is not None:
            self.goal_reached_count.copy_from(source.goal_reached_count)

    def info(self):
        return super().info() | {
            "goal_success_%": np.nan if self._goal_selections == 0 else self._goal_reached / self._goal_selections,
            "self_goal_%": np.nan if self._goal_selections == 0 else self._self_goal_selections / self._goal_selections,
            "goal_reselections_%": np.nan if self._policy_steps == 0 else self._goal_reselections / self._policy_steps,
            "goal_selection_rate": np.nan if self._policy_steps == 0 else self._goal_selections / self._policy_steps,
            "avg_steps_to_goal": np.nan if self._goal_reached == 0 else self._steps_to_goal_sum / self._goal_reached,
            "random_tail_steps_%": np.nan if (self.t - self._policy_steps) == 0 else self._random_tail_steps / (self.t - self._policy_steps),
        }

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

    def reset(self, seed=None):
        super().reset(seed=seed)
        # `self._goal_used_last_step` is needed because `self.goal` is reset to
        # None inside `_goal_reached_check` (called during exploration),
        # and outside functions would miss it.
        self.goal = None
        self._goal_used_last_step = None
        self._random_exploration = False
        self._previous_was_random = False
        self._sel_obs = None
        self._episode_critic_idx = 0
        self._goal_selections = 0
        self._goal_reselections = 0
        self._goal_reached = 0
        self._self_goal_selections = 0
        self._policy_steps = 0
        self._random_tail_steps = 0
        self._steps_to_goal_current = 0
        self._steps_to_goal_sum = 0
        if self.goal_selected_count is not None:
            self.goal_selected_count.reset()
        if self.goal_reached_count is not None:
            self.goal_reached_count.reset()

    def _goal_reached_check(self, obs, act, replay_memory):
        if self.goal is None:
            return False

        # The goal is reached if |s_t - g_t| < η, where η is the bin width of true counts.
        # this is equivalent to using an oracle, aligned with how GCRL environments are implemented.
        is_reached = bool(is_neighbor_binned(
            obs,
            self.goal["obs"][None],
            replay_memory.counter,
        )[0]) and act == self.goal["act"]

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

        self.goal = {
            "obs": goal_obs[goal_obs_idx],
            "act": int(goal_act),
        }
        self._goal_selections += 1

    def explore(self, obs, rng_generator, replay_memory, **kwargs):
        if (
            rng_generator.random() < self._eps.value or
            self._random_exploration
        ):
            self._previous_was_random = True
            if self._random_exploration:
                self._random_tail_steps += 1
            # No goal drove this action, but keep the caller informed of the
            # goal currently held (matches the pre-snapshot behavior).
            self._goal_used_last_step = self.goal
            return rng_generator.integers(self._critic.n_actions)

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
            if np.array_equal(self.goal["obs"], obs):
                self._self_goal_selections += 1
            if old_goal is not None and (
                not np.array_equal(old_goal["obs"], self.goal["obs"])
                or old_goal["act"] != self.goal["act"]
            ):
                self._goal_reselections += 1

            if self.goal_selected_count is not None:
                self.goal_selected_count.update(self.goal["obs"], self.goal["act"])

        q_visit = self._acting_critic(obs=obs, goal=self.goal["obs"])

        act = random_argmax(q_visit[..., self.goal["act"]], rng_generator, axis=-1)

        self._steps_to_goal_current += 1
        # Snapshot the goal before _goal_reached_check may reset self.goal
        self._goal_used_last_step = self.goal
        self._goal_reached_check(obs, act, replay_memory)
        self._previous_was_random = False
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
    def _score(self, q_visit, n):
        return q_visit.max(-2) + np.sqrt(np.log(self.t) / n)


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
