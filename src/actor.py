import warnings
import numpy as np
from abc import ABC, abstractmethod
from omegaconf import DictConfig

from src.utils.misc import random_argmax, random_argmin, random_choice
import src.parameter
from src.critic import Critic


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


class EpsilonGreedyBalancedWandering(EpsilonGreedy):
    """
    Inspired by "Near-Optimal Reinforcement Learning in Polynomial Time".
    Like ε-greedy, but instead of a random action the agent selects the
    action with the lowest count.
    """

    def explore(self, obs, rng_generator, **kwargs):
        if rng_generator.random() < self._eps.value:
            return random_argmax(-self._critic.visit_count(obs), rng_generator, axis=-1)
        else:
            return random_argmax(self._critic(obs), rng_generator, axis=-1)


class EpsilonGreedyWithUCB(EpsilonGreedy):
    """
    Actor with UCB exploration.
    """

    def explore(self, obs, rng_generator, **kwargs):
        if rng_generator.random() < self._eps.value:
            return rng_generator.integers(self._critic.n_actions)
        else:
            n = self._critic.visit_count(obs)
            if n.sum() == 0:
                return rng_generator.integers(self._critic.n_actions)
            ucb = np.sqrt(2.0 * np.log(n.sum()) / (n + 1e-12))
            return random_argmax(self._critic(obs) + ucb, rng_generator, axis=-1)


class EpsilonGreedyMinCount(EpsilonGreedy):
    """
    Instead of a random action, select one with the lowest count in the current state.
    """

    def explore(self, obs, rng_generator, **kwargs):
        if rng_generator.random() < self._eps.value:
            return random_argmin(self._critic.visit_count(obs), rng_generator, axis=-1)
        else:
            return random_argmax(self._critic(obs), rng_generator, axis=-1)


class EpsilonGreedyWithQVisit(EpsilonGreedy):
    """
    Actor for Q-visit critics.
    States that are never visited cannot be picked as goals.
    Before acting greedy, always select an action that has never been done in
    the current state.
    """

    def __init__(self, critic: Critic, beta_bar: float, **kwargs):
        """
        Args:
            critic (Critic): the critic providing estimates of state-action values,
            beta_bar (float): ratio threshold for exploration / exploitation,
        """

        EpsilonGreedy.__init__(self, critic, **kwargs)
        self._beta_bar = beta_bar
        self.beta = np.inf
        self.goal = None
        self.goal_selected_count = None # for debugging, initialized by the experiment
        self.goal_reached_count = None

    def info(self):
        return EpsilonGreedy.info(self) | {"beta": self.beta}

    def _score(self, q_visit, n):
        return q_visit / (n[None] + 1e-8)

    def explore(self, obs, rng_generator, **kwargs):
        # Cleared every step: only one of the branches below selects a goal, and
        # a goal left over from an earlier step would be recorded as this step's.
        self.goal = None

        n = self._critic.visit_count()
        unvisited_actions_now = n[obs] == 0
        if np.any(unvisited_actions_now):
            return random_argmax(unvisited_actions_now, rng_generator, axis=-1)

        never_visited = n == 0
        with warnings.catch_warnings():  # ignore "RuntimeWarning: divide by zero encountered in double_scalars"
            warnings.simplefilter("ignore", category=RuntimeWarning)
            beta = np.log(self.t) / (n + never_visited * n.max()).min()  # filter out never-visited pairs (possibly unreachable)
            self.beta = beta

        if beta > self._beta_bar:
            if rng_generator.random() < self._eps.value:
                return rng_generator.integers(self._critic.n_actions)

            q_visit = self._critic.q_visit(obs).reshape(-1, *n.shape)
            q_visit *= (1.0 - never_visited)  # reachability of never-visited pairs is 0
            score = self._score(q_visit, n)
            best = np.argmax(score.flatten(), axis=-1)  # break ties deterministically
            act, goal_s, goal_a = np.unravel_index(best, score.shape)
            self.goal = {"obs": goal_s, "act": goal_a}
            if self.goal_selected_count is not None:
                self.goal_selected_count.update(goal_s, goal_a)
            act = random_argmax(score[:, goal_s, goal_a], rng_generator, axis=-1)
            if obs == goal_s and act == goal_a and self.goal_reached_count is not None:
                self.goal_reached_count.update(obs, act)
            return act

        else:
            return random_argmax(self._critic.q(obs), rng_generator, axis=-1)
