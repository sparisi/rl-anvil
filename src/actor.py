import warnings
import numpy as np
from abc import ABC, abstractmethod
from omegaconf import DictConfig

from src.utils.misc import random_argmax, random_choice
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
