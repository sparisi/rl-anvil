import numpy as np
from abc import ABC, abstractmethod
from omegaconf import DictConfig

from src.utils.misc import random_argmax, random_choice
import src.approximator
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
        _, advantages = self._critic.returns_and_advantages(batch)

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
