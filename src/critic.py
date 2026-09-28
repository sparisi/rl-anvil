import numpy as np
from abc import ABC, abstractmethod
from omegaconf import DictConfig
import gymnasium

import src.approximator
import src.parameter
from src.utils.running_stats import RunningStandardization
from src.utils.misc import cantor_pairing, random_argmax

def _unpack_batch(batch):
    obs = batch["obs"]
    act = batch["act"]
    rwd = batch["rwd"]
    next_obs = batch["next_obs"]
    term = batch["term"]
    trunc = batch["trunc"]
    weights = batch.get("weights", 1.0)
    return obs, act, rwd, next_obs, term, trunc, weights


def resmax(x: np.array, eps: float = 1.0, axis: int = -1):
    x_max = x.max(axis=axis, keepdims=True)
    denominator = x.shape[axis] + 1.0 / max(eps, 1e-12) * (x_max - x)
    return (
        (x / denominator).sum(axis=axis, keepdims=True) + x_max * (1.0 - (1.0 / denominator).sum(axis=axis, keepdims=True))
    ).squeeze(axis=axis)

def mellowmax(x: np.array, eps: float = 1.0, axis: int = -1):
    x_eps = x / max(eps, 1e-12)
    x_eps_max = x_eps.max(axis=axis, keepdims=True)
    return (
        np.log(np.exp(x_eps - x_eps_max).sum(axis=axis)) + x_eps_max.squeeze(axis=axis) - np.log(x.shape[axis])
    ) * eps


def is_act_greedy(q: np.array, act: np.array, axis: int, rtol: float = 0.0):
    """
    Returns an array of booleans where True denotes if the action is greedy given
    a Q-function, i.e., if max_a' Q(s, a') == Q(s, a) with some relative tolerance.
    """

    extra_nd = q.ndim - act.ndim - 1
    idx = act[(...,) + (None,) * (1 + extra_nd)]
    chosen_q = np.take_along_axis(q, idx, axis=axis).squeeze(axis=axis)
    q_max = q.max(axis=axis)
    return chosen_q >= q_max - np.abs(q_max) * rtol

def td_target(
    rwd: np.array,
    term: np.array,
    trunc: np.array,
    is_next_act_greedy: np.array,
    next_value: np.array,
    gamma: float,
    lmbda: float,
):
    """
    TD(λ) n-step target. Vectorized.

    Args:
        act (np.array): a_t,
        rwd (np.array): r_t,
        term (np.array): True if s_t is terminal, False otherwise,
        trunc (np.array): True if the sequence was truncated at s_t, False otherwise,
        is_next_act_greedy (np.array): True if a_{t+1} is greedy, False otherwise,
        next_value (np.array): value of the next state, i.e., max_a Q(s_{t+1}, a),
        gammma (float): discount factor,
        lmbda (float): eligibility trace factor,
    """

    bootstrap_val = next_value * (1.0 - term)
    lambda_returns = rwd + gamma * bootstrap_val
    T = rwd.shape[1]  # shape is (batch size, sequence length, ...)

    if lmbda == 0.0 or T == 1:
        return lambda_returns

    cut = np.logical_or(term, trunc)

    for t in range(T - 2, -1, -1):
        bellman_error = lambda_returns[:, t + 1] - bootstrap_val[:, t]
        current_lambda = lmbda * is_next_act_greedy[:, t]  # WATKIN'S CUT
        factor = gamma * current_lambda
        lambda_returns[:, t] += (1.0 - cut[:, t]) * factor * bellman_error

    return lambda_returns


class Critic(ABC):
    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def update(self, **kwargs):
        pass

    @abstractmethod
    def reset(self, seed=None):
        pass

    @abstractmethod
    def copy_from(self, source):
        pass

    @abstractmethod
    def rng_generator(self, seed=None):
        pass

    def post_update(self, *args, **kwargs):
        pass

    def post_step(self, *args, **kwargs):
        pass

    def train(self):
        pass

    def eval(self):
        pass


class QCritic(Critic):
    """
    Generic class for Q-function critics.
    """

    def __init__(
        self,
        gamma: float,
        lmbda: float,
        lr: DictConfig,
        clip_reward: bool,
        batch_size: int,
        sequence_length: int,
        **kwargs,
    ):
        """
        Args:
            gamma (float): discount factor,
            lmbda (float): eligibility trace factor,
            lr (DictConfig): configuration to initialize the learning rate,
            clip_reward (bool): if True, rewards are clipped in [-1, 1],
            batch_size (int): number of samples per mini-batch,
            sequence_length (int): number of consecutive steps per sample (for n-step / TD(λ) updates),
        """

        self.gamma = gamma
        self.lmbda = lmbda
        self.lr = getattr(src.parameter, lr.id)(**lr)
        self.q = None  # Q-function
        self.q_target = None  # for computing Q(next_state) in the TD target
        self.clip_reward = clip_reward
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def __call__(self, obs, act=None, target=False, **kwargs):
        if target:
            return self.q_target(obs, act)
        return self.q(obs, act)

    def reset(self, seed=None):
        self.lr.reset()
        self.q.reset(seed=seed)
        self.q_target.reset(seed=seed)

    def post_step(self, *args, **kwargs):
        self.lr.step()

    def copy_from(self, source):
        self.gamma = source.gamma
        self.lmbda = source.lmbda
        self.clip_reward = source.clip_reward
        self.batch_size = source.batch_size
        self.sequence_length = source.sequence_length
        self.lr.copy_from(source.lr)
        self.q.copy_from(source.q)
        self.q_target.copy_from(source.q_target)

    def rng_generator(self, seed=None):
        return np.random.default_rng(seed=seed)


class QTable(QCritic):
    """
    Instance of QCritic that uses tabular Q-functions.
    Visit counts are managed by the experiment (set as `visit_count`);
    keep as None here so a deep-copied critic (used for testing) does not
    carry a stale counter.
    """

    def __init__(
        self,
        obs_space: gymnasium.spaces.Discrete,
        act_space: gymnasium.spaces.Discrete,
        approximator: DictConfig,
        seed: int = None,
        **kwargs,
    ):
        QCritic.__init__(self, **kwargs)
        self.n_observations = obs_space.n
        self.n_actions = act_space.n
        self.q = getattr(src.approximator, approximator.id)(
            self.n_observations, self.n_actions, **approximator, seed=seed,
        )
        self.q_target = self.q  # with tabular Q we don't need a different target
        self.visit_count = None  # Will be set in the experiment
        QCritic.reset(self, seed=seed)

    def update(self, replay_memory, rng_generator=None, **kwargs):
        if rng_generator is None:
            rng_generator = self.rng_generator()

        batch = replay_memory.get(
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            rng_generator=rng_generator,
            priority_key="td_err",
        )
        obs, act, rwd, next_obs, term, trunc, weights = _unpack_batch(batch)

        if self.clip_reward:
            rwd = np.clip(rwd, -1.0, 1.0)

        B, T = rwd.shape
        is_next_act_greedy = None
        if self.lmbda > 0.0 and T > 1:  # compute mask to cut eligibility traces
            # act[:, 1:] is the next action within the sequence (a_{t+1} for t=0..T-2).
            # Use the online Q (not the target) for the Watkins mask.
            is_next_act_greedy = is_act_greedy(
                self.q(next_obs)[:, :-1],
                act[:, 1:],
                axis=-1,
            )
            trace_cut_frac = (1.0 - is_next_act_greedy).mean()
        else:
            trace_cut_frac = np.nan

        q_next = self.q_target(next_obs).max(-1)
        target = td_target(rwd, term, trunc, is_next_act_greedy, q_next, self.gamma, self.lmbda)
        stepsize = np.asarray(self.lr.value * weights)

        error, gradient_norm = self.q.update(
            obs.ravel(),
            act.ravel(),
            target=target.ravel(),
            stepsize=np.broadcast_to(stepsize, target.shape).ravel(),
        )
        error = error.reshape(target.shape)

        # Update priorities in PER (no-op on plain ReplayMemory).
        # Every sample in the (B, T) batch gets its priority set from its own
        # TD error, independent of how sampling picked the sequence endpoints.
        replay_memory.post_sampling(batch["idx"], error, "td_err")

        return {
            "td_err": error,
            "grad_norm": gradient_norm,
            "trace_cut_frac": trace_cut_frac,
        }


class QVisitTable(QTable):
    """
    This critic learns Q-functions based on the successor representation, to
    approximate first time visitation of state-action pairs.
    These are often called S-functions or W-functions in the RL literature. Here,
    they are they are called Q-visit.
    """

    def __init__(
        self,
        obs_space: gymnasium.spaces.Discrete,
        act_space: gymnasium.spaces.Discrete,
        approximator_visit: DictConfig,
        gamma_visit: float,
        lmbda_visit: float,
        lr_visit: DictConfig,
        batch_size_visit: int,
        sequence_length_visit: int,
        seed: int = None,
        **kwargs,
    ):
        QTable.__init__(self, obs_space, act_space, **kwargs, seed=seed)
        self.lr_visit = getattr(src.parameter, lr_visit.id)(**lr_visit)
        self.gamma_visit = gamma_visit
        self.lmbda_visit = lmbda_visit
        self.batch_size_visit = batch_size_visit
        self.sequence_length_visit = sequence_length_visit
        self.q_visit = getattr(src.approximator, approximator_visit.id)(
            self.n_observations,
            self.n_actions,
            self.n_observations * self.n_actions,
            **approximator_visit,
            seed=seed,
        )
        self.q_visit_target = self.q_visit
        QVisitTable.reset(self, seed=seed)

    def copy_from(self, source):
        QTable.copy_from(self, source)
        self.lr_visit.copy_from(source.lr_visit)
        self.gamma_visit = source.gamma_visit
        self.lmbda_visit = source.lmbda_visit
        self.batch_size_visit = source.batch_size_visit
        self.sequence_length_visit = source.sequence_length_visit
        self.q_visit.copy_from(source.q_visit)
        self.q_visit_target.copy_from(source.q_visit_target)

    def reset(self, seed=None):
        self.lr_visit.reset()
        self.q_visit.reset(seed=seed)
        self.q_visit_target.reset(seed=seed)
        QTable.reset(self, seed=seed)

    def update(self, replay_memory, rng_generator=None, **kwargs):
        if rng_generator is None:
            rng_generator = self.rng_generator()

        q_stats = QTable.update(self, replay_memory, rng_generator, **kwargs)

        # Separate batch for the visit critic (may use its own PER priority
        # and its own batch/sequence sizes).
        batch = replay_memory.get(
            batch_size=self.batch_size_visit,
            sequence_length=self.sequence_length_visit,
            rng_generator=rng_generator,
            priority_key="td_err_visit",
        )
        obs, act, rwd, next_obs, term, trunc, weights = _unpack_batch(batch)

        # q_visit shape is (n_obs, n_act, n_obs * n_act), storing value functions
        # of shape (n_obs, n_act) for all state-action pairs
        B, T = rwd.shape
        is_next_act_greedy = None
        if self.lmbda_visit > 0.0 and T > 1:
            is_next_act_greedy = is_act_greedy(
                self.q_visit(next_obs)[:, :-1],
                act[:, 1:],
                axis=-2,
            )
            trace_cut_frac_visit = (1.0 - is_next_act_greedy).mean()
        else:
            trace_cut_frac_visit = np.nan

        q_visit_next = self.q_visit_target(next_obs).max(-2)
        rwd_visit = np.zeros_like(q_visit_next)
        idx = np.ravel_multi_index((obs, act), (self.n_observations, self.n_actions))
        np.put_along_axis(rwd_visit, axis=-1, indices=idx[..., None], values=1.0)
        term_visit = np.logical_or(term[..., None], rwd_visit == 1.0)
        trunc_visit = np.broadcast_to(trunc[..., None], term_visit.shape)

        target = td_target(
            rwd_visit,
            term_visit,
            trunc_visit,
            is_next_act_greedy,
            q_visit_next,
            self.gamma_visit,
            self.lmbda_visit,
        )

        stepsize = np.asarray(self.lr_visit.value * weights)
        error_visit, gradient_norm_visit = self.q_visit.update(
            obs.ravel(),
            act.ravel(),
            target=target.reshape(-1, self.n_observations * self.n_actions),
            stepsize=np.broadcast_to(stepsize, target.shape[:-1]).ravel()[..., None],
        )
        error_visit = error_visit.reshape(target.shape)

        # Update priorities in PER (no-op on plain ReplayMemory).
        # Every sample in the (B, T) batch gets its priority set from its own
        # error (reduced over goals with max, since each sample is trained
        # against every goal), independent of how sampling picked endpoints.
        replay_memory.post_sampling(batch["idx"], error_visit.max(-1), "td_err_visit")

        return q_stats | {
            "td_err_visit": error_visit,
            "grad_norm_visit": gradient_norm_visit,
            "trace_cut_frac_visit": trace_cut_frac_visit,
        }

    def post_step(self, *args, **kwargs):
        self.lr_visit.step()
        QTable.post_step(self, *args, **kwargs)
