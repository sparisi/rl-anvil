import numpy as np
from abc import ABC, abstractmethod
from omegaconf import DictConfig
import gymnasium

import src.approximator
import src.parameter
import src.goal_relabeling
from src.pseudocount import is_neighbor
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


class QNetwork(QCritic):
    """
    Instance of QCritic that uses neural network Q-functions.
    It also keeps visit counts that are updated by the data collection procedure.
    """

    def __init__(
        self,
        obs_space: gymnasium.spaces.Box,
        act_space: gymnasium.spaces.Discrete,
        target_copy_frequency: int,
        tau: float,
        normalize_observation: bool,
        approximator: DictConfig,
        goal_idx: list = None,
        explore_only: bool = False,
        seed: int = None,
        **kwargs,
    ):
        """
        Args:
            obs_space (gymnasium.spaces.Box): observation space,
            act_space (gymnasium.spaces.Discrete): action space,
            target_copy_frequency (int): after how many updates the target
                network is updated (hard or Polyak copy of Q-network),
            tau (float): Polyak averaging coefficient,
            normalize_observation (bool): if True, observations are normalized using their
                running mean and standard deviation,
            approximator (DictConfig): configuration to initialize the Q-network,
            goal_idx (list): indices of the observation elements a goal-conditioned
                agent treats as the goal. None means all of them,
            explore_only (bool): if True, the Q-function w.r.t. extrinsic reward
                will not be learned,
            seed (int): seed to initialize the network for reproducibility,
        """

        QCritic.__init__(self, **kwargs)
        self.n_actions = act_space.n
        self.goal_idx = slice(goal_idx) if goal_idx is None else goal_idx
        self.explore_only = explore_only
        self.q = getattr(src.approximator, approximator.id)(
            obs_space.shape, act_space.n, **approximator, seed=seed,
        )
        self.q_target = getattr(src.approximator, approximator.id)(
            obs_space.shape, act_space.n, **approximator, seed=seed,
        )
        self.q_target.eval()
        self.target_copy_frequency = target_copy_frequency
        self.target_copy_counter = 0
        self.tau = tau
        self.n_updates = 0
        self.running_obs = RunningStandardization(obs_space.shape)
        self.normalize_observation = normalize_observation
        self.visit_count = None  # Will be set in the experiment
        QNetwork.reset(self, seed=seed)

    def __call__(self, obs, act=None, target=False, **kwargs):
        if target:
            return self.q_target(self._normalize_obs(obs), act)
        return self.q(self._normalize_obs(obs), act)

    def _normalize_obs(self, obs):
        if self.normalize_observation:
            return self.running_obs.normalize(obs)
        else:
            return obs

    def post_step(self, obs, act, *args, **kwargs):
        QCritic.post_step(self, obs, act, *args, **kwargs)
        self.running_obs.update(obs)

    def reset(self, seed=None):
        QCritic.reset(self, seed=seed)
        self.target_copy_counter = 0
        self.running_obs.reset()

    def update(self, replay_memory, rng_generator=None, **kwargs):
        self.n_updates += 1
        self.train()

        # A purely exploratory agent never learns the extrinsic Q-function; the
        # visit Q-function of a subclass is all it acts on.
        if self.explore_only:
            return {}

        if rng_generator is None:
            rng_generator = self.rng_generator()

        batch = replay_memory.get(
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            rng_generator=rng_generator,
            priority_key="td_err",
        )
        obs, act, rwd, next_obs, term, trunc, weights = _unpack_batch(batch)

        obs = self._normalize_obs(obs)
        next_obs = self._normalize_obs(next_obs)

        B, T = rwd.shape

        if self.clip_reward:
            rwd = np.clip(rwd, -1.0, 1.0)

        # Double DQN
        tolerance_max = 1.0 - self.gamma
        q_next = self.q(next_obs, with_gradient=False)
        next_a_max = random_argmax(
            q_next,
            rng_generator=rng_generator,
            axis=-1,
            # rtol=tolerance_max,
        )
        target_q_next = np.take_along_axis(
            self.q_target(next_obs, with_gradient=False),
            next_a_max[..., None],
            axis=-1,
        ).squeeze(-1)

        is_next_act_greedy = None
        if self.lmbda > 0.0 and T > 1:  # compute mask to cut eligibility traces
            is_next_act_greedy = is_act_greedy(
                q_next[:, :-1],
                act[:, 1:],
                axis=-1,
                # rtol=tolerance_max,
            )  # use the main Q, not the target Q
            trace_cut_frac = (1.0 - is_next_act_greedy).mean()
        else:
            trace_cut_frac = np.nan

        target = td_target(
            rwd,
            term,
            trunc,
            is_next_act_greedy,
            target_q_next,
            self.gamma,
            self.lmbda,
        )
        stepsize = self.lr.value * weights
        error, gradient_norm, q = self.q.update(
            obs, act, target=target, stepsize=stepsize, rng_generator=rng_generator,
        )

        # Update priorities in PER (will skip if no PER)
        replay_memory.post_sampling(batch["idx"], error, "td_err")  # td_err is already squared

        return {
            "td_err": error,
            "grad_norm": gradient_norm,
            "q_mean": q.mean(),
            "trace_cut_frac": trace_cut_frac,
        }

    def post_update(self):
        if self.n_updates > 0:
            self.target_copy_counter += 1
            if self.target_copy_counter % self.target_copy_frequency == 0:
                self.target_copy_counter = 0
                self.q_target.copy_from(self.q, self.tau)

    def train(self):
        self.q.train()
        # self.q_target.train()

    def eval(self):
        self.q.eval()
        # self.q_target.eval()

    def copy_from(self, source):
        QCritic.copy_from(self, source)
        self.target_copy_frequency = source.target_copy_frequency
        self.target_copy_counter = source.target_copy_counter
        self.tau = source.tau
        self.n_updates = source.n_updates
        self.running_obs.copy_from(source.running_obs)


class QVisitNetwork(QNetwork):
    """
    This critic also learns Q-functions based on the successor representation, to
    approximate first time visitation of state-action pairs.
    These are often called S-functions or W-functions in the RL literature. Here,
    they are they are called Q-visit.
    """

    def __init__(
        self,
        obs_space: gymnasium.spaces.Box,
        act_space: gymnasium.spaces.Discrete,
        approximator_visit: DictConfig,
        gamma_visit: float,
        lmbda_visit: float,
        lr_visit: DictConfig,
        batch_size_visit: int,
        sequence_length_visit: int,
        relabeling: DictConfig,
        negative_visit_reward: bool,
        reward_visit_hard: bool,
        seed: int = None,
        **kwargs,
    ):
        assert hasattr(src.goal_relabeling, relabeling.id), f"Unknown relabeling '{relabeling.id}'"
        QNetwork.__init__(self, obs_space, act_space, **kwargs, seed=seed)
        self.lr_visit = getattr(src.parameter, lr_visit.id)(**lr_visit)
        self.gamma_visit = gamma_visit
        self.lmbda_visit = lmbda_visit
        self.batch_size_visit = batch_size_visit
        self.sequence_length_visit = sequence_length_visit
        self.negative_visit_reward = negative_visit_reward
        self.reward_visit_hard = reward_visit_hard
        self.relabeling = relabeling
        goal_shape = np.empty(obs_space.shape, dtype=obs_space.dtype)[..., self.goal_idx].shape
        self.q_visit = getattr(src.approximator, approximator_visit.id)(
            goal_shape, obs_space.shape, act_space.n, **approximator_visit, seed=seed,
        )
        self.q_visit_target = getattr(src.approximator, approximator_visit.id)(
            goal_shape, obs_space.shape, act_space.n, **approximator_visit, seed=seed,
        )
        self.q_visit_target.eval()
        QVisitNetwork.reset(self, seed=seed)

    def __call__(self, obs, act=None, goal=None, target=False, **kwargs):
        if target:
            if goal is None:
                return self.q_target(self._normalize_obs(obs), act)
            else:
                return self.q_visit_target(self._normalize_obs(obs), self._normalize_obs(goal)[..., self.goal_idx], act)
        if goal is None:
            return self.q(self._normalize_obs(obs), act)
        else:
            return self.q_visit(self._normalize_obs(obs), self._normalize_obs(goal)[..., self.goal_idx], act)

    def reset(self, seed=None):
        self.lr_visit.reset()
        self.q_visit.reset(seed=seed)
        self.q_visit_target.reset(seed=seed)
        QNetwork.reset(self, seed=seed)

    def update(self, replay_memory, rng_generator=None, **kwargs):
        self.train()

        if rng_generator is None:
            rng_generator = self.rng_generator()

        q_stats = QNetwork.update(self, replay_memory, rng_generator, **kwargs)

        batch = replay_memory.get(
            batch_size=self.batch_size_visit,
            sequence_length=self.sequence_length_visit,
            rng_generator=rng_generator,
            priority_key="td_err_visit",
        )

        # Data is duplicated with positives and negatives, for a final shape of (B, T, k)
        relabeling_kwargs = {k: v for k, v in self.relabeling.items() if k != 'id'}
        obs, next_obs, goal, act, term, trunc, stepsize = getattr(src.goal_relabeling, self.relabeling.id)(
            batch, replay_memory, rng_generator, **relabeling_kwargs,
        )
        stepsize = stepsize * self.lr_visit.value
        T = obs.shape[1]

        if self.reward_visit_hard:
            rwd_visit = np.all(
                obs[..., self.goal_idx] == goal[..., self.goal_idx],
                axis=-1,
            ) * 1.0
            # The reward fires only if the state is EXACTLY the goal. This always
            # happens for positive samples in future HER. TD(λ) then makes positive
            # credit assignment easier.
        else:
            # Other GCRL environments/repos define a threshold for reaching goals
            # |s - g| < η, but this requires prior knowledge on the environment.
            # If you want to use the same bins of true counts, modify
            # `src.pseudocount.is_neighbor_binned` to support batch obs.
            # Another alternative is to use algorithm-specific mechanism, e.g.,
            # SUN's pseudocount radius, but this class is supposed to be generic.
            raise NotImplementedError

        # Note on rwd_visit. Other algorithms/repos use the negative squared
        # distance from the goal. Not only can it be unstable and depends on the
        # obs scale, but it also breaks the condition on terminal transition
        # "term = rwd_visit == 1", unless a threshold check |s - g | < η is used.

        obs = self._normalize_obs(obs)
        next_obs = self._normalize_obs(next_obs)
        goal = self._normalize_obs(goal)

        # Double DQN
        tolerance_max = 1.0 - self.gamma_visit
        q_visit_bound = 0.0 if self.negative_visit_reward else 1.0
        q_visit_next = self.q_visit(
            next_obs,
            goal[..., self.goal_idx],
            with_gradient=False,
        )
        next_a_max = random_argmax(
            q_visit_next,
            rng_generator=rng_generator,
            axis=-2,
            rtol=tolerance_max,
        )
        target_q_visit_next = np.take_along_axis(
            self.q_visit_target(
                next_obs,
                goal[..., self.goal_idx],
                with_gradient=False,
            ),
            next_a_max[..., None, :],
            axis=-2,
        ).squeeze(-2)
        target_q_visit_next = np.clip(target_q_visit_next, None, q_visit_bound)

        # By default, rwd_visit is 1 on hit and 0 otherwise.
        # This is used to check termination as well (first-hit termination).
        A = self.n_actions
        first_hit = rwd_visit > 0.0
        rwd_visit = np.repeat(rwd_visit[..., None], A, -1)
        mask = act[..., None] == np.arange(A)
        rwd_visit *= mask
        term_visit = rwd_visit == 1.0
        term_visit = np.logical_or(term[..., None], term_visit)

        # Truncate at first-hit for all action-goals: cuts TD(λ) backprop at the
        # state-goal step regardless of which action was taken, for explicit
        # negative/positive sub-sequence truncation.
        trunc_visit = np.logical_or(trunc, first_hit)
        trunc_visit = np.repeat(trunc_visit[..., None], A, -1)

        is_next_act_greedy = None
        if self.lmbda_visit > 0.0 and T > 1:
            is_next_act_greedy = is_act_greedy(
                q_visit_next[:, :-1],
                act[:, 1:],
                axis=-2,
                rtol=tolerance_max,
            )

        # Subtract -1 if rewards are encoded differently (-1/0 rather than 0/1)
        target = td_target(
            rwd_visit - self.negative_visit_reward * 1.0,
            term_visit,
            trunc_visit,
            is_next_act_greedy,
            target_q_visit_next,
            self.gamma_visit,
            self.lmbda_visit,
        )

        error_visit, gradient_norm_visit, q_visit = self.q_visit.update(
            obs,
            goal[..., self.goal_idx],
            act,
            target=target,
            stepsize=stepsize[..., None],  # broadcast to action dim
            rng_generator=rng_generator,
        )

        # Average the error over positives and negatives
        mean_error_visit = error_visit.mean(-1)

        # Update priorities in PER (will skip if no PER)
        replay_memory.post_sampling(batch["idx"], mean_error_visit, "td_err_visit")  # td_err is already squared

        return q_stats | {
            "td_err_visit": mean_error_visit,
            "grad_norm_visit": gradient_norm_visit,
            "q_visit_mean": q_visit.mean(),
        }

    def post_step(self, obs, act, *args, **kwargs):
        # lr_visit steps here, not in post_update, because on this branch every
        # schedule runs on the environment-step clock (see QCritic.post_step).
        QNetwork.post_step(self, obs, act, *args, **kwargs)
        self.lr_visit.step()

    def post_update(self):
        # QNetwork.post_update zeroes target_copy_counter exactly on the update
        # it copies q_target, so testing it for 0 afterwards keeps q_visit_target
        # on the same schedule without duplicating the counter arithmetic.
        QNetwork.post_update(self)
        if self.n_updates > 0 and self.target_copy_counter == 0:
            self.q_visit_target.copy_from(self.q_visit, self.tau)

    def train(self):
        QNetwork.train(self)
        self.q_visit.train()

    def eval(self):
        QNetwork.eval(self)
        self.q_visit.eval()

    def copy_from(self, source):
        QNetwork.copy_from(self, source)
        self.lr_visit.copy_from(source.lr_visit)
        self.gamma_visit = source.gamma_visit
        self.lmbda_visit = source.lmbda_visit
        self.relabeling = source.relabeling
        self.q_visit.copy_from(source.q_visit)
        self.q_visit_target.copy_from(source.q_visit_target)


class QEnsemble(Critic):
    """
    Ensemble of Q-critics. Each member has its own target network, i.e.,

        Q_1  Q_2  ...  Q_K
        ↓    ↓         ↓
        Qt_1 Qt_2 ...  Qt_K

    Each Qt_k is updated from its main Q_k.
    Each main Q_k is update with a different mini-batch, and its TD target is
    computed with a randomly assigned target network, i.e.,

        Q_1(s_t, a_t) target: r + γ max_a Qt_k(s_t, a) with k random
        same for all Q_k

    The attribute "aggregation" determines how the ensemble aggregates the values
    of the critics:
    - min: Q(s, a) = min_k Q_k(s, a)
    - max: Q(s, a) = max_k Q_k(s, a)
    - mean: Q(s, a) = mean Q_k(s, a)
    - random: Q(s, a) = Q_k(s, a) with k random
    """

    def __init__(
        self,
        obs_space: gymnasium.spaces.Box,
        act_space: gymnasium.spaces.Discrete,
        n_critics: int,
        critics_id: str,
        aggregation: str = "min",
        seed: int = None,
        **kwargs,
    ):
        self.n_critics = n_critics
        assert (
            aggregation in ("min", "max", "mean", "random")
        ), f"aggregation must be either min, max, mean, or random (got {aggregation})"
        self.aggregation = aggregation
        critic_cls = globals()[critics_id]
        self.critics = [
            critic_cls(
                obs_space,
                act_space,
                seed=cantor_pairing(seed, i) if seed is not None else None,
                **kwargs,
            )
            for i in range(n_critics)
        ]

    def __getattr__(self, name):
        # Proxy attribute lookups to the first critic so that callers can
        # access e.g. self.n_actions, self.gamma, self.visit_count transparently.
        #
        # __getattr__ is only called when normal lookup fails, so QEnsemble's
        # own attributes (critics, n_critics) are never intercepted.
        #
        # Use object.__getattribute__ to access critics: if critics is not yet
        # set (e.g. during deepcopy reconstruction), this raises AttributeError
        # cleanly instead of recursing back into __getattr__.
        critics = object.__getattribute__(self, 'critics')
        return getattr(critics[0], name)

    def __call__(self, *args, rng_generator=None, **kwargs):
        if rng_generator is None:
            rng_generator = self.rng_generator()

        if self.aggregation == "random":
            i = rng_generator.integers(self.n_critics)
            return self.critics[i](*args, *kwargs)

        values = np.stack([c(*args, **kwargs) for c in self.critics], axis=0)
        if self.aggregation == "min":
            return values.min(axis=0)
        elif self.aggregation == "max":
            return values.max(axis=0)
        elif self.aggregation == "mean":
            return values.mean(axis=0)

    def update(self, replay_memory, rng_generator=None, **kwargs):
        if rng_generator is None:
            rng_generator = self.rng_generator()

        # Randomly assign each critic a target network from the ensemble
        original_targets = [c.q_target for c in self.critics]
        assigned = rng_generator.integers(self.n_critics, size=self.n_critics)
        for i, critic in enumerate(self.critics):
            critic.q_target = original_targets[assigned[i]]

        # Update one random critic
        i = rng_generator.integers(self.n_critics)
        stats = self.critics[i].update(replay_memory, rng_generator=rng_generator, **kwargs)

        # Revert assigned target networks
        for i, critic in enumerate(self.critics):
            critic.q_target = original_targets[i]

        return stats

    def post_update(self):
        for c in self.critics:
            c.post_update()

    def post_step(self, *args, **kwargs):
        for c in self.critics:
            c.post_step(*args, **kwargs)

    def reset(self, seed=None):
        for c in self.critics:
            c.reset(seed=seed)

    def copy_from(self, source):
        self.n_critics = source.n_critics
        for c, sc in zip(self.critics, source.critics):
            c.copy_from(sc)

    def rng_generator(self, seed=None):
        return np.random.default_rng(seed=seed)

    def train(self):
        for c in self.critics:
            c.train()

    def eval(self):
        for c in self.critics:
            c.eval()


class SUNRISE(QEnsemble):
    """
    SUNRISE (Lee et al., ICML 2021): ensemble DQN with three additions
    on top of QEnsemble.

    - UCB acting: return `mean(Q) + kappa * std(Q)` across ensemble members
      (kappa=0 recovers plain mean-Q greedy behavior).
    - Weighted Bellman backup: pool one batch across all members; pick the
      next-state greedy action from `mean(Q_target)` and bootstrap with
      `min(Q_target)` (clipped double-Q).
    - Bootstrap masks + uncertainty-weighted loss: each member sees each
      transition with probability `mask_p` (Bernoulli), and every sample is
      re-weighted by `sigmoid(-temperature * std(max Q_target)) + 0.5` so
      transitions the ensemble already agrees on get down-weighted.
    """

    def __init__(
        self,
        *args,
        kappa: DictConfig,
        temperature: DictConfig,
        mask_p: float,
        **kwargs,
    ):
        """
        Args:
            kappa (DictConfig): schedule for the UCB coefficient in acting
                (λ in SUNRISE Eq. 7),
            temperature (DictConfig): schedule for the sigmoid temperature in
                the uncertainty-weighted loss (T in SUNRISE Eq. 6),
            mask_p (float): probability that a transition is included in
                each member's update (Bernoulli bootstrap mask).
        """

        super().__init__(*args, **kwargs)
        self.kappa = getattr(src.parameter, kappa.id)(**kappa)
        self.temperature = getattr(src.parameter, temperature.id)(**temperature)
        self.mask_p = mask_p

    def __call__(self, *args, **kwargs):
        values = np.stack([c(*args, **kwargs) for c in self.critics], axis=0)
        return values.mean(axis=0) + self.kappa.value * values.std(axis=0)

    def update(self, replay_memory, rng_generator=None, **kwargs):
        if rng_generator is None:
            rng_generator = self.rng_generator()

        c0 = self.critics[0]
        self.train()

        batch = replay_memory.get(
            batch_size=c0.batch_size,
            sequence_length=c0.sequence_length,
            rng_generator=rng_generator,
            priority_key="td_err",
        )
        obs, act, rwd, next_obs, term, trunc, weights = _unpack_batch(batch)

        obs = c0._normalize_obs(obs)
        next_obs = c0._normalize_obs(next_obs)

        if c0.clip_reward:
            rwd = np.clip(rwd, -1.0, 1.0)

        B, T = rwd.shape
        n = self.n_critics

        # (n_critics, B, T, n_actions) target-net evaluations at next_obs
        q_target_next = np.stack(
            [c.q_target(next_obs, with_gradient=False) for c in self.critics], axis=0
        )

        # Uncertainty weight from ensemble disagreement on max Q(next)
        # (SUNRISE Eq. 6/9): sigmoid(-T * std) + 0.5
        q_target_next_max_std = q_target_next.max(-1).std(0)  # (B, T)
        u_weights = 1.0 / (1.0 + np.exp(self.temperature.value * q_target_next_max_std)) + 0.5

        # Weighted Bellman backup: action from mean, value from min (clipped double-Q)
        q_target_next_mean = q_target_next.mean(0)  # (B, T, n_actions)
        next_a_max = random_argmax(q_target_next_mean, rng_generator=rng_generator, axis=-1)
        target_q_next = np.take_along_axis(
            q_target_next.min(0), next_a_max[..., None], axis=-1,
        ).squeeze(-1)  # (B, T)

        is_next_act_greedy = None
        if c0.lmbda > 0.0 and T > 1:
            is_next_act_greedy = is_act_greedy(
                q_target_next_mean[:, :-1], act[:, 1:], axis=-1,
            )
            trace_cut_frac = (1.0 - is_next_act_greedy).mean()
        else:
            trace_cut_frac = np.nan

        target = td_target(
            rwd,
            term,
            trunc,
            is_next_act_greedy,
            target_q_next,
            c0.gamma,
            c0.lmbda,
        )

        # Bootstrap mask, shape (n_critics, B, T)
        critic_mask = (rng_generator.random((n, B, T)) < self.mask_p).astype(np.float32)

        base_stepsize = weights * u_weights  # (B, T)

        errors, grad_norms, q_means = [], [], []
        for i, c in enumerate(self.critics):
            c.n_updates += 1
            stepsize = c.lr.value * base_stepsize * critic_mask[i]
            error, grad_norm, q = c.q.update(
                obs, act, target=target, stepsize=stepsize, rng_generator=rng_generator,
            )
            errors.append(error)
            grad_norms.append(grad_norm)
            q_means.append(q.mean())

        mean_error = np.mean(errors, axis=0)
        replay_memory.post_sampling(batch["idx"], mean_error, "td_err")

        return {
            "td_err": mean_error,
            "grad_norm": np.mean(grad_norms),
            "q_mean": np.mean(q_means),
            "q_std_next": q_target_next_max_std.mean(),
            "trace_cut_frac": trace_cut_frac,
        }

    def post_step(self, *args, **kwargs):
        super().post_step(*args, **kwargs)
        self.kappa.step()
        self.temperature.step()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.kappa.reset()
        self.temperature.reset()

    def copy_from(self, source):
        super().copy_from(source)
        self.kappa.copy_from(source.kappa)
        self.temperature.copy_from(source.temperature)
        self.mask_p = source.mask_p
