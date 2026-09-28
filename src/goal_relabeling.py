import numpy as np


def shuffle_observations(x, p, rng_generator):
    """
    Shuffle observations. For example say an observation has 3 elements [a, b, c].
    Given observations [a_i, b_i, c_i] with i = 1 ... N, every element has
    probability p to be from another observation, i.e.,
    [a_1, b_1, c_2]
    [a_2, b_1, c_3]
    ...
    Shuffling happens only across the batch dimension, not across the element dimension.
    That is, there cannot be a shuffled observation such as [b_3, a_2, c_1].
    """

    if p <= 0.0:
        return x

    B, _, d = x.shape  # (B, 1, d)
    mask = rng_generator.random((B, 1, d)) < p  # where to replace
    rand_idx = rng_generator.integers(0, B, size=(B, 1, d))  # random indices along last axis (same row)
    shuffled = np.take_along_axis(x, rand_idx, axis=0)  # gather values
    return np.where(mask, shuffled, x)  # apply replacement


def _random_goals(replay_memory, n, rng_generator):
    """
    `n` goals drawn uniformly at random from the goals stored in the replay
    memory -- the ones the actor acted on -- as an (n, 1, d) array.

    Steps acted without a goal (e.g., ε-greedy random action) are marked with
    `goal_valid=False`; those draws are replaced by a random observation from
    the whole memory.
    """

    goals = replay_memory.get(
        batch_size=n,
        sequence_length=1,
        rng_generator=rng_generator,
        keys=["goal_obs", "goal_valid"],
        priority_key=None,
    )
    random_obs = replay_memory.get(
        batch_size=n,
        sequence_length=1,
        rng_generator=rng_generator,
        keys=["obs"],
        priority_key=None,
    )["obs"]
    return np.where(goals["goal_valid"][..., None], goals["goal_obs"], random_obs)


def her_future(batch, replay_memory, rng_generator, K, *args, **kwargs):
    """
    The original HER `future` strategy assigns each timestep its own future goal,
    yielding T sub-sequences of varying length. This repository's critic, however,
    computes TD targets at all intermediate steps until the goal (T*T samples for
    a length-T trajectory), so sub-sequences must share a length to batch
    efficiently — making the per-timestep `future` strategy impractical.

    This function preserves the spirit of `future` while keeping a fixed sequence
    length:
      - Start at t = 0. Sample k_0 ~ Uniform[0, T). Assign g = s_{k_0} to all
        positions in [0, k_0].
      - Move to t = k_0 + 1. If past T-1, done. Otherwise sample
        k_1 ~ Uniform[k_0+1, T). Assign g = s_{k_1} to [k_0+1, k_1].
      - Repeat until all positions are covered.
      - Repeat K times.

    Example (T = 7):
      states at timestep:       0 1 2 3 4 5 6
      sample k_0 in [0, 6]: 2 → segment [0, 2] goal is 2
      sample k_1 in [3, 6]: 4 → segment [3, 4] goal is 4
      sample k_2 in [5, 6]: 5 → segment [5, 6] goal is 6
      goals per timestep:       2 2 2 4 4 6 6
    Repeat K times.

    On top of the K in-trajectory positives, K random negatives are drawn from
    the goals stored in the replay memory.
    """

    from src.critic import _unpack_batch
    obs, act, rwd, next_obs, term, trunc, weights = _unpack_batch(batch)
    B, T, d = obs.shape
    stepsize = np.ones((B, T)) * weights
    K_total = 2 * K  # K episode goals + K random goals

    # Build (B, T, K) array of segment anchors via a sequential scan over t.
    # `current_anchor[b, k]` holds the active anchor; `next_resample_at[b, k]`
    # is the first t at which (b, k) must draw a fresh anchor (init 0 → all
    # cells draw at t = 0).
    goal_idx         = np.zeros((B, T, K), dtype=np.int64)
    next_resample_at = np.zeros((B, K), dtype=np.int64)
    current_anchor   = np.zeros((B, K), dtype=np.int64)

    for t in range(T):
        needs_new = (next_resample_at == t)
        if needs_new.any():
            u    = rng_generator.random((B, K))
            span = float(T - t)
            new_anchor = (t + np.floor(u * span)).astype(np.int64)   # in [t, T)
            new_anchor = np.minimum(new_anchor, T - 1)               # safety
            current_anchor   = np.where(needs_new, new_anchor,     current_anchor)
            next_resample_at = np.where(needs_new, new_anchor + 1, next_resample_at)
        goal_idx[:, t, :] = current_anchor

    goal = obs[np.arange(B)[:, None, None], goal_idx, :]             # (B, T, K, d)

    # Index-based segment boundary for positives slice: positions where the
    # anchor changes between t and t+1. OR'd in critic with value-based.
    seg_end  = goal_idx[:, :-1, :] != goal_idx[:, 1:, :]             # (B, T-1, K)
    trunc    = np.broadcast_to(trunc[:, :, None], (B, T, K_total)).copy()
    trunc[:, :-1, :K] |= seg_end

    next_obs = np.broadcast_to(next_obs[:, :, None, :], (B, T, K_total, d))
    obs      = np.broadcast_to(obs[:, :, None, :],      (B, T, K_total, d))
    act      = np.broadcast_to(act[:, :, None],         (B, T, K_total))
    term     = np.broadcast_to(term[:, :, None],        (B, T, K_total))
    stepsize = np.broadcast_to(stepsize[:, :, None],    (B, T, K_total))

    neg_goals = _random_goals(replay_memory, B*K, rng_generator)
    neg_goals = neg_goals.reshape(B, K, d)
    neg_goals = np.broadcast_to(neg_goals[:, None, :, :], (B, T, K, d))

    goal = np.concatenate([goal, neg_goals], axis=2)  # (B, T, K_total, d)

    return obs, next_obs, goal, act, term, trunc, stepsize


def her_future_and_present(batch, replay_memory, rng_generator, K, *args, **kwargs):
    """
    Like `her_future`, but samples after the first goal take the current
    state as their goal. For positive goal index k (0 ≤ k < T):
      - positions t ≤ k: goal = s_k  (future goal; first-hit at t=k)
      - positions t > k: goal = s_t  (present goal; isolated rwd=1 terminal)

    Example (T = 7):
      states at timestep:  0 1 2 3 4 5 6
      sample g in [0, 6]:  2
      goals per timestep:  2 2 2 3 4 5 6
    Repeat K times.
    """

    from src.critic import _unpack_batch
    obs, act, rwd, next_obs, term, trunc, weights = _unpack_batch(batch)
    B, T, d = obs.shape
    stepsize = np.ones((B, T)) * weights
    K_total = 2 * K  # K episode goals + K random goals

    # Sample K distinct goal indices per episode (no repetitions), fixed across t.
    goal_idx   = np.argpartition(rng_generator.random((B, T)), K - 1, axis=-1)[:, None, :K]  # (B, 1, K)
    goal_idx_b = np.broadcast_to(goal_idx, (B, T, K))                # (B, T, K)

    # Original positive goal s_k and self-goal s_t.
    goal_pos  = obs[np.arange(B)[:, None, None], goal_idx_b, :]      # (B, T, K, d)
    goal_self = np.broadcast_to(obs[:, :, None, :], (B, T, K, d))    # (B, T, K, d)

    # For t > k, replace s_k with s_t (current state).
    t_idx     = np.arange(T)[None, :, None]                          # (1, T, 1)
    past_goal = t_idx > goal_idx_b                                   # (B, T, K)
    goal      = np.where(past_goal[..., None], goal_self, goal_pos)  # (B, T, K, d)

    # Mark segment boundaries with trunc=True so TD(λ) does not mix goals:
    # the effective goal changes at every t in [k_anchor, T-2] (positive→first
    # self-goal at t=k, then a new self-goal at every subsequent step).
    boundary = (t_idx >= goal_idx) & (t_idx < T - 1)                 # (B, T, K)
    trunc = np.broadcast_to(trunc[:, :, None], (B, T, K_total)).copy()
    trunc[..., :K] |= boundary

    next_obs = np.broadcast_to(next_obs[:, :, None, :], (B, T, K_total, d))
    obs      = np.broadcast_to(obs[:, :, None, :],      (B, T, K_total, d))
    act      = np.broadcast_to(act[:, :, None],         (B, T, K_total))
    term     = np.broadcast_to(term[:, :, None],        (B, T, K_total))
    stepsize = np.broadcast_to(stepsize[:, :, None],    (B, T, K_total))

    neg_goals = _random_goals(replay_memory, B*K, rng_generator)
    neg_goals = neg_goals.reshape(B, K, d)
    neg_goals = np.broadcast_to(neg_goals[:, None, :, :], (B, T, K, d))
    goal = np.concatenate([goal, neg_goals], axis=2)

    return obs, next_obs, goal, act, term, trunc, stepsize
