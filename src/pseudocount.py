"""
WARNING! These classes discretize the observation space to count agent visits.
However, very few RL environments have 2D observations that can be easily counted or displayed.
These classes define custom discretizations for environments with larger observation spaces,
mostly by counting the agent's Cartesian coordinates.
For example, in Acrobot, we transform [sin(theta_1), cos(theta_1), sin(theta_2), cos(theta_2)]
to [theta_1, theta_2] and DO NOT COUNT angular velocities.
In LunarLander, we count only the agent's (x, y) position.
"""

from abc import ABC
import gymnasium as gym
import numpy as np

from src.approximator import CountTable
from src.wrappers.gym_wrappers import get_wrapper_names


class GridworldCount(CountTable):
    """
    Version of CountTable for Gridworlds.

    The agent's cell arrives in one of two forms, and `one_hot` says which:
    the cell index itself, or a binary map of the grid where every element is 0
    except the one at the agent's position (MatrixWrapper), which is trivial to
    discretize with obs.argmax(). The two cannot be told apart from the array
    alone, since a batch of indices and a single map have the same shape.
    """

    def __init__(self, *shape, one_hot: bool = False, **kwargs):
        self._one_hot = one_hot
        super().__init__(*shape, **kwargs)

    def bin_index(self, obs):
        """
        Index of the agent's cell, which is the bin this counter counts in.
        """
        obs = np.asarray(obs)
        if not self._one_hot:
            return obs

        n = self.shape[0]
        d = obs.shape[-1]
        if d != n:
            raise ValueError(
                f"Observation of size {d} does not match the {n}-cell grid. "
                "GridworldCount only supports plain MatrixWrapper observations."
            )
        return obs.argmax(axis=-1)

    def is_neighbor(self, x, batch):
        """
        Boolean mask with True where batch[i] is in the same cell as x.
        Cells have no width to give a radius, so this is an exact match.
        """
        return self.bin_index(batch) == self.bin_index(x)

    def __call__(self, obs=None, act=None):
        if obs is None:
            return super().__call__()
        return super().__call__(self.bin_index(obs), act)

    def update(self, obs=None, act=None):
        return super().update(self.bin_index(obs), act)

    def bin_centers_raw(self):
        """
        One observation per cell, in the frame the environment emits: the
        inverse of `bin_index`. Either the (n_obs, 1) cell indices, or the
        (n_obs, n_obs) one-hot maps.
        """
        if not self._one_hot:
            return np.arange(self.shape[0])[:, None]
        return np.eye(self.shape[0])


class BinnedCount(CountTable):
    """
    Version of CountTable for continuous observations and discrete actions.
    Observations are discretized with bins, resulting in a table of shape
    (obs_shape[0] * n_bins[0], obs_shape[1] * n_bins[1], ..., n_actions).
    It is possible to convert observations before binning.

    The attribute map_shape is used by plotting scripts to display heatmaps.
    """

    def _convert_obs(self, obs):
        return obs

    def __init__(self, obs_low, obs_high, n_bins, n_actions, **kwargs):
        self.obs_dim = len(obs_high)
        if isinstance(n_bins, int):
            self.n_bins = np.full(self.obs_dim, n_bins)
        else:
            assert (
                len(n_bins) == self.obs_dim
            ), f"Number of bins {len(n_bins)} does not match observation shape {self.obs_dim}"
            self.n_bins = np.array(n_bins)
        self.n_actions = n_actions
        self.obs_low = np.array(obs_low)
        self.obs_high = np.array(obs_high)
        self.bin_widths = (self.obs_high - self.obs_low) / self.n_bins
        shape = tuple(self.n_bins) + (n_actions,)
        kwargs.setdefault("map_shape", tuple(self.n_bins))
        super().__init__(*shape, **kwargs)

    def _bin_obs(self, obs):
        obs = np.asarray(obs)
        if obs.ndim == 1:
            obs = obs[None, :]
        idx = np.floor((obs - self.obs_low) / self.bin_widths).astype(int)
        idx = np.clip(idx, 0, self.n_bins - 1)

        # Move the feature dimension to the front, preserving all batch dimensions.
        # For example, (B, T, obs_dim) -> (obs_dim, B, T).
        # When list() unpacks it, we feed obs_dim arrays of shape (B, T) into the table,
        # and counts will be returned with shape (B, T).
        return list(np.moveaxis(idx, -1, 0))

    def bin_index(self, obs):
        """
        Flat index of the bin every observation falls into, i.e. the per-dimension
        indices of `_bin_obs` raveled so that one integer identifies one bin.
        """
        return np.ravel_multi_index(
            self._bin_obs(self._convert_obs(obs)),
            tuple(self.n_bins),
        )

    def is_neighbor(self, x, batch):
        """
        Boolean mask with True where batch[i] is within `bin_widths` of x
        in every dimension, in the frame `_convert_obs` maps into.

        The bin widths are the radius, so the tolerance is the same one the counts
        are binned with, but centered on x instead of on fixed edges: two
        observations on opposite sides of an edge are neighbors here, while
        `bin_index` calls them different bins.
        """
        x = self._convert_obs(np.asarray(x, dtype=float))
        batch = self._convert_obs(np.asarray(batch, dtype=float))
        return np.all(np.abs(batch - x) <= self.bin_widths, axis=-1)

    def __call__(self, obs=None, act=None):
        if obs is None:
            return super().__call__()
        if act is not None:
            act = np.atleast_1d(act)
        return super().__call__(*self._bin_obs(self._convert_obs(obs)), act)

    def update(self, obs=None, act=None):
        if act is not None:
            act = np.atleast_1d(act)
        return super().update(*self._bin_obs(self._convert_obs(obs)), act)

    def bin_centers(self):
        """
        Coordinates of every bin's center in the binned frame, i.e., the space
        `_convert_obs` maps into (for this class, the observation itself).

        Bin k along axis i covers [low_i + k * w_i, low_i + (k + 1) * w_i), so its
        center is low_i + (k + 0.5) * w_i. The first and last bins are the exception:
        `_bin_obs` clips, so they also absorb everything below low_i and above high_i,
        and their center is not the center of what they actually capture.

        Bins are enumerated with `indexing="ij"`, so row i of the output is the bin
        whose count is `counter().sum(-1).ravel()[i]`: the returned coordinates and
        `counter().sum(-1)` are the same grid in the same order, one holding positions
        and the other counts.
        """
        axes = [
            self.obs_low[i] + (np.arange(self.n_bins[i]) + 0.5) * self.bin_widths[i]
            for i in range(self.obs_dim)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=-1)

    def bin_centers_raw(self):
        """
        The same bin centers as `bin_centers`, mapped back to the raw observation
        frame, i.e., what the environment emits and what a function approximator
        expects as input.

        In this base class, it returns the same bins as `bin_centers`, but subclasses
        overriding `_convert_obs` may drop or reshape dimensions, so the inverse
        cannot be derived automatically.
        """
        return self.bin_centers()


class BinnedCountPendulum(BinnedCount):
    """
    Convert [cos(theta), sin(theta)] to [theta].
    """

    def _convert_obs(self, obs):
        return np.stack(
            (np.arctan2(obs[..., 1], obs[..., 0]), obs[..., 2]),
            axis=-1
        )

    def __init__(self, obs_low, obs_high, n_bins, n_actions, **kwargs):
        BinnedCount.__init__(
            self,
            [-np.pi, obs_low[2]],
            [np.pi, obs_high[2]],
            n_bins,
            n_actions,
            **kwargs,
        )

    def bin_centers_raw(self):
        """
        Inverts `_convert_obs`: (theta, theta_dot) -> (cos(theta), sin(theta), theta_dot).
        The mapping is exact because [cos(theta), sin(theta)] <-> theta is a bijection
        between the unit circle and (-pi, pi], so nothing is lost either way.
        """
        c = self.bin_centers()  # (N, 2): (theta, theta_dot)
        theta, theta_dot = c[..., 0], c[..., 1]
        return np.stack([np.cos(theta), np.sin(theta), theta_dot], axis=-1)


class BinnedCountAcrobot(BinnedCount):
    """
    Convert [cos(theta), sin(theta)] to [theta], for both theta_1 and theta_2.
    Does not count other observation elements.
    """

    def _convert_obs(self, obs):
        return np.stack(
            (np.arctan2(obs[..., 1], obs[..., 0]), np.arctan2(obs[..., 3], obs[..., 2])),
            axis=-1
        )

    def __init__(self, obs_low, obs_high, n_bins, n_actions, **kwargs):
        BinnedCount.__init__(
            self,
            [-np.pi, -np.pi],
            [np.pi, np.pi],
            n_bins,
            n_actions,
            **kwargs,
        )

    def bin_centers_raw(self):
        c = self.bin_centers()  # (N, 2): (theta_1, theta_2)
        t1, t2 = c[..., 0], c[..., 1]
        return np.stack([np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2)], axis=-1)


class BinnedCountCartPole(BinnedCount):
    """
    Counts only (pos, theta).
    High/low bounds of the observation space are actually wrong. Even though the
    cart position in defined in [-4.8, 4.8], episodes end when the agent goes
    outside of [-2.4, 2.4].
    Similarly, the angle is defined in [-24°, 24°], but episodes end when the
    angles is outside of [-12°, 12°].
    https://gymnasium.farama.org/environments/classic_control/cart_pole/
    """

    def _convert_obs(self, obs):
        return obs[..., [0, 2]]

    def __init__(self, obs_low, obs_high, n_bins, n_actions, **kwargs):
        BinnedCount.__init__(
            self,
            obs_low[..., [0, 2]] / 2.0,
            obs_high[..., [0, 2]] / 2.0,
            n_bins,
            n_actions,
            **kwargs,
        )


class BinnedCountLunarLander(BinnedCount):
    """
    Count only the (x, y) coordinates.
    The observation space bounds are just "advisory"
    https://github.com/Farama-Foundation/Gymnasium/issues/377
    - (x) The episode terminates if x is outside of [-1, 1], even though the space
    bounds are [-2.5, 2.5].
    - (y) The goal is always at y = 0, but the terrain can have slopes. The lower
    bound is thus slightly below 0 (-0.59). No higher bound (the agent can fly
      higher than 2.5 without ending the episode).
    """

    def _convert_obs(self, obs):
        return obs[..., :2]

    def __init__(self, obs_low, obs_high, n_bins, n_actions, **kwargs):
        BinnedCount.__init__(
            self,
            [-1.0, -0.25],
            [1.0, 10.0],
            n_bins,
            n_actions,
            **kwargs,
        )


class BinnedCountLunarLanderFull(BinnedCount):
    """
    Count all 8 observation elements.
    Legs contact (dims 6, 7) are booleans, so they get exactly 2 bins; the six
    continuous dims get n_bins each.
    The table is therefore n_bins^6 * 4 * n_actions cells, so n_bins has to be
    chosen accordingly.
    """

    def __init__(self, obs_low, obs_high, n_bins, n_actions, **kwargs):
        BinnedCount.__init__(
            self,
            [-1.0, -0.25, obs_low[2], obs_low[3], obs_low[4], obs_low[5], 0.0, 0.0],
            [1.0, 10.0, obs_high[2], obs_high[3], obs_high[4], obs_high[5], 1.0, 1.0],
            [n_bins, n_bins, n_bins, n_bins, n_bins, n_bins, 2, 2],
            n_actions,
            **kwargs,
        )


# -----------------------------------------------------------------------------
# --- Functions ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def tabular_count(env, n_bins: int = 40):
    if len(env.observation_space.shape) > 1:
        return None

    # Envs not built by gymnasium.make (eg, OGBench) have no spec.
    env_id = getattr(getattr(env.unwrapped, "spec", None), "id", "") or ""
    n_actions = env.action_space.n

    # Count discrete observations directly, no need to bin. This comes before
    # low/high, which a Discrete space does not have.
    if isinstance(env.observation_space, gym.spaces.Discrete):
        if "Gym-Gridworlds" in env_id:
            return GridworldCount(
                env.unwrapped.grid.size,
                n_actions,
                map_shape=tuple(env.unwrapped.grid.shape),
            )
        return CountTable(
            int(env.observation_space.n),
            n_actions,
            map_shape=None,
        )

    low = env.observation_space.low
    high = env.observation_space.high
    if "Pendulum" in env_id:
        return BinnedCountPendulum(low, high, n_bins, n_actions)
    if "Acrobot" in env_id:
        return BinnedCountAcrobot(low, high, n_bins, n_actions)
    if "CartPole" in env_id:
        return BinnedCountCartPole(low, high, n_bins, n_actions)
    if "LunarLander" in env_id:
        return BinnedCountLunarLander(low, high, n_bins, n_actions)
    elif "Gym-Gridworlds" in env_id and "MatrixWrapper" in get_wrapper_names(env):
        return GridworldCount(
            env.unwrapped.grid.size,
            env.unwrapped.action_space.n,
            map_shape=tuple(env.unwrapped.grid.shape),
            one_hot=True,
        )
    elif isinstance(env.observation_space, gym.spaces.Box):
        if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
            if len(low) > 3:
                return None
            if "Gym-Gridworlds" in env_id:
                n_bins = env.unwrapped.grid.shape
            return BinnedCount(low, high, n_bins, n_actions)
        return None




SCALE_EPS = 1e-8
IQR_TO_STD = 1.349


def robust_scale(batch, rel_floor: float = 0.01):
    """Compute a per-dimension robust scale from batch and return it as a (D,) array.

    The scale is the interquartile range divided by 1.349, which matches the
    standard deviation for Gaussian data and is insensitive to the tails, so a
    few far excursions do not inflate the scale of a whole dimension.
    Dimensions whose scale falls below rel_floor times the median scale are
    raised to that floor, which keeps a near-constant feature from having its
    variation amplified; degenerate dimensions fall back to 1.
    """

    batch = np.asarray(batch)
    if batch.shape[0] == 0:
        return np.ones(batch.shape[-1], dtype=float)
    q75, q25 = np.percentile(
        batch,
        [75.0, 25.0],
        axis=0,
    )
    scale = (q75 - q25) / IQR_TO_STD
    positive = scale[scale > SCALE_EPS]
    if positive.size > 0 and rel_floor > 0.0:
        scale = np.maximum(scale, rel_floor * np.median(positive))
    return np.where(scale > SCALE_EPS, scale, 1.0)


def is_neighbor(x, batch, radius: float, scale = None):
    """Boolean mask, True where batch[i] falls within radius of x after normalization.
    scale is used directly if provided, otherwise computed from batch."""

    batch = np.asarray(batch)
    if batch.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if scale is None:
        scale = robust_scale(batch)
    scale = np.asarray(scale)
    if batch.dtype.kind == "f" and scale.dtype != batch.dtype:
        # Keep the temporaries at the observations' precision: scale comes back
        # as float64 from np.percentile, which would otherwise promote every
        # intermediate array in the expression below.
        scale = scale.astype(batch.dtype)
    dist2 = np.sum(((batch - x) / scale) ** 2, axis=-1)
    return dist2 <= radius ** 2


def is_neighbor_binned(x, batch, counter):
    """Boolean mask, True where batch[i] is within one bin of x.
    The binned counterpart of is_neighbor: the counter's bin widths are the radius.
    """

    batch = np.asarray(batch)
    if batch.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    return counter.is_neighbor(x, batch)


def pseudocount(x, batch, radius: float, scale=None):
    orig_shape = x.shape
    d = orig_shape[-1]

    x = np.asarray(x, dtype=float).reshape(-1, d)
    batch = np.asarray(batch, dtype=float)

    if batch.shape[0] == 0:
        return np.zeros(orig_shape[:-1], dtype=np.int64)

    if scale is None:
        scale = robust_scale(batch)
    x = x / scale
    batch = batch / scale

    x2 = np.sum(x**2, axis=1, keepdims=True)  # (Nx, 1)
    b2 = np.sum(batch**2, axis=1)  # (Nb,)

    # (Nx, Nb) using (a - b)^2 = a^2 + b^2 - 2ab
    dist2 = x2 + b2 - 2 * x @ batch.T
    np.maximum(dist2, 0, out=dist2)  # clamp floating-point negatives

    counts = (dist2 <= radius**2).sum(axis=1)

    return counts.reshape(orig_shape[:-1])


def pseudocount_prefixes(x, batch, radii, prefixes, scale=None):
    """Count the points of `batch` within each radius of each row of `x`, for
    every prefix of `batch`, and return an array of shape
    (len(radii), len(prefixes)) + x.shape[:-1].

    `pseudocount` over every (radius, prefix) pair recomputes the pairwise
    distances for each of them, and the distances depend on neither: a radius
    only thresholds them, and a prefix only decides how many columns are summed.
    Here they are computed once and reused, which is the whole saving -- the
    matrix product is the cost, and thresholding and summing it are not.

    `prefixes` is a number of rows of `batch`, not a fraction. They are walked in
    increasing order and accumulated, so the mask of a radius is summed once
    across all of them rather than once per prefix.

    The memory is the same as `pseudocount`'s over the full batch, since the
    distances to every point are held at once; feed `x` in chunks as before.
    """

    orig_shape = x.shape
    d = orig_shape[-1]
    out_shape = (len(radii), len(prefixes)) + orig_shape[:-1]

    x = np.asarray(x, dtype=float).reshape(-1, d)
    batch = np.asarray(batch, dtype=float)

    if batch.shape[0] == 0:
        return np.zeros(out_shape, dtype=np.int64)

    if scale is None:
        scale = robust_scale(batch)
    x = x / scale
    batch = batch / scale

    x2 = np.sum(x**2, axis=1, keepdims=True)  # (Nx, 1)
    b2 = np.sum(batch**2, axis=1)  # (Nb,)

    # (Nx, Nb) using (a - b)^2 = a^2 + b^2 - 2ab
    dist2 = x2 + b2 - 2 * x @ batch.T
    np.maximum(dist2, 0, out=dist2)  # clamp floating-point negatives

    bounds = [min(max(int(p), 0), batch.shape[0]) for p in prefixes]
    order = sorted(range(len(bounds)), key=lambda j: bounds[j])

    out = np.empty((len(radii), len(prefixes), x.shape[0]), dtype=np.int64)
    for i, radius in enumerate(radii):
        inside = dist2 <= radius**2
        start = 0
        running = np.zeros(x.shape[0], dtype=np.int64)
        for j in order:
            end = bounds[j]
            if end > start:
                # What this segment of the batch adds to the prefix before it.
                running = running + inside[:, start:end].sum(axis=1)
                start = end
            out[i, j] = running

    return out.reshape(out_shape)
