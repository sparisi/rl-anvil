"""Aggregate metrics over runs and environments, with stratified-bootstrap
confidence intervals, after Agarwal et al., "Deep Reinforcement Learning at the
Edge of the Statistical Precipice" (NeurIPS 2021).

The definitions are those of the rliable library. A score array is
(..., runs, environments): every metric reduces the last two axes, so a leading
axis of bootstrap replicates, or of points along training, is computed in the
same call.

Environments may hold different numbers of runs. The runs of each environment
come first along the run axis and NaN pads it to the longest (see stack_runs);
every metric skips the padding, and the per-environment quantities -- the mean,
the median, the optimality gap, the profile and the probability of improvement
-- weigh each environment the same whatever its run count. On equal counts they
are rliable's numbers.
"""

import numpy as np

METRICS = ("Median", "IQM", "Mean", "Optimality Gap")

# Bootstrap replicates drawn and scored per call of the metric, so the memory a
# call takes is bounded however many replicates are asked for.
REPS_PER_CHUNK = 1000


def stack_runs(per_env):
    """Stack the runs of each environment into one array with the environments
    on a new last axis, as the metrics take it. `per_env` holds one array per
    environment with its runs on the first axis; an environment with fewer runs
    than the longest is padded with NaN after its own."""

    n = max(len(p) for p in per_env)
    padded = [
        np.concatenate(
            [
                np.asarray(p, dtype=float),
                np.full((n - len(p), *np.shape(p)[1:]), np.nan),
            ]
        )
        for p in per_env
    ]
    return np.stack(padded, axis=-1)


def flat_runs(scores):
    """The runs of every environment along one last axis."""

    return scores.reshape(*scores.shape[:-2], -1)


def env_means(scores):
    """The mean over each environment's runs, along a last environment axis."""

    return np.nanmean(scores, axis=-2)


def median(scores):
    """The median over the environments of the mean over each one's runs."""

    return np.median(env_means(scores), axis=-1)


def iqm(scores):
    """The interquartile mean: the mean of every run of every environment after
    the lowest and highest quarter are cut, as scipy.stats.trim_mean(x, 0.25).
    With fewer than four runs no run is cut and it is the mean of them."""

    runs = np.sort(flat_runs(scores), axis=-1)
    n = np.isfinite(runs).sum(axis=-1, keepdims=True)
    cut = (0.25 * n).astype(int)
    position = np.arange(runs.shape[-1])
    kept = (position >= cut) & (position < n - cut)
    return np.where(kept, runs, 0.0).sum(axis=-1) / kept.sum(axis=-1)


def mean(scores):
    """The mean over the environments of the mean over each one's runs."""

    return env_means(scores).mean(axis=-1)


def optimality_gap(scores, gamma=1.0):
    """How far the runs fall short of `gamma` on average, a run above it counting
    as reaching it, averaged per environment and then over them. Lower is
    better."""

    return gamma - env_means(np.minimum(scores, gamma)).mean(axis=-1)


def aggregate_metrics(scores):
    """The median, IQM, mean and optimality gap along a last axis, in METRICS
    order."""

    return np.stack(
        [median(scores), iqm(scores), mean(scores), optimality_gap(scores)],
        axis=-1,
    )


def performance_profile(scores, taus):
    """The fraction of runs scoring above each of `taus`, along a last axis:
    taken per environment and averaged over them."""

    above = (scores[..., None] > np.asarray(taus)).sum(axis=-3)
    runs = np.isfinite(scores).sum(axis=-2)[..., None]
    return (above / runs).mean(axis=-2)


def probability_of_improvement(x, y):
    """P(X > Y): in every environment, the fraction of (run of x, run of y) pairs
    where x scores higher, a tie counting one half -- the Mann-Whitney U
    statistic over the number of pairs -- averaged over the environments. `x`
    and `y` are (..., runs, environments) and may hold different numbers of
    runs."""

    diff = x[..., :, None, :] - y[..., None, :, :]
    wins = ((diff > 0) + 0.5 * (diff == 0)).sum(axis=(-3, -2))
    pairs = np.isfinite(x).sum(axis=-2) * np.isfinite(y).sum(axis=-2)
    return (wins / pairs).mean(axis=-1)


def resample(scores, reps, rng):
    """`reps` stratified bootstrap replicates of `scores`, stacked on a new first
    axis: each resamples the runs of every environment with replacement, from
    that environment's own runs, and leaves its padding where it is. A run is
    drawn whole, so axes before the run axis are carried along with it."""

    runs, envs = scores.shape[-2:]
    lead = tuple(range(scores.ndim - 2))
    counts = np.any(np.isfinite(scores), axis=lead).sum(axis=0)
    row = np.arange(runs)[:, None]
    drawn = (rng.random((reps, runs, envs)) * counts).astype(int)
    index = np.where(row < counts, drawn, row)
    index = index.reshape(reps, *([1] * (scores.ndim - 2)), runs, envs)
    return np.take_along_axis(scores[None], index, axis=-2)


def interval_estimates(fn, *scores, reps, seed=0):
    """`fn` of `scores` with its 95% percentile interval, as (estimate, low, high).

    Every score array is resampled separately (see resample), which with two of
    them is the stratified independent bootstrap the probability of improvement
    takes. The replicates are drawn and scored REPS_PER_CHUNK at a time. `seed`
    fixes them, so the same scores give the same interval every time."""

    rng = np.random.default_rng(seed)
    replicates = []
    for start in range(0, reps, REPS_PER_CHUNK):
        size = min(REPS_PER_CHUNK, reps - start)
        replicates.append(fn(*(resample(s, size, rng) for s in scores)))
    low, high = np.percentile(np.concatenate(replicates), [2.5, 97.5], axis=0)
    return fn(*scores), low, high
