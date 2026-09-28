import numpy as np
from abc import ABC


class RunningStandardization(ABC):
    """
    Compute a running standardization of values according to Welford's online
    algorithm: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    """
    def __init__(self, shape, alpha=1e-32):
        """
        Constructor.

        Args:
            shape (tuple): shape of the data to standardize;
            alpha (float, 1e-32): minimum learning rate.
        """

        self._shape = shape

        assert 0.0 < alpha < 1.0
        self._alpha = alpha

        self._n = 1
        self._m = np.zeros(self._shape)
        self._s = np.ones(self._shape)

    def reset(self):
        """
        Reset the mean and standard deviation.
        """

        self._n = 1
        self._m = np.zeros(self._shape)
        self._s = np.ones(self._shape)

    def update(self, value):
        """
        Update the statistics with the current batch of data.
        The first dimension must be the number of samples in the batch.

        Args:
            value (Array): current data value to use for the update.
        """

        value = np.atleast_2d(value).reshape(-1, *self._shape)
        batch_size = len(value)
        self._n += batch_size
        alpha = max(batch_size / self._n, self._alpha)
        batch_mean = np.nanmean(value, 0, keepdims=True)
        new_m = (1 - alpha) * self._m + alpha * batch_mean
        new_s = self._s + (batch_mean - self._m) * (batch_mean - new_m)
        self._m, self._s = new_m, new_s

    def normalize(self, value):
        """
        (x - μ) / max(σ, 1)
        The standard deviation is clipped to 1 for numerical stability.

        """

        return (value - self.mean) / np.maximum(self.std, 1.0)

    @property
    def mean(self):
        """
        Returns:
            The estimated mean value.
        """

        return np.squeeze(self._m)

    @property
    def std(self):
        """
        Returns:
            The estimated standard deviation value.
        """

        return np.squeeze(np.sqrt(self._s / self._n))

    @property
    def n_samples(self):
        """
        Returns:
            The number of samples observed so far.
        """

        return self._n

    def copy_from(self, source):
        self._n = source._n
        self._m = source._m.copy()
        self._s = source._s.copy()
        self._shape = source._shape
        self._alpha = source._alpha


class StreamingQuantiles:
    def __init__(self, shape: tuple, lr: float = 0.01, min_init_samples: int = 100):
        """
        Tracks quantiles using stochastic approximation.

        Args:
            shape (tuple): shape of the input data,
            lr (float): learning rate (step size) for the estimates, as a
                fraction of the current interquartile range,
            min_init_samples (int): estimates stay uninitialized until this many
                samples have been seen, so that the first percentiles are not
                taken from a single observation,
        """

        self._shape = shape
        self._lr = lr
        self._min_init_samples = min_init_samples

        self._q25 = np.zeros(shape)
        self._q50 = np.zeros(shape)
        self._q75 = np.zeros(shape)
        self._buffer = []
        self._n_buffered = 0
        self._initialized = False

    def update(self, value):
        """
        Update estimates with a batch or single sample.

        Args:
            value (np.ndarray): input data (first dimension is the batch size),
        """

        value = np.atleast_2d(value).reshape(-1, *self._shape)

        if not self._initialized:
            self._buffer.append(value)
            self._n_buffered += value.shape[0]
            if self._n_buffered < self._min_init_samples:
                return
            stacked = np.concatenate(self._buffer, axis=0)
            self._q25 = np.nanpercentile(stacked, 25, axis=0)
            self._q50 = np.nanpercentile(stacked, 50, axis=0)
            self._q75 = np.nanpercentile(stacked, 75, axis=0)
            self._buffer = []
            self._n_buffered = 0
            self._initialized = True
            return

        # The bracket below is a probability, while the estimates carry the units
        # of the data, so the step is scaled by the current spread. Without this a
        # fixed lr moves a wide dimension far too slowly and a narrow one far too
        # fast.
        step = self._lr * np.maximum(self._q75 - self._q25, 1e-8)

        # Calculate what fraction of the batch is above the current estimates:
        # np.mean(value > q, axis=0) gives a value between 0 and 1

        # Q25: we want 75% of data to be above it.
        # If mean(above) > 0.75, estimate is too low, move UP.
        self._q25 += step * (np.mean(value > self._q25, axis=0) - 0.75)

        # Q50: we want 50% above
        self._q50 += step * (np.mean(value > self._q50, axis=0) - 0.50)

        # Q75: we want 25% above
        self._q75 += step * (np.mean(value > self._q75, axis=0) - 0.25)

    @property
    def iqr(self):
        return self._q75 - self._q25

    @property
    def q75(self):
        return self._q75

    @property
    def q25(self):
        return self._q25

    @property
    def q50(self):
        return self._q50

    @property
    def initialized(self):
        return self._initialized

    def reset(self):
        self._q25.fill(0)
        self._q50.fill(0)
        self._q75.fill(0)
        self._buffer = []
        self._n_buffered = 0
        self._initialized = False

    def copy_from(self, source):
        self._q25 = source._q25.copy()
        self._q50 = source._q50.copy()
        self._q75 = source._q75.copy()
        self._buffer = [b.copy() for b in source._buffer]
        self._n_buffered = source._n_buffered
        self._initialized = source._initialized
