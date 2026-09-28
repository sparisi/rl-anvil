import numpy as np
import random

def faster_choice(p: np.array, n: int, rng_generator: np.random.Generator):
    """
    Faster (for small arrays) implementation of np.random.choice.
    """

    cdf = np.cumsum(p)
    u = rng_generator.random(n)
    sampled_indices = np.searchsorted(cdf, u)
    return sampled_indices


# def faster_choice(p: np.array, n: int, rng_generator: np.random.Generator):
#     """
#     Faster (for small arrays) implementation of np.random.choice.
#     """
#
#     return (1 - rng_generator.random(n)[None] > p.cumsum()[..., None]).sum(0)


# https://stackoverflow.com/a/47722393/754136
def random_choice(p: np.array, rng_generator: np.random.Generator, axis: int = -1):
    """
    Vectorized np.random.choice (can be slow for large sizes).
    Slices that sum to 0 fall back to a uniform distribution.
    """

    p = np.where(p.sum(axis, keepdims=True) == 0, 1.0, p)
    p = p / p.sum(axis, keepdims=True)
    r = np.expand_dims(rng_generator.random(np.delete(p.shape, axis)), axis=axis)
    return (p.cumsum(axis=axis) > r).argmax(axis=axis)


def random_argmax(x: np.array, rng_generator: np.random.Generator, axis: int = None, rtol: float = 0.0):
    """
    Random tiebreak for np.argmax() if there are multiple entries with the same
    max value.
    It is possible to consider the max within some relative tolerance rtol < 1:
    x >= max(x) - |max(x)| * rtol
    """

    if axis is None:
        x_max = x.max()
        mask = x == x_max if rtol == 0.0 else x >= x_max - np.abs(x_max) * rtol
        best = np.argwhere(mask)
        i = rng_generator.choice(range(best.shape[0]))
        return best[i]

    x_max = x.max(axis=axis, keepdims=True)
    mask = x == x_max if rtol == 0.0 else x >= x_max - np.abs(x_max) * rtol
    return random_choice(mask, rng_generator, axis)


def random_argmin(x: np.array, rng_generator: np.random.Generator, axis: int = None, rtol: float = 0.0):
    """
    Random tiebreak for np.argmin() if there are multiple entries with the same
    min value.
    It is possible to consider the min within some relative tolerance rtol < 1:
    x <= min(x) + |min(x)| * rtol
    """

    if axis is None:
        x_min = x.min()
        mask = x == x_min if rtol == 0.0 else x <= x_min + np.abs(x_min) * rtol
        best = np.argwhere(mask)
        i = rng_generator.choice(range(best.shape[0]))
        return best[i]

    x_min = x.min(axis=axis, keepdims=True)
    mask = x == x_min if rtol == 0.0 else x <= x_min + np.abs(x_min) * rtol
    return random_choice(mask, rng_generator, axis)


# https://en.wikipedia.org/wiki/Pairing_function
def cantor_pairing(x: int, y: int) -> int:
    """
    Cantor pairing function to uniquely encode two
    natural numbers into a single natural number.
    Used for seeding.

    Args:
        x (int): first number,
        y (int): second number,

    Returns:
        A unique integer computed from x and y.
    """

    return int(0.5 * (x + y) * (x + y + 1) + y)


def set_rng_seed(seed: int = None) -> None:
    """
    Set random number generator seed across modules
    with random/stochastic computations.

    Args:
        seed (int)
    """

    np.random.seed(seed)
    random.seed(seed)


def dict_to_id(d: dict) -> str:
    """
    Parse a dictionary and generate a unique id.
    The id will have the initials of every key followed by its value.
    Entries are separated by underscore.
    If a key's value is None, it will be skipped.

    Example:
        d = {"first_key": 0, "some_key": True, "another_key": None} -> fk0_skTrue
    """

    def make_prefix(key: str) -> str:
        return "".join(w[0] for w in key.split("_"))

    return "_".join([f"{make_prefix(k)}{v}" for k, v in d.items() if v is not None])


def pprint_4d(arr):
    """
    Utility to pretty print 4D np.array.
    Example:

    >>> arr = np.random.rand(2, 3, 4, 2)
    >>> pprint_4d(arr)

        [[0.8   0.239 0.776 0.59      | 0.338 0.383 0.996 0.508]
         [0.149 0.004 0.963 0.645     | 0.974 0.349 0.314 0.686]
         [    -     -     -     -     |     -     -     -     -]
         [0.917 0.438 0.875 0.876     | 0.684 0.976 0.993 0.904]
         [0.463 0.661 0.835 0.023     | 0.555 0.609 0.69  0.47 ]
         [    -     -     -     -     |     -     -     -     -]
         [0.997 0.155 0.027 0.001     | 0.44  0.783 0.678 0.272]
         [0.164 0.5   0.369 0.734     | 0.526 0.692 0.138 0.614]]

    Outer rows are for the 2nd dimension.
    Outer columns are for the 4th dimension.
    Inner matrices are for dimensions (1, 3).
    """

    a, b, c, d = arr.shape
    tmp = arr.reshape(a * b, c * d, order="F")
    tmp = np.insert(tmp, np.arange(a, a * b, a), np.nan, 0)
    tmp = np.insert(tmp, np.arange(c, c * d, c), np.inf, 1)

    with np.printoptions(
        precision=3,
        suppress=True,
        threshold=np.inf,
        linewidth=np.inf,
        infstr="|",
        nanstr="-",
        edgeitems=1_000,
    ):
        print(tmp)


def mesh_combo(*args):
    """
    Simple meshgrid + flatten to get all combinations of many arrays.
    """

    return np.stack([i.flatten() for i in np.meshgrid(*args)]).T


def format_value(v) -> str:
    """
    Format one statistic for printing: integers as integers, floats with at most
    four decimals and no trailing zeros, everything else as str().
    """

    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:d}"
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def format_dict_row(
    d: dict,
    print_keys: bool = False,
    sorted: bool = False,
    col_width: int = None,
) -> tuple:
    """
    Return a string for a dictionary as a nicely aligned table row, optionally
    printing the header as well.
    Useful for printing a stream of statistics passed as dict.

    Args:
        d (dict): dictionary of values,
        print_keys (bool): if True, return the header row as well,
        sorted (bool): if True, sort keys alphabetically,
        col_width (int): width of each column. If None, it is set to the minimum
            width possible,

    Example:
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> for i in range(5):
    ...    results = {
    ...      "loss": rng.random() + (rng.random() < 0.5) * (rng.uniform(0.1, 1.0) * 1e5),
    ...      "updates": rng.integers(10),
    ...      "clipped": rng.random() < 0.5,
    ...    }
    ...    (header_str, row_str) = format_dict_row(results, print_keys=(i==0), width=14)
    ...    if header_str is not None:
    ...       print(header_str)  # colorize if you want
    ...    print(row_str)
    ...
              loss |        updates |        clipped
        87274.5867 |              0 |           True
            0.9756 |              6 |           True
        93409.2994 |              7 |          False
        59913.0742 |              6 |           True
            0.8276 |              7 |          False
    """

    keys = sorted(d.keys()) if sorted else list(d.keys())
    row = []
    col_width = [col_width] * len(keys)
    for i, k in enumerate(keys):
        s = format_value(d[k])
        if col_width[i] is None:
            col_width[i] = max(len(k), len(s))
        row.append(f"{s:>{col_width[i]}}")

    row_str = " | ".join(row)
    header = " | ".join(f"{k:>{w}}" for k, w in zip(keys, col_width)) if print_keys else None
    return (header, row_str)


class StatsTracker:
    """
    Simple class to keep tracks of statistics and return their mean.
    """

    def __init__(self):
        self.sums = {}
        self.counts = {}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.sums:
                self.sums[key] = 0.0
                self.counts[key] = 0
            self.sums[key] += np.asarray(value).mean()
            self.counts[key] += 1

    def reset(self):
        self.sums.clear()
        self.counts.clear()

    def get(self):
        return {
            key: self.sums[key] / self.counts[key]
            for key in self.sums
            if self.counts[key] > 0
        }

    def __str__(self):
        if not self.sums:
            return f"{self.__class__.__name__} (empty)"
        lines = [f"{k}: {v:.4f}" for k, v in self.get().items()]
        return "\n".join(lines)

    __repr__ = __str__  # optional: make repr match str


def sample_from_gym_space(space, n: int, rng_generator: np.random.Generator):
    """
    Batched version of gymnasium.spaces.Box.sample(): draws `n` samples using
    the same per-dimension distributions as the built-in sampler (uniform if
    both-sided bounded, shifted exponential if one-sided, normal if unbounded).
    Returns an array of shape (n,) + space.shape.
    """
    ub = ~space.bounded_below & ~space.bounded_above
    lb = space.bounded_below & ~space.bounded_above
    hb = ~space.bounded_below & space.bounded_above
    bb = space.bounded_below & space.bounded_above
    out = np.empty((n,) + space.shape, dtype=np.float64)
    out[:, ub] = rng_generator.normal(size=(n, ub.sum()))
    out[:, lb] = rng_generator.exponential(size=(n, lb.sum())) + space.low[lb]
    out[:, hb] = space.high[hb] - rng_generator.exponential(size=(n, hb.sum()))
    out[:, bb] = rng_generator.uniform(space.low[bb], space.high[bb], size=(n, bb.sum()))
    return out
