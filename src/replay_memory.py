import numpy as np
import src.parameter
from omegaconf import DictConfig

# This implementation is in pure NumPy to make it compatible with any deep learning
# library you'd like to use. The disadvantage is that you'll have to convert every
# batch to the desired format (e.g., you'll have to call `torch.from_numpy(obs)`
# before passing the observation to your neural network).
# To make it faster, convert data to whatever library you use (JAX or PyTorch)
# before storing it.

def zeros_resident(shape, dtype):
    """
    np.zeros(), but resident in RAM from the start.

    np.zeros() only reserves address space: the kernel maps one shared zero page
    for the whole array and allocates real pages lazily, on first write. Since
    add() writes one sample per env step, allocated memory would grow over time,
    and a run may not cause OOM until the end.
    With this function, all memory is allocated immediately --- if a job does not
    request enought memory it would crash immediately, not at the end.
    """

    arr = np.empty(shape, dtype=dtype)
    arr.fill(0)  # np.empty + fill, not np.zeros + fill, to memset only once
    return arr


class ReplayMemory(object):
    """
    Minimalistic replay memory.
    To initialize, call `memory.init(...)` with dummy data. For example,

        >>> mem = ReplayMemory(
                min_size=10,
                max_size=100,
            )
        >>> mem.init(
                state=np.zeros((3,)),
                action=np.zeros((1,), dtype=int),
            )

    This creates a replay memory for states of shape (3,) and actions of shape (1,).
    ReplayMemory uses the names of the argumets passed to init() to add and
    retrieve data. For example, to add samples to the memory, do

        >>> action = policy(state)
        >>> mem.add(state=state, action=action)

    And to get a batch, do

        >>> batch = mem.get(
                batch_size=32,
                sequence_length=10,
            )

    The batch will be a dictionary with keys "state" and "action". The first two
    dimensions of `batch["state"]` and `batch["action"]` are the mini-batch size and
    the length of the sequence. In the example above, their shape is (32, 10, ...).
    To use classic one-step estimator, use `sequence_length=1`.
    Longer sequences are needed for n-step estimators.
    """

    def __init__(
        self,
        min_size: int,
        max_size: int,
        **kwargs,
    ):
        """
        Args:
            min_size (int): the minimum number of samples to be collected before
                the memory is ready (often called "warm-up"),
            max_size (int): maximum number of samples stored in the memory,
        """

        assert (
            min_size <= max_size
        ), f"min_size {min_size} larger than max_size {max_size}"
        self._min_size = min_size
        self._max_size = max_size
        self.reset()

    def init(self, **kwargs):
        self.keys = list(kwargs.keys())
        for k, v in kwargs.items():
            setattr(self, k, zeros_resident((self._max_size, *v.shape), v.dtype))

    def add_keys(self, **kwargs):
        self.keys += list(kwargs.keys())
        for k, v in kwargs.items():
            setattr(self, k, zeros_resident((self._max_size, *v.shape), v.dtype))

    def export_data(self, filename: str):
        np.savez(
            filename,
            **{k: getattr(self, k)[:self.size] for k in self.keys},
        )

    def reset(self):
        self._idx = 0
        self._full = False
        self._tot_steps = 0

    def add(self, **kwargs):
        for k, v in kwargs.items():
            getattr(self, k)[self._idx] = v
        self._tot_steps += 1
        self._idx += 1
        if self._idx >= self._max_size:
            self._idx = 0
            self._full = True

    def get(
        self,
        batch_size: int,
        sequence_length: int,
        rng_generator: np.random.Generator = None,
        keys: list = None,
        **kwargs,
    ):
        if keys is None:
            keys = self.keys
        if rng_generator is None:
            rng_generator = self.rng_generator()

        idx = rng_generator.integers(self.size, size=batch_size)
        idx = idx[:, None] + np.arange(-sequence_length + 1, 1)
        idx = np.remainder(idx, self.size)
        batch = {k: getattr(self, k)[idx] for k in keys}

        # A sequence holding the newest sample before its last position continues
        # with the oldest sample, which is not its successor: mark truncation here.
        if "trunc" in batch:
            is_newest = idx == (self._idx - 1) % self.size
            batch["trunc"] = np.logical_or(batch["trunc"], is_newest)

        batch["idx"] = idx
        batch["priority_key"] = None
        return batch

    def get_sorted(
        self,
        batch_size: int,
        sequence_length: int,
        sort_key: str,
        order: str = "descending",
        keys: list = None,
        **kwargs,
    ):
        """
        Like `get`, but returns the top `batch_size` entries ranked by `sort_key`
        instead of a random sample. `order` is "descending" (highest first) or
        "ascending" (lowest first). If `sort_key` data is multi-dimensional per
        entry, it is reduced with max ("descending") or min ("ascending") across
        the trailing axes before sorting.
        For example, passing `order=ascending` and `sort_key=count` returns the
        least-visited samples.
        """

        if keys is None:
            keys = self.keys

        values = getattr(self, sort_key)[:self.size]

        if order == "descending":
            if values.ndim > 1:
                values = values.reshape(self.size, -1).max(axis=-1)
            if batch_size == 1:
                top_idx = np.array([np.argmax(values)])
            else:
                top_idx = np.argpartition(-values, batch_size - 1)[:batch_size]
                top_idx = top_idx[np.argsort(-values[top_idx])]
        elif order == "ascending":
            if values.ndim > 1:
                values = values.reshape(self.size, -1).min(axis=-1)
            if batch_size == 1:
                top_idx = np.array([np.argmin(values)])
            else:
                top_idx = np.argpartition(values, batch_size - 1)[:batch_size]
                top_idx = top_idx[np.argsort(values[top_idx])]
        else:
            raise ValueError(
                f"order must be 'ascending' or 'descending', got {order!r}"
            )

        idx = top_idx[:, None] + np.arange(-sequence_length + 1, 1)
        idx = np.remainder(idx, self.size)
        batch = {k: getattr(self, k)[idx] for k in keys}

        # A sequence holding the newest sample before its last position continues
        # with the oldest sample, which is not its successor: mark truncation here.
        if "trunc" in batch:
            is_newest = idx == (self._idx - 1) % self.size
            batch["trunc"] = np.logical_or(batch["trunc"], is_newest)

        batch["idx"] = idx
        batch["priority_key"] = None
        return batch

    def post_sampling(self, *args, **kwargs):
        pass

    def set_values(self, idx, **kwargs):
        """
        Overwrite the stored values of the given keys at the indices returned in a
        batch by `get`.
        """

        for k, v in kwargs.items():
            getattr(self, k)[idx] = v

    def post_update_round(self, *args, **kwargs):
        pass

    def post_step(self, *args, **kwargs):
        pass

    def rng_generator(self, seed=None):
        return np.random.default_rng(seed=seed)

    @property
    def is_ready(self):
        return self.size >= self._min_size

    @property
    def full(self):
        return self._full

    @property
    def tot_steps(self):
        return self._tot_steps

    @property
    def size(self):
        return self._idx if not self._full else self._max_size

    @property
    def max_size(self):
        return self._max_size

    def _random_fill(self):
        """
        Used for debugging.
        """
        rng_generator = np.random.default_rng(42)
        for k in self.keys:
            getattr(self, k)[:] = rng_generator.random(
                getattr(self, k).shape,
            ).astype(getattr(self, k).dtype)
        self._full = True


# ------------------------------------------------------------------------------
# Prioritized Experience Replay
# ------------------------------------------------------------------------------

class SumTree:
    """
    Sum tree over `capacity` priorities with batched sampling and updates.

    Leaves are padded to the next power of two so that all of them sit at the
    same depth: a batch of values then descends the tree in lockstep, one level
    per iteration, and a batch of updates writes its leaves and recomputes
    their ancestors level by level from the children, so internal sums never
    drift.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.n_leaves = 1 << (capacity - 1).bit_length()
        self.depth = self.n_leaves.bit_length() - 1
        self.tree = zeros_resident(2 * self.n_leaves - 1, np.float64)
        self._first_leaf = self.n_leaves - 1

    def total(self):
        return self.tree[0]

    def get(self, idx):
        """Return the priorities of the leaves at data indices `idx`."""

        return self.tree[self._first_leaf + np.asarray(idx)]

    def update(self, idx, priorities):
        """
        Set the priorities of the leaves at data indices `idx` and recompute
        their ancestors. If an index is repeated, its last priority is kept.
        """

        idx = np.asarray(idx).ravel()[::-1]
        priorities = np.asarray(priorities, dtype=np.float64).ravel()[::-1]
        idx, last = np.unique(idx, return_index=True)
        nodes = self._first_leaf + idx
        self.tree[nodes] = priorities[last]
        for _ in range(self.depth):
            nodes = (nodes - 1) // 2
            self.tree[nodes] = self.tree[2 * nodes + 1] + self.tree[2 * nodes + 2]

    def sample(self, values):
        """
        Return the data indices of the leaves that `values`, in [0, total),
        fall into: leaf i is chosen by the values in (sum_{j<i} p_j, sum_{j<=i} p_j],
        and leaf 0 also by the value 0.
        """

        values = np.minimum(
            np.asarray(values, dtype=np.float64),
            np.nextafter(self.total(), 0.0),
        )
        nodes = np.zeros(len(values), dtype=np.int64)
        for _ in range(self.depth):
            left = 2 * nodes + 1
            left_sum = self.tree[left]
            go_left = values <= left_sum
            nodes = np.where(go_left, left, left + 1)
            values = np.where(go_left, values, values - left_sum)
        return nodes - self._first_leaf

    def _random_fill(self, min_p=0.01, max_p=1.0):
        rng_generator = np.random.default_rng(42)
        self.update(
            np.arange(self.capacity),
            rng_generator.uniform(min_p, max_p, size=self.capacity),
        )


class PrioritizedReplayMemory(ReplayMemory):
    """
    This memory supports Prioritized Experience Replay (PER), i.e., sampling is
    not uniform but depends on "prioritites". All priorities are set to an initial
    default max value, and then updated after the sample is used for training.
    For example, in the original PER paper, priorities are proportional to the
    TD error of the sample.
    Priorities are stored in a SumTree structure for faster sampling.
    SumTree is batched: sampling, adding samples and updating priorities are
    vectorized over the batch.
    The "idx" of a batch holds data indices, as in ReplayMemory, and
    `post_sampling` takes the same indices.
    It is possible to keep multiple SumTree and sample data according to different
    priorities.
    You don't need to define immediately what priorities you will use: if a new
    priority key is passed to the memory, a new SumTree will be created with default
    priorites.

    For example, if you want to sample batches according to both the TD error,

        >>> mem = PrioritizedReplayMemory(
                min_size=10,
                max_size=100,
            )
        >>> mem.init(
                state=np.zeros((3,)),
                action=np.zeros((1,), dtype=int),
            )
        >>> action = policy(state)
        >>> mem.add(state=state, action=action)

    And to get a batch, do

        >>> batch = mem.get(
                batch_size=32,
                sequence_length=10,
                priority_key="td_error",
            )
        >>> td_error = q.update(...)
        >>> mem.post_sampling(batch["idx"], td_error, "td_error")

    Then, you can sample again using anothe priority, e.g.,

        >>> batch = mem.get(
                batch_size=32,
                sequence_length=10,
                priority_key="rarity",
            )
        >>> n = visit_count(batch["obs"], batch"[act"])
        >>> mem.post_sampling(batch["idx"], 1.0 / n, "rarity")

    When sampling n-step sequences, priorities are used to sample the LAST step of
    a sequence. For example, let's say we want to sample sequences of 10 steps.
    If the element of index 12 (in the memory) has high priority and is sampled,
    the mini-batch will return elements of index [3 ... 12].
    This is to give importance to sequences LEADING TO sample 12, rather than
    sequences STARTING FROM sample 12 (i.e., we care about HOW the agents reaches
    sample 12).
    """

    def __init__(
        self,
        alpha: DictConfig,
        beta: DictConfig,
        **kwargs,
    ):
        """
        Args:
            alpha (DictConfig): configuration for the parameter that regulates
                priorities (0 → no priority, 1 → full priority),
            beta (DictConfig): configuration for the parameter that regulates
                importance sampling correction (0 → no correction, 1 → full correction),

        The original PER paper linearly increases beta to 1, and keeps alpha
        constant. See its Section 3.4.
        """

        super().__init__(**kwargs)
        self.alpha = getattr(src.parameter, alpha.id)(**alpha)
        self.beta = getattr(src.parameter, beta.id)(**beta)
        self.trees = {}  # Dictionary to store multiple SumTrees
        self._default_priority = 1.0
        self._pending_idx = []
        self._pending_priority = []

    def _create_tree(self, key: str):
        """Initialize a new SumTree and fill it with current default priorities."""

        tree = SumTree(self._max_size)
        initial_p = self._default_priority ** self.alpha.value
        tree.update(np.arange(self.size), np.full(self.size, initial_p))
        self.trees[key] = tree
        return tree

    def _flush(self):
        """Write the priorities of the samples added since the last flush to every tree."""

        if not self._pending_idx:
            return
        idx = np.array(self._pending_idx)
        priorities = np.array(self._pending_priority)
        self._pending_idx = []
        self._pending_priority = []
        for tree in self.trees.values():
            tree.update(idx, priorities)

    def add(self, **kwargs):
        self._pending_idx.append(self._idx)
        self._pending_priority.append(self._default_priority ** self.alpha.value)
        super().add(**kwargs)

    def get(
        self,
        batch_size: int,
        sequence_length: int,
        rng_generator: np.random.Generator = None,
        keys: list = None,
        priority_key: str = None,
        **kwargs,
    ):
        self._flush()
        if priority_key is None:
            return ReplayMemory.get(self, batch_size, sequence_length, rng_generator, keys)

        if priority_key not in self.trees:
            self._create_tree(priority_key)
        active_tree = self.trees[priority_key]

        if keys is None:
            keys = self.keys
        if rng_generator is None:
            rng_generator = self.rng_generator()

        # Stratified sampling: one value in each of batch_size segments of equal mass
        segment = active_tree.total() / batch_size
        values = (np.arange(batch_size) + rng_generator.random(batch_size)) * segment
        data_idxs = np.minimum(active_tree.sample(values), self.size - 1)
        priorities = active_tree.get(data_idxs)

        idx = data_idxs[:, None] + np.arange(-sequence_length + 1, 1)
        idx = np.remainder(idx, self.size)
        batch = {k: getattr(self, k)[idx] for k in keys}

        # A sequence holding the newest sample before its last position continues
        # with the oldest sample, which is not its successor: mark truncation here.
        if "trunc" in batch:
            is_newest = idx == (self._idx - 1) % self.size
            batch["trunc"] = np.logical_or(batch["trunc"], is_newest)

        # Importance sampling weights computed ONLY from sampled endpoints
        sampling_probabilities = priorities / (active_tree.total() + 1e-8)
        sampling_probabilities = np.clip(sampling_probabilities, 1e-12, None)
        weights = (self.size * sampling_probabilities) ** (-self.beta.value)
        max_w = np.max(weights)
        if max_w > 0 and np.isfinite(max_w):
            weights /= max_w
        else:
            weights = np.ones_like(weights)

        # Broadcast weights over sequence dimension
        batch["weights"] = weights[:, None]
        # Data indices for ALL (B, T) samples, not just endpoints, so
        # post_sampling can update each sample's priority.
        batch["idx"] = idx
        batch["priority_key"] = priority_key
        return batch

    def update_priorities(self, idx, new_priorities, priority_key):
        self._flush()
        if priority_key not in self.trees:
            self._create_tree(priority_key)
        tree = self.trees[priority_key]
        flat_priorities = np.asarray(new_priorities, dtype=np.float64).ravel()
        flat_priorities = np.nan_to_num(flat_priorities, nan=0.0, posinf=1e6)
        self._default_priority = max(self._default_priority, float(flat_priorities.max()))
        tree.update(np.asarray(idx).ravel(), (flat_priorities + 1e-5) ** self.alpha.value)

    def post_sampling(self, idx, new_priorities, priority_key, **kwargs):
        if priority_key is None:
            return
        self.update_priorities(
            idx,
            np.clip(new_priorities, 0, None) + 1e-8,
            priority_key,
        )

    def post_step(self, *args, **kwargs):
        self.alpha.step()
        self.beta.step()

    def _random_fill(self):
        super()._random_fill()
        self._pending_idx = []
        self._pending_priority = []
        for tree in self.trees.values():
            tree._random_fill()


# ------------------------------------------------------------------------------
# No Memory
# ------------------------------------------------------------------------------

class OnlineMemory(ReplayMemory):
    """
    Replay memory that samples only the transitions collected since the last
    round of updates (marked by calling `post_update_round`), but retains full
    history (to save it and use it for plotting and debugging).

    Sampling always returns a batch of shape (1, T, ...), where T are the steps
    since the last round of updates.

    Configure `max_size >= update_frequency` so the buffer can hold a
    full inter-update segment; otherwise the ring overflows between updates.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_update_step = 0

    def reset(self):
        super().reset()
        self._last_update_step = 0

    def get(self, keys: list = None, **kwargs):
        if keys is None:
            keys = self.keys

        n_new = self._tot_steps - self._last_update_step
        assert n_new <= self._max_size, (
            f"Replay memory overflowed: {n_new} new steps since last "
            f"update > max_size {self._max_size}. Increase max_size to at "
            f"least update_frequency so the segment since the last update is "
            f"not overwritten."
        )

        # Walk forward from the last update position, wrapping modulo max_size.
        # Handles the no-wrap, wrap, and n_new == 0 cases in one expression.
        start = (self._idx - n_new) % self._max_size
        idx = (start + np.arange(n_new)) % self._max_size
        batch = {k: getattr(self, k)[idx][None, ...] for k in keys}
        batch["idx"] = idx[None, :]
        batch["priority_key"] = None
        return batch

    def post_update_round(self, *args, **kwargs):
        self._last_update_step = self._tot_steps
