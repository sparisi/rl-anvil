import os
import zipfile
import numpy as np
import src.parameter
from omegaconf import DictConfig

from src.pseudocount import SCALE_EPS, is_neighbor, pseudocount, robust_scale, scaled_dist2
from src.utils.misc import atomic_open

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
    request enough memory it would crash immediately, not at the end.
    """

    arr = np.empty(shape, dtype=dtype)
    arr.fill(0)  # np.empty + fill, not np.zeros + fill, to memset only once
    return arr


def npy_header(npz, key: str) -> tuple:
    """Return (shape, dtype) of the array `key` of the open npz file `npz`, read
    from its .npy header without loading the array."""

    with npz.zip.open(f"{key}.npy") as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(member)
        else:
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(member)
    return shape, dtype


def merge_chunks(chunk_paths: list, path: str):
    """Concatenate the npz files `chunk_paths` key by key, in the order given,
    into the npz file `path` (atomic_open), then delete the chunks. A single
    chunk is renamed to `path` (os.replace).

    `path` is written as np.savez writes an npz, an uncompressed zip with one
    .npy entry per key. Each entry gets a header for the total number of rows,
    read from the chunks' headers, and the chunks' rows are appended one chunk
    at a time, so one chunk's array of one key is the most held in memory.
    Chunks that disagree on a key's dtype or trailing shape raise a ValueError."""

    if len(chunk_paths) == 1:
        os.replace(chunk_paths[0], path)
        return

    files = [np.load(chunk) for chunk in chunk_paths]
    try:
        with atomic_open(path) as f:
            with zipfile.ZipFile(f, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
                for key in files[0].files if files else []:
                    headers = [npy_header(npz, key) for npz in files]
                    shape, dtype = headers[0]
                    if any((s[1:], d) != (shape[1:], dtype) for s, d in headers):
                        raise ValueError(
                            f"{key}: the chunks disagree on dtype or trailing shape "
                            f"{headers}"
                        )
                    header = {
                        "descr": np.lib.format.dtype_to_descr(dtype),
                        "fortran_order": False,
                        "shape": (sum(s[0] for s, _ in headers), *shape[1:]),
                    }
                    with zf.open(f"{key}.npy", "w", force_zip64=True) as out:
                        np.lib.format.write_array_header_2_0(out, header)
                        for npz in files:
                            out.write(np.ascontiguousarray(npz[key]).data)
    finally:
        for npz in files:
            npz.close()
    for chunk in chunk_paths:
        os.remove(chunk)


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
    ReplayMemory uses the names of the arguments passed to init() to add and
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
        rho: float = None,
        scale_check_every: int = 1000,
        scale_tol: float = 0.01,
        scale_max_samples: int = None,
        scale_rel_floor: float = 0.01,
        **kwargs,
    ):
        """
        Args:
            min_size (int): the minimum number of samples to be collected before
                the memory is ready (often called "warm-up"),
            max_size (int): maximum number of samples stored in the memory,
            rho (float): radius for pseudocount. When a new observation is
                inserted, neighbors within this radius (in standardized space)
                are used for pseudocounts. Only needed after init_counting().
                None selects the environment's true binned visit counts instead,
                read from the counter given to init_counting(), and no
                pseudocount is maintained,
            scale_check_every (int): number of insertions between two checks of
                the pseudocount scale,
            scale_tol (float): the scale is frozen once its largest relative
                per-dimension change between two checks falls below this value,
            scale_max_samples (int): number of stored samples past which the
                scale is frozen regardless of whether it has stabilized. It also
                bounds the cost of the forced count rebuild, which is quadratic
                in the number of stored samples. If None, the scale is frozen
                only once it stabilizes within scale_tol,
            scale_rel_floor (float): per-dimension scales below this fraction of
                the median scale are raised to it, to keep a near-constant
                feature from being amplified.
        """

        assert (
            min_size <= max_size
        ), f"min_size {min_size} larger than max_size {max_size}"
        self._min_size = min_size
        self._max_size = max_size
        self.rho = rho
        self._scale_check_every = scale_check_every
        self._scale_tol = scale_tol
        self._scale_max_samples = scale_max_samples
        self._scale_rel_floor = scale_rel_floor
        self.reset()

    def init(self, **kwargs):
        self.keys = list(kwargs.keys())
        for k, v in kwargs.items():
            setattr(self, k, zeros_resident((self._max_size, *v.shape), v.dtype))

    def add_keys(self, **kwargs):
        self.keys += list(kwargs.keys())
        for k, v in kwargs.items():
            setattr(self, k, zeros_resident((self._max_size, *v.shape), v.dtype))

    def save_chunk(self, path: str):
        """Write the samples added since the memory last wrapped to the npz file
        `path`, i.e., samples in [0, idx)."""

        end = self._idx if self._idx > 0 else self.size
        np.savez(path, **{k: getattr(self, k)[:end] for k in self.keys})

    def init_counting(self, n_actions: int, goal_idx, counter = None):
        assert not (self.rho is None and counter is None), (
            "rho=None asks for true counts, which need a binned counter, but this "
            "environment has none"
        )
        self.keys += ["count"]
        self.count = zeros_resident((self._max_size, n_actions), np.int32)
        self.goal_idx = goal_idx
        self.counter = counter
        if self.rho is None:
            # Bin of every stored entry, kept so that the entries sharing a bin can
            # be found without binning the stored observations again.
            self.bin_idx = zeros_resident((self._max_size,), np.int64)
        self._reset_count_scale()

    @property
    def count_scale(self):
        """Return the per-dimension scale the stored pseudocounts are computed in.
        """

        return self._count_scale

    @property
    def count_scale_frozen(self):
        """Return whether the pseudocount scale has stopped changing."""

        return self._scale_frozen

    def _reset_count_scale(self):
        """Clear the pseudocount scale and the bookkeeping that decides when to freeze it."""

        self._count_scale = None
        self._counts_initialized = False
        self._scale_frozen = False
        self._since_scale_check = 0

    def reset(self):
        self._idx = 0
        self._full = False
        self._tot_steps = 0
        if hasattr(self, "goal_idx"):
            self._reset_count_scale()
            self.count.fill(0)
            if self.rho is None:
                self.bin_idx.fill(0)

    def add(self, **kwargs):
        write_idx = self._idx

        # Read evicted entry before overwriting (needed for count maintenance when full)
        evicted = None
        if hasattr(self, "goal_idx") and self._full:
            evicted = (
                self.obs[write_idx][self.goal_idx].copy(),
                int(np.asarray(self.act[write_idx]).flat[0]),
            )

        for k, v in kwargs.items():
            getattr(self, k)[write_idx] = v

        if hasattr(self, "goal_idx"):
            if self.rho is None:
                # The counter already includes this step, so every other stored entry
                # in the same bin is one visit behind on this action. The counter
                # never decrements, so eviction needs no correction.
                new_bin = int(np.ravel(self.counter.bin_index(kwargs["obs"]))[0])
                new_act = int(np.asarray(kwargs["act"]).flat[0])
                n = self._max_size if self._full else write_idx
                self.count[:n][self.bin_idx[:n] == new_bin, new_act] += 1
                self.bin_idx[write_idx] = new_bin
                self.count[write_idx] = self.counter(kwargs["obs"])
            else:
                self._update_counts(write_idx, kwargs, evicted)

        self._tot_steps += 1
        self._idx += 1
        if self._idx >= self._max_size:
            self._idx = 0
            self._full = True

    def _stored_count_data(self, write_idx):
        """Return the stored goal-space observations and actions as a (N, D) and a (N,) array.

        Both include the entry just written at write_idx, which add() has already
        stored by the time counts are updated.
        """

        n = self._max_size if self._full else write_idx + 1
        obs = np.asarray(self.obs[:n, ..., self.goal_idx])
        return obs.reshape(n, -1), self.act[:n].ravel()

    def _maybe_freeze_scale(self, write_idx, rebuild: bool = True):
        """Refresh the pseudocount scale on a fixed cadence and freeze it once it stabilizes.

        Returns True if the scale was frozen on this call. If rebuild is True,
        every stored count has then been rebuilt and no incremental update is
        needed; if it is False, the stored counts are left untouched.
        """

        if self._scale_frozen:
            return False
        self._since_scale_check += 1
        if self._since_scale_check < self._scale_check_every:
            return False
        self._since_scale_check = 0

        obs, _ = self._stored_count_data(write_idx)
        n = obs.shape[0]
        previous = self._count_scale
        self._count_scale = robust_scale(obs, rel_floor=self._scale_rel_floor)
        if previous is not None:
            change = np.max(
                np.abs(self._count_scale - previous)
                / np.maximum(previous, SCALE_EPS)
            )
            if change < self._scale_tol:
                self._freeze_scale(write_idx, rebuild=rebuild)
                return True
        if self._scale_max_samples is not None and n >= self._scale_max_samples:
            self._freeze_scale(write_idx, rebuild=rebuild)
            return True
        return False

    def _init_count_scale(self, write_idx):
        """Compute the pseudocount scale from the samples stored so far."""

        obs, _ = self._stored_count_data(write_idx)
        self._count_scale = robust_scale(obs, rel_floor=self._scale_rel_floor)

    def _init_scale_and_counts(self, write_idx):
        """Compute the pseudocount scale from the warm-up data and build every count from it."""

        self._init_count_scale(write_idx)
        self._counts_initialized = True
        self._rebuild_counts(write_idx)

    def _freeze_scale(self, write_idx, rebuild: bool = True):
        """Mark the pseudocount scale as final and, if rebuild is True, recompute every count."""

        self._scale_frozen = True
        if rebuild:
            self._rebuild_counts(write_idx)

    def _rebuild_counts(self, write_idx):
        """Recompute every stored count from scratch under the current scale."""

        obs, act = self._stored_count_data(write_idx)
        n = obs.shape[0]
        if n == 0:
            return
        n_actions = self.count.shape[-1]
        onehot = np.zeros((n, n_actions), dtype=float)
        onehot[np.arange(n), act] = 1.0

        radius2 = self.rho**2
        # Chunk the pairwise distances so the temporaries stay bounded regardless
        # of how many samples are stored when the scale freezes.
        chunk = max(1, int(2**22) // n)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            dist2 = scaled_dist2(obs[start:stop], obs, self._count_scale)
            mask = (dist2 <= radius2).astype(float)
            self.count[start:stop] = (mask @ onehot).astype(self.count.dtype)

    def _update_counts(self, write_idx, kwargs, evicted):
        n_actions = self.count.shape[-1]
        new_obs = np.asarray(kwargs["obs"])[self.goal_idx]
        new_act = int(np.asarray(kwargs["act"]).flat[0])

        # No count is maintained during warm-up: the scale still moves there, and a
        # count computed under one scale is not corrected when the scale changes.
        # Once min_size samples are stored, the scale is computed once from all of
        # them and every count is built from it in one pass.
        n_stored = self._max_size if self._full else write_idx + 1
        if n_stored < self._min_size:
            return
        if not self._counts_initialized:
            self._init_scale_and_counts(write_idx)
            return

        # Use slice-based views over the stored arrays to avoid the O(N * obs_dim)
        # copy that np.arange(N)-style fancy indexing would trigger every add().
        if self._full:
            stored_obs = self.obs[..., self.goal_idx]
            stored_act = self.act[:].ravel()
        else:
            stored_obs = self.obs[:write_idx, ..., self.goal_idx]
            stored_act = self.act[:write_idx].ravel()

        # The incremental update only corrects pairs involving the new entry, so
        # it maintains a fixed-radius count only if the scale does not change:
        # otherwise two stored entries could become neighbors without either being
        # a neighbor of the new entry, and neither count would be corrected. The
        # scale is therefore refreshed until it stabilizes, then frozen once and
        # all counts rebuilt under it.
        if self._maybe_freeze_scale(write_idx):
            return

        # Decrement counts of entries that were neighbors of the now-evicted entry
        if evicted is not None:
            evicted_obs, evicted_act = evicted
            close_to_evicted = is_neighbor(
                evicted_obs,
                stored_obs,
                radius=self.rho,
                scale=self._count_scale,
            )
            # write_idx row will be overwritten below; keep it out of the decrement.
            close_to_evicted[write_idx] = False
            evicted_rows = np.nonzero(close_to_evicted)[0]
            self.count[evicted_rows, evicted_act] = np.maximum(
                self.count[evicted_rows, evicted_act] - 1, 0
            )

        # Increment counts of entries that are neighbors of the new entry
        close_to_new = is_neighbor(
            new_obs,
            stored_obs,
            radius=self.rho,
            scale=self._count_scale,
        )
        if self._full:
            close_to_new[write_idx] = False
        new_rows = np.nonzero(close_to_new)[0]
        self.count[new_rows, new_act] += 1

        # Set count for the newly written entry
        count = np.bincount(
            stored_act[close_to_new],
            minlength=n_actions,
        ).astype(np.int64)
        count[new_act] += 1  # count self
        self.count[write_idx] = count

    def counts(self, obs):
        """Count of every action in each observation of obs, shaped (..., n_actions).

        For observations that are not stored, such as a goal candidate the actor
        is considering. Stored entries are counted by get().
        """

        if self.rho is None:
            return self.counter(obs)

        assert self._count_scale is not None, (
            "pseudocounts were queried before the scale exists, which happens "
            "only before min_size samples have been stored; every action would "
            "otherwise be counted under a scale of its own and the resulting "
            "counts would not be comparable"
        )
        obs = np.asarray(obs)
        n_actions = self.count.shape[-1]
        stored_obs = self.obs[:self.size, ..., self.goal_idx]
        stored_act = self.act[:self.size].ravel()
        n = np.zeros(obs.shape[:-1] + (n_actions,))
        for a in range(n_actions):
            n[..., a] = pseudocount(
                obs[..., self.goal_idx],
                stored_obs[stored_act == a],
                radius=self.rho,
                scale=self._count_scale,
            )
        return n

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
    not uniform but depends on "priorities". All priorities are set to an initial
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

    Then, you can sample again using another priority, e.g.,

        >>> batch = mem.get(
                batch_size=32,
                sequence_length=10,
                priority_key="rarity",
            )
        >>> n = visit_count(batch["obs"], batch["act"])
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
        """
        Set the priorities of the samples at `idx` in the tree of `priority_key`.
        `new_priorities` has as many elements as `idx`, and each one must be >= 0.
        Nothing is updated when `priority_key` is None, i.e., when the batch was
        not sampled by priority.
        """

        if priority_key is None:
            return
        new_priorities = np.asarray(new_priorities)
        if new_priorities.size != np.size(idx):
            raise ValueError(
                f"post_sampling got {new_priorities.size} priorities for "
                f"{np.size(idx)} indices of shape {np.shape(idx)}: pass one "
                f"priority per index."
            )
        if np.any(new_priorities < 0):
            raise ValueError("post_sampling got negative priorities.")
        self.update_priorities(idx, new_priorities + 1e-8, priority_key)

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
