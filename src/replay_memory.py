import numpy as np
import src.parameter
from omegaconf import DictConfig

from src.pseudocount import SCALE_EPS, is_neighbor, pseudocount, robust_scale

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

    def export_data(self, filename: str):
        np.savez(
            filename,
            **{k: getattr(self, k)[:self.size] for k in self.keys},
        )

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
        scaled = obs.astype(float) / self._count_scale
        squared = np.sum(scaled**2, axis=1)
        onehot = np.zeros((n, n_actions), dtype=float)
        onehot[np.arange(n), act] = 1.0

        radius2 = self.rho**2
        # Chunk the pairwise distances so the temporaries stay bounded regardless
        # of how many samples are stored when the scale freezes.
        chunk = max(1, int(2**22) // n)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            dist2 = (
                squared[start:stop, None]
                + squared[None, :]
                - 2.0 * scaled[start:stop] @ scaled.T
            )
            np.maximum(dist2, 0.0, out=dist2)  # clamp floating-point negatives
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
        batch["idx"] = idx
        batch["priority_key"] = None
        return batch

    def post_sampling(self, *args, **kwargs):
        pass

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


class SumTree:
    """
    A binary sum tree data structure for prioritized sampling.

    This structure is used in Prioritized Experience Replay (PER) to efficiently:
    - Store scalar priorities associated with each transition in a fixed-capacity buffer.
    - Sample indices proportionally to priority values in O(log N) time.
    - Update individual priorities and propagate changes up the tree in O(log N) time.

    The tree is stored as a flat NumPy array of size `2 * capacity - 1`.
    The `capacity` corresponds to the number of leaf nodes (i.e., number of transitions),
    and all leaf nodes store the actual priority values.
    The internal nodes store the sum of the priorities of their children.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = zeros_resident(2 * capacity - 1, np.float32)

    def add(self, idx, priority):
        """Add new priority to the leaf node for memory index idx."""
        tree_idx = idx + self.capacity - 1
        self.update(tree_idx, priority)

    def update(self, tree_idx, priority):
        """Update tree and propagate change."""
        # `update` must only ever target a leaf node. A non-leaf index (e.g. data-
        # space indices accidentally passed instead of tree-space ones) would either
        # silently corrupt the internal sums or, when it hits the root, send
        # `_propagate` into infinite recursion via a negative parent. Fail loudly.
        assert self.capacity - 1 <= tree_idx < len(self.tree), (
            f"tree_idx {tree_idx} is not a leaf node in "
            f"[{self.capacity - 1}, {len(self.tree)}); did you pass data-space "
            f"indices instead of tree-space ones?"
        )
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def _propagate(self, tree_idx, change):
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def total(self):
        return self.tree[0]

    def get(self, value):
        """Sample value in [0, total) and return index."""
        # Guard against value >= total(): the stratified sampler computes segment
        # bounds in floating point, so `uniform(a, b)` can occasionally return a
        # value at or just above total(). Left unclamped, the traversal would walk
        # fully right into an unwritten (zero-priority) leaf and return a data_idx
        # outside [0, size). Clamp to strictly below total() so we always land in a
        # real, non-zero leaf.
        total = self.total()
        if value >= total:
            value = np.nextafter(total, np.float32(0.0))
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf_idx = parent
                break
            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, data_idx, self.tree[leaf_idx]

    def _random_fill(self, min_p=0.01, max_p=1.0):
        rng_generator = np.random.default_rng(42)
        leaf_priorities = rng_generator.uniform(min_p, max_p, size=self.capacity)
        self.tree[self.capacity - 1 : 2 * self.capacity - 1] = leaf_priorities
        for i in reversed(range(self.capacity - 1)):
            left = 2 * i + 1
            right = left + 1
            self.tree[i] = self.tree[left] + self.tree[right]


class PrioritizedReplayMemory(ReplayMemory):
    """
    This memory supports Prioritized Experience Replay (PER), i.e., sampling is
    not uniform but depends on "prioritites". All priorities are set to an initial
    default max value, and then updated after the sample is used for training.
    For example, in the original PER paper, priorities are proportional to the
    TD error of the sample.
    Priorities are stored in a SumTree structure for faster sampling.
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

    def _create_tree(self, key: str):
        """Initializes a new SumTree and fills it with current default priorities."""

        new_tree = SumTree(self._max_size)
        initial_p = self._default_priority ** self.alpha.value
        for i in range(self.size):
            new_tree.add(i, initial_p)
        self.trees[key] = new_tree
        return new_tree

    def add(self, **kwargs):
        write_index = self._idx  # capture before super() advances _idx
        super().add(**kwargs)
        for tree in self.trees.values():
            tree.add(write_index, self._default_priority ** self.alpha.value)

    def get(
        self,
        batch_size: int,
        sequence_length: int,
        rng_generator: np.random.Generator = None,
        keys: list = None,
        priority_key: str = None,
        **kwargs,
    ):
        if priority_key is None:
            batch = ReplayMemory.get(self, batch_size, sequence_length, rng_generator, keys)
            batch["idx"] += self._max_size - 1  # convert data idx to tree idx (_max_size is the tree capacity)
            return batch

        if priority_key not in self.trees:
            self._create_tree(priority_key)

        active_tree = self.trees[priority_key]

        if keys is None:
            keys = self.keys
        if rng_generator is None:
            rng_generator = self.rng_generator()

        segment = active_tree.total() / batch_size

        leaf_idxs = []
        data_idxs = []
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = rng_generator.uniform(a, b)
            leaf_idx, data_idx, priority = active_tree.get(v)

            leaf_idxs.append(leaf_idx)
            data_idxs.append(data_idx)
            priorities.append(priority)

        leaf_idxs = np.array(leaf_idxs)
        data_idxs = np.array(data_idxs)
        priorities = np.array(priorities)

        # Build sequences ending at sampled indices
        buffer_idx = data_idxs[:, None] + np.arange(-sequence_length + 1, 1)
        buffer_idx = np.remainder(buffer_idx, self.size)

        batch = {k: getattr(self, k)[buffer_idx] for k in keys}

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
        # Tree indices for ALL (B, T) samples, not just endpoints, so
        # post_sampling can update each sample's priority.
        batch["idx"] = buffer_idx + self._max_size - 1
        batch["priority_key"] = priority_key

        return batch

    def get_sorted(self, *args, **kwargs):
        # The base implementation returns data-space indices, but this subclass's
        # `get` returns tree-space indices (offset by _max_size - 1). Keep the two
        # consistent so a caller can safely pass batch["idx"] to post_sampling.
        batch = super().get_sorted(*args, **kwargs)
        batch["idx"] += self._max_size - 1
        return batch

    def update_priorities(self, tree_idxs, new_priorities, priority_key):
        if priority_key not in self.trees:
            self._create_tree(priority_key)
        tree = self.trees[priority_key]
        flat_priorities = new_priorities.flatten()
        flat_priorities = np.nan_to_num(flat_priorities, nan=0.0, posinf=1e6)
        self._default_priority = max(self._default_priority, float(flat_priorities.max()))
        alpha = self.alpha.value
        for idx, priority in zip(tree_idxs.flatten(), (flat_priorities + 1e-5) ** alpha):
            tree.update(idx, float(priority))

    def post_sampling(self, tree_idxs, new_priorities, priority_key, **kwargs):
        if priority_key is None:
            return
        self.update_priorities(
            tree_idxs,
            np.clip(new_priorities, 0, None) + 1e-8,
            priority_key,
        )

    def post_step(self, *args, **kwargs):
        self.alpha.step()
        self.beta.step()

    def _random_fill(self):
        super()._random_fill()
        for tree in self.trees.values():
            tree._random_fill()


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
