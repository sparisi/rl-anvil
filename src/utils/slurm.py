import ctypes
import getpass
import os
import subprocess
import sys

from src.utils.id import CONFIG_ID_LEN, make_id
from src.utils.sweep import format_override


# ------------------------------------------------------------------------------
# ----------------------------------- MEMORY -----------------------------------
# ------------------------------------------------------------------------------

def rss_bytes() -> int:
    """Resident set size of THIS process (0 where /proc is not available).

    With several workers running in parallel, each one reports that process's
    resident memory, while cgroup_memory() reports the memory charged to the
    whole job cgroup. These numbers therefore should not be added together:
    RSS is a per-process view of resident anonymous and file-backed pages,
    whereas cgroup_memory() reports the cgroup's aggregate accounting. Shared
    pages can contribute to the RSS of multiple processes but are accounted for
    only once in the cgroup's aggregate usage.
    """

    txt = _read_file("/proc/self/statm")
    if not txt:
        return 0
    return int(txt.split()[1]) * os.sysconf("SC_PAGE_SIZE")


def peak_rss_bytes() -> int:
    """Largest resident set this process has reached, ever (0 off Unix).

    The kernel maintains it, so unlike rss_bytes() and cgroup_memory() -- both of
    which sample whatever is true at the instant they are called -- nothing is
    missed between two reads. A spike that allocates and frees between
    checkpoints is invisible to the other two and shows up here.

    Same per-process scope as rss_bytes(): this is the high-water mark for one
    process, and its accounting includes resident file-backed pages. Shared
    pages can therefore contribute to the peak RSS of multiple worker processes.
    The value never decreases, so it describes the largest resident set the
    process reached, not the memory it currently holds.
    """

    try:
        import resource
    except ImportError:  # not Unix
        return 0
    # ru_maxrss is kilobytes on Linux, bytes on macOS.
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak_kb * (1 if sys.platform == "darwin" else 1024)


_libc = None


def malloc_trim() -> int:
    """Return free heap memory to the kernel. Bytes of RSS released.

    free() does not necessarily give memory back to the OS: for allocations served
    from glibc's heap, freed blocks normally return to glibc's free lists, so a run
    that allocates and frees a lot can keep its high-water mark resident even while
    Python holds nothing. Large allocations served directly via mmap() are an
    exception: when freed, glibc can normally unmap those pages and return them to
    the OS.

    By default glibc dynamically adjusts its mmap threshold. Freeing a sufficiently
    large mmapped allocation can raise this threshold, causing subsequent
    allocations in that size range to be served from the heap rather than via mmap().
    Those allocations may therefore remain resident after free(). The SLURM jobs do
    not normally behave that way: submit_jobs.py exports MALLOC_MMAP_THRESHOLD_,
    which fixes the mmap threshold and disables its dynamic adjustment, and
    MALLOC_TRIM_THRESHOLD_, which controls when glibc automatically attempts to
    trim the heap. Consequently, malloc_trim() may find less reclaimable memory in
    the cluster jobs than in a bare run without those environment variables. If
    trim_mb looks unexpectedly large, check whether the run inherited them.

    This function serves two purposes. First, it can reclaim memory retained by the
    allocator due to heap fragmentation, including free pages that can be released
    from the heap. It does not prevent fragmentation; it only attempts to return
    currently releasable pages to the OS. Second, it can help distinguish
    allocator-retained memory from some types of leaks: if RSS has been increasing
    but drops substantially after malloc_trim(), that indicates that at least part
    of the increase was memory retained by the allocator rather than memory still
    held by the program. A drop does not, however, prove that the program has no
    memory leak.

    Returns 0 where there is no glibc to ask (macOS, musl, Windows).

    Related: https://github.com/pytorch/pytorch/issues/165319
    """

    global _libc
    if _libc is None:
        try:
            _libc = ctypes.CDLL("libc.so.6")
        except OSError:
            _libc = False
    if not _libc or not hasattr(_libc, "malloc_trim"):
        return 0

    before = rss_bytes()
    _libc.malloc_trim(0)
    return max(before - rss_bytes(), 0)


def _read_file(path: str):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def _stat_fields(text: str, *keys):
    """Values of the "<key> <int>" lines cgroup stat files are made of.

    Missing keys come back as None, so a caller can tell "the kernel does not
    expose this" from "the kernel says zero".
    """

    found = {k: None for k in keys}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in found and found[parts[0]] is None:
            try:
                found[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return found


def cgroup_memory():
    """Memory of this process' cgroup, in bytes, as a dict of:

        anon      Anonymous memory: heap allocations, numpy buffers, torch tensors,
                  and other non-file-backed process memory. This is generally
                  unreclaimable without swap, so it is the main memory component
                  that remains resident under memory pressure.
        file      File-backed memory, including page cache and mapped library text.
                  For this stack, this is mostly libtorch text. It is generally
                  reclaimable under memory pressure, so it can be evicted rather
                  than necessarily causing an OOM kill by itself.
        kernel    Kernel memory charged to the cgroup, including slab, page tables,
                  and per-CPU allocations.
        limit     The cgroup's limit, i.e. what SLURM's --mem became.
        hit_max   How many times the cgroup reached `limit` and the kernel had to
                  reclaim to keep it under. Monotonic.
        oom_kill  Processes killed for memory. Monotonic. (Always 0 on cgroup v1,
                  which does not expose it here.)

    Returns None when there is no cgroup limit to read (a laptop, a node without
    cgroup enforcement).

    Two distinctions are important:

    - Parallel runs share one cgroup: the seeds of a chunk are separate srun
      tasks and separate processes, but they are charged to the same cgroup and
      SLURM enforces the memory limit on their aggregate usage. Per-process RSS
      therefore shows only one task's memory, not the total charged to the job.
    - `anon` alone is not how close the job is to its limit: the limit applies to
      the cgroup's aggregate memory accounting, including `anon`, `file`, and
      `kernel`. Reporting only `anon` can therefore substantially understate the
      total. For this stack, `file` is typically about a gigabyte, which is why
      `mem_all_%` is recorded alongside `mem_%`.

      The distinction also matters operationally. Anonymous memory is generally
      not reclaimable without swap, so sustained memory pressure from it can
      lead to an OOM kill. File-backed memory, by contrast, can often be evicted
      and later faulted back in. If the cgroup repeatedly reaches its limit,
      file-backed pages such as libtorch text may be evicted and then reloaded
      from Lustre, causing severe thrashing and loss of progress without
      necessarily killing a process. `hit_max` records those occasions when the
      cgroup reached its configured memory limit.
    """

    # Find the cgroup this process is in, then walk up to the first ancestor
    # that actually sets a limit: SLURM puts the limit on the job cgroup while
    # the process sits in a step cgroup below it.
    rel = ""
    for line in (_read_file("/proc/self/cgroup") or "").splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[1] == "":  # cgroup v2 line: "0::/some/path"
            rel = fields[2].strip().lstrip("/")
            break

    bases = []
    # os.path.normpath removes trailing slashes if rel is empty, preventing a double check
    path = os.path.normpath(os.path.join("/sys/fs/cgroup", rel))
    while path.startswith("/sys/fs/cgroup"):
        bases.append(path)
        if path == "/sys/fs/cgroup":
            break
        path = os.path.dirname(path)

    for base in bases:
        raw_limit = (_read_file(os.path.join(base, "memory.max")) or "").strip()
        if not raw_limit or raw_limit == "max":
            continue
        # memory.stat is recursive on v2, so reading it at the level that holds
        # the limit already covers every task and every parent below it.
        stat = _stat_fields(
            _read_file(os.path.join(base, "memory.stat")), "anon", "file", "kernel",
        )
        if stat["anon"] is None:
            continue
        events = _stat_fields(
            _read_file(os.path.join(base, "memory.events")), "max", "oom_kill",
        )
        return {
            "anon": stat["anon"],
            "file": stat["file"] or 0,
            "kernel": stat["kernel"] or 0,
            "limit": int(raw_limit),
            "hit_max": events["max"] or 0,
            "oom_kill": events["oom_kill"] or 0,
        }

    # cgroup v1: the limit is "unlimited" as a huge number, `anon` is spelled
    # `rss`, `file` is `cache`, and the reclaim counter is memory.failcnt.
    v1 = "/sys/fs/cgroup/memory"
    raw_limit = (_read_file(os.path.join(v1, "memory.limit_in_bytes")) or "").strip()
    if raw_limit.isdigit() and int(raw_limit) < (1 << 60):
        stat = _stat_fields(_read_file(os.path.join(v1, "memory.stat")), "rss", "cache")
        if stat["rss"] is not None:
            failcnt = (_read_file(os.path.join(v1, "memory.failcnt")) or "").strip()
            return {
                "anon": stat["rss"],
                "file": stat["cache"] or 0,
                "kernel": 0,
                "limit": int(raw_limit),
                "hit_max": int(failcnt) if failcnt.isdigit() else 0,
                "oom_kill": 0,
            }

    return None


# ------------------------------------------------------------------------------
# ------------------------------------ JOBS ------------------------------------
# ------------------------------------------------------------------------------

def job_config_id(hp_overrides):
    """The config_id of the configuration a job will run, before submitting it.

    `hp_overrides` must hold every override that reaches the config: the swept
    keys, `environment` and `algorithm` included, and anything the submission
    script pins on top, such as the approximator device. Leaving one out would
    name a different configuration than the one main.py writes to disk.
    """
    overrides = [
        format_override(k, v)
        for k, v in (hp_overrides or {}).items()
    ]
    return make_id(overrides=overrides)


def job_name(config_id, seeds):
    """SLURM job name for one submission: "<config_id>_<seeds>", e.g. a1b2c3d4_0-4.

    The seed part is a range when the seed set is contiguous (which chunking a
    seed range always is) and the explicit list otherwise, so that two different
    seed sets -- say [0, 3, 7], as plot_results.py's missing-runs generator
    produces, and 0..7 -- never collapse into the same name. Names are what
    `queued_job_names` matches on, so a collision would silently skip a job.
    """
    if len(seeds) == 1:
        seeds_str = str(seeds[0])
    elif list(seeds) == list(range(seeds[0], seeds[-1] + 1)):
        seeds_str = f"{seeds[0]}-{seeds[-1]}"
    else:
        seeds_str = ".".join(map(str, seeds))
    return f"{config_id}_{seeds_str}"


def queued_job_names():
    """Names of this user's pending and running jobs.

    Returns an empty set if squeue cannot be reached: failing to ask must not
    stop the submission, it only means duplicates go undetected.
    """
    user = os.environ.get("USER") or getpass.getuser()
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", user, "-t", "PD,R", "-o", "%j"],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"WARNING: could not query squeue ({exc}); cannot check running/pending jobs")
        return set()
    return {name.strip() for name in out.splitlines() if name.strip()}


def parse_seeds(tokens):
    """Seed list from CLI tokens: ints are literal seeds, "a-b" an inclusive range.

        ["0-9"]       -> [0, 1, ..., 9]
        ["5"]         -> [5]
        ["0-4", "7"]  -> [0, 1, 2, 3, 4, 7]

    The result is sorted and deduplicated, so the same set of seeds always
    produces the same `job_name`.
    """
    seeds = set()
    for token in tokens:
        token = str(token).strip()
        try:
            if "-" in token:
                first, _, last = token.partition("-")
                seeds.update(range(int(first), int(last) + 1))
            else:
                seeds.add(int(token))
        except ValueError:
            raise SystemExit(
                f"Cannot read '{token}' as seeds. Use ints for single seeds and "
                "a-b for an inclusive range, e.g. 0-9, 5, or 0-4 7 9."
            )
    return sorted(seeds)


def parse_job_name(name):
    """(config_id, seeds) from a name built by `job_name`, or None if it is not one.

    It parses the format produced by `job_name`:
    "a1b2c3d4_0-4" -> ("a1b2c3d4", [0, 1, 2, 3, 4]).

    Anything that does not have the "<CONFIG_ID_LEN hex>_<seeds>" shape belongs to
    some other tool and is not ours to interpret. The parser returns None when
    the seed portion cannot be parsed as integers, rather than raising an
    exception.
    """
    config_id, sep, seeds_str = name.partition("_")
    if not sep or len(config_id) != CONFIG_ID_LEN:
        return None
    if any(c not in "0123456789abcdef" for c in config_id):
        return None
    try:
        if "-" in seeds_str:
            first, _, last = seeds_str.partition("-")
            seeds = list(range(int(first), int(last) + 1))
        else:
            seeds = [int(s) for s in seeds_str.split(".")]
    except ValueError:
        return None
    return config_id, seeds


def queued_runs():
    """(config_id, seed) pairs this user already has pending or running.

    `job_name` encodes exactly that in the SLURM job name, so the queue itself is
    the record of what not to resubmit -- there is no state file to keep in sync.
    Matching per seed rather than per name means two submissions that overlap
    only partly (0-9 while 0-4 runs) never train the same seed twice.
    """
    runs = set()
    for name in queued_job_names():
        parsed = parse_job_name(name)
        if parsed is None:
            continue
        config_id, seeds = parsed
        runs.update((config_id, seed) for seed in seeds)
    return runs
