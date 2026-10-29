"""
Use this script to run multiple Hydra sweeps in parallel with SLURM job arrays.
If runs require only CPUs, it's best to run them in parallel over multiple CPUs
as separate SLURM TASKS of the same job.
To further parallelize everything, we recommend to submit multiple SLURM jobs
with low compute requirements. Having low requirements ensures your job gets
high priority.
Each job will take care of a chunk of the seed range.

Sweeps are defined entirely in SWEEP_DIR (see --sweep):
environments, algorithms and hyperparameters. Every combination RUNS ON A
SEPARATE JOB, with fine-grained control over which combinations are generated.
That is, you won't blindly generate all combinations of whatever argument sweep
you defined.

This script submits one job array per (env, algorithm, configuration), each array
element running a chunk of seeds. Within an element, every seed is its own SLURM
task, launched by srun and running one plain `python main.py` for that seed alone.
The number of CPUs is proportional to the number of seeds in the chunk.
Time and memory needed for a run are adjusted manually depending on the environment.

Why tasks? Why not Joblib?

    Joblib runs the seeds of a chunk as workers of ONE process, so they share
    one .out/.err stream and one crash takes the sweep down with it. SLURM tasks
    are separate processes with separate log files: a seed that dies does not
    abort its siblings' Python, and its traceback is in a file of its own.
    This relies on srun running with --kill-on-bad-exit=0, which this script
    passes explicitly: with the cluster's KillOnBadExit=1, the first task to
    exit non-zero would otherwise terminate every other task of the step.
    Parallelism is unchanged -- the seeds of a chunk still run at the same time,
    within the same single job.

    What tasks do NOT buy is memory isolation: all tasks of a job share one step
    cgroup and one --mem pool. A seed that spikes borrows from quieter siblings,
    but a genuine runaway OOMs the whole job (all seeds of the chunk), not just
    itself. Size PER_SEED_MEM_MB accordingly.

Why job arrays? Why not one job per chunk?

    Each array element counts as a separate job for scheduling and accounting,
    exactly as a standalone job does, so an array does not buy quota. What it
    buys is one sbatch call per configuration instead of one per chunk, and
    `slurm_array_parallelism`, which throttles how many elements OF ONE ARRAY
    run at once (max_running_per_array in the cluster settings). It does not
    limit the arrays as a whole: total parallelism is bounded by the cluster's
    MaxJobs for this user, times seeds_per_chunk.

Example:

    python submit_jobs_slurm.py --sweep=encoders --data_dir=data_encoders --cluster=roihu --seeds 0-9 --seeds_per_chunk=5

    <SWEEP_DIR>/encoders.yaml defines a sweep over network encoders and some
    hyperparameters, for a total of 234 configurations.
    Each of them is tested against 10 seeds split into 2 chunks of 5 seeds, for
    a total of 468 jobs (each with 5 tasks, one per seed).

Logging:

    To check a seed's progress (one file per task)
    cat $SCRATCH/slurm_out/<jobid>_<element>_<task>_log.out

    To check a seed's errors
    cat $SCRATCH/slurm_out/<jobid>_<element>_<task>_log.err

    To analyze error logs
    bash detect_jobs_err.sh

Sizing BASE_MEM_MB and PER_SEED_MEM_MB:

    First, here is how to read SLURM statistics:

    - `sacct MaxRSS` is the largest SINGLE task, sampled every
      JobAcctGatherFrequency seconds. It misses the sibling tasks that share the
      pool, misses the submitit parents and squashfuse mounts, and misses
      file-backed memory entirely. An OOM-killed job routinely reports a MaxRSS
      at half of ReqMem.
    - `memory.peak` is the high-water mark of memory.current, which counts page
      cache. Cache expands into whatever --mem leaves free, so memory.peak reads
      ~100% of the limit on perfectly healthy jobs.

    Python memory profilers (memray, Fil, scalene) do not size these either:

    - They hook the malloc family, but a shared library's .data/.bss becomes
      private-dirty anonymous memory the moment it is touched, and was never
      malloc'd. libtorch's dispatch tables and op registries live there, so no
      profiler can see them.
    - They report what is allocated and not yet freed. Memory the program freed
      but glibc still holds is charged to the cgroup all the same -- that is
      what `src.utils.slurm.malloc_trim` measures.

    What they are good for is WHICH code allocates, and above all finding leaks.
    Reach for one when a run's `rss_mb` climbs in a way `trim_mb` does not
    explain -- that is memory the program is holding, not the allocator, and a
    profiler names the frame holding it, which no cgroup number can do.

    What to size from is the cgroup's own breakdown, which splits exactly along
    the two constants:

        BASE_MEM_MB     = memory.stat `file` + `kernel`.  Mapped libtorch text
                          and page cache, charged ONCE however many seeds run.
        PER_SEED_MEM_MB = (summed anon of the main.py processes + the per-task
                          overhead) / number of seeds. The overhead is the
                          submitit parent (~25 MB) plus the Apptainer
                          squashfuse_ll mounts (~18 MB) that every task carries.

    inspect_jobs.sh prints both, and the [analysis] block states the constants
    they imply. It can only read them while a job RUNS -- the cgroup is
    destroyed when the job ends -- so:

    1. Submit the same configuration at different task counts and with
       DIFFERENT seeds (the same seeds would run twice, or be skipped under
       --skip_duplicates), with both constants below temporarily doubled so
       nothing is reclaimed:

           python submit_jobs_slurm.py --sweep=<file> --data_dir=<dir> --cluster=<name> --seeds 0    --seeds_per_chunk=1
           python submit_jobs_slurm.py --sweep=<file> --data_dir=<dir> --cluster=<name> --seeds 1-4  --seeds_per_chunk=4
           python submit_jobs_slurm.py --sweep=<file> --data_dir=<dir> --cluster=<name> --seeds 5-12 --seeds_per_chunk=8
           ...

    2. Poll them for the whole short runs. `anon` is an instantaneous sample, not a
       high-water mark -- the kernel exposes no peak for anon alone -- and it
       reaches its peak after all imports are done (which may take a bit of time).
       inspect_jobs.sh keeps the max across polls in $SCRATCH/<slurm_out>/<jobid>.cgpeak,
       which is also what lets --fit run on jobs that have since ended:

           ( while squeue -h -j <idA>,<idB>,<idC> -o %i 2>/dev/null | grep -q .; do
                 bash inspect_jobs.sh <idA> <idB> <idC>; sleep 20; done ) >/dev/null 2>&1 &

    3. Once all jobs end, run:

           bash inspect_jobs.sh --fit <idA> <idB> <idC>

       The intercept is BASE_MEM_MB, the slope PER_SEED_MEM_MB. Add 10-20%.

Duplicate jobs:

    Every array is named "<config_id>_<seeds>" (e.g. a1b2c3d4_0-9), where config_id
    is the ID main.py derives from the configuration and uses to name the data
    folder (<data_dir>/<config_id>/<seed>). All elements of an array share the name
    of the seeds THAT array carries.

    With --skip_duplicates, this script checks the user's jobs in the queue, in
    any state, and skips the seeds their names cover. It also skips seeds that
    already produced <data_dir>/<config_id>/<seed>/data.npz. If squeue cannot
    be queried, it stops without submitting anything.
    Without this flag, every seed is submitted as given, so a seed already queued
    is submitted a second time.

    The config_id includes the overrides passed with --with, so they must be
    the same ones the earlier submission used for its runs to be recognized.
    Seeds that already produced a data.npz are also skipped by main.py itself,
    which is why every command below passes results.force_run=False.

    Every task is one run of one configuration, so the name always describes
    exactly what the array produces.

Editing configs or code while jobs are pending:

    A job composes its configuration when it STARTS, from configs/ and the code
    as they are at that moment, while its name carries the config_id computed
    at submission. Editing either while jobs are pending changes what those
    jobs run, and they write to a folder whose config_id their name does not
    carry. A later submission computes the new config_id, which no queued name
    matches, so --skip_duplicates resubmits seeds that are still in the queue.
    Wait until pending jobs start before editing, or cancel and resubmit them
    after the edit.

Job Limit and Automatic Submission

    Your cluster may limit the number of jobs you can submit. Set
    max_submit_jobs in the cluster settings (configs/slurm) accordingly.
    If you have to submit more jobs, you have to re-run this script multiple times
    (be sure to pass --skip_duplicates)

        python submit_jobs_slurm.py [...] --skip_duplicates

    Instead of doing it manually, you can make a simple script to submit runs
    periodically, for example with tmux

        tmux new -s jobs_submit
        while true; do date; python submit_jobs_slurm.py [...] --skip_duplicates; sleep 1800; done

Note:

    In `submit_jobs(jobs, ...)`, `jobs` is a list of (hp_overrides, seeds)
    tuples, one per run. `hp_overrides` holds every Hydra override of the run,
    `environment` and `algorithm` included. For example:
        [
            ({'environment': 'gymnasium/lunar_lander', 'algorithm': 'eps_greedy', 'agent.critic.approximator.encoder.grid_id': 'FuzzyTileActivation'}, [0, 1, 2]),
            ({'environment': 'gymnasium/lunar_lander', 'algorithm': 'eps_greedy', 'agent.critic.approximator.encoder.grid_id': 'GaussianActivation'}, [0, 1, 2]),
        ]

    This function is used as public entry point for other tools (e.g.,
    plot_results.py's missing-runs generator).
"""

import os
import shlex
import sys
import subprocess
import submitit
import cloudpickle
import argparse

from src.utils.slurm import (
    SLURM_CONFIG_DIR,
    job_name,
    job_config_id,
    parse_seeds,
    queued_runs,
    queued_job_count,
    load_cluster,
    slurm_setup,
)
from src.utils.sweep import expand_sweep, format_override, load_sweep


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CPUS_PER_SEED = 1  # per task, i.e. per seed; increase if a run needs parallelism
SLURM_SETUP = [
    f"cd {shlex.quote(SCRIPT_DIR)}",
    f"export OMP_NUM_THREADS={CPUS_PER_SEED}",
    f"export NUMEXPR_NUM_THREADS={CPUS_PER_SEED}",
    f"export MKL_NUM_THREADS={CPUS_PER_SEED}",
    f"export OPENBLAS_NUM_THREADS={CPUS_PER_SEED}",
    "export HYDRA_FULL_ERROR=1",
    "export MALLOC_MMAP_THRESHOLD_=131072",
    "export MALLOC_TRIM_THRESHOLD_=131072",
    "export MALLOC_ARENA_MAX=2",
]

# Why the *_NUM_THREADS variables?
# They cap the thread pools OpenBLAS, MKL, OpenMP and numexpr create inside
# numpy and torch, which size themselves from the CPUs they detect on the node
# rather than from what SLURM allocated. Without them one task with
# cpus_per_task=1 can spawn a thread per core and have them fight over its
# single CPU -- slower, and heavier, since every thread carries a stack and can
# claim a malloc arena of its own (see below).
# They are set to CPUS_PER_SEED, so they follow it if it is raised.

# Why the MALLOC variables?
# glibc keeps freed blocks rather than returning them to the kernel, and it
# raises its own mmap threshold (up to 32 MB) whenever it sees a large block
# freed -- after which allocations of that size come from the heap, where
# free() cannot release them. Pinning both thresholds keeps large temporaries
# on mmap, where free() does give the pages back; capping the arenas stops
# each thread from claiming a heap of its own. This trades a few more
# mmap/munmap syscalls for a resident set that follows what is actually live.
# Related: https://github.com/pytorch/pytorch/issues/165319

# Applied to every run, but a sweep file can override them.
# These are override literals, not YAML: the grammar reads True and False in any
# case as booleans and `null` as None, while None arrives as the string "None".
DEFAULT_OVERRIDES = [
    "results.progress_report=dict",
    "results.slurm_debug=True",
    "results.force_run=False",
    "wandb.mode=disabled",
]

# Per-env wallclock (minutes) for the *base* algorithm. TIMEOUTS_ADJUST_ALGO
# scales it up for algorithms that take longer per step. A job's wallclock is
# this budget, not the sum over its seeds, because the seeds run concurrently as
# tasks.
TIMEOUTS_ENV = {
    "gymnasium/lunar_lander": 280,
    "gymnasium/cart_pole": 200,
    "gymnasium/pendulum_discrete": 120,
    "gymnasium/acrobot": 200,
    "gymnasium/mountain_car": 200,
    "gym_gridworlds/four_rooms": 60,
    "gym_gridworlds/large": 30,
    "gym_gridworlds/travel_field_small": 90,
}

# Use this to customize timeouts: {algorithm: factor} applied to TIMEOUTS_ENV.
# Algorithms not listed here use a factor of 1.
TIMEOUTS_ADJUST_ALGO = {
    "random": 0.8,
}

# Memory: `mem_mb = BASE_MEM_MB + PER_SEED_MEM_MB[env] * len(seeds_in_chunk)`.
# SLURM's --mem is PER NODE, and an array element holds the whole chunk on one
# node, so the two parts are not interchangeable:
#  - BASE_MEM_MB: what the job maps once however many seeds run -- the torch and
#    gymnasium library pages (file-backed, so the cgroup is charged once even
#    though every task's RSS reports them) plus the batch step's bash and srun.
#  - PER_SEED_MEM_MB: what each seed adds on its own -- its Python heap, its
#    networks and its replay memory. This is anonymous memory, never shared
#    between tasks, so it is the part that multiplies.
# Folding the base into the per-seed value would pay for the library pages once
# per seed instead of once per job, over-requesting more the more seeds a chunk
# holds.
# All tasks of a job share ONE step cgroup and therefore one memory pool: a seed
# that spikes borrows from quieter siblings, but a runaway OOMs the whole chunk.
# PER_SEED_MEM_MB is measured on CPU runs. With --cuda, every task that
# initializes CUDA also holds the CUDA context in host anonymous memory, which
# these values do not include, and all tasks of an element share the single GPU
# requested without MPS. With --cuda and seeds_per_chunk > 1, measure the CUDA
# runs separately and expect OOMs or GPU time-slicing.
# See the docstring for instructions to estimate requested memory.
BASE_MEM_MB = 1250
PER_SEED_MEM_MB = {
    "gymnasium/lunar_lander": 850,
    "gymnasium/cart_pole": 650,
    "gymnasium/pendulum_discrete": 650,
    "gymnasium/acrobot": 650,
    "gymnasium/mountain_car": 550,
    "gym_gridworlds/four_rooms": 600,
    "gym_gridworlds/large": 300,
    "gym_gridworlds/travel_field_small": 900,
}


def run_sweep(cmds):
    """Run ONE command from `cmds`, selected by this task's rank.

    This executes ON THE COMPUTE NODE (submitit ships it there). Submitit
    launches this payload once per SLURM task (one per seed of the array's
    largest chunk), handing every task the same argument. `local_rank` is what
    makes them differ: task 0 runs cmds[0], task 1 runs cmds[1], and so on.
    Each executes in its own process, and submitit routes its stdout/stderr to
    a log file carrying the task rank, so seeds sharing a job never interleave
    their output.

    The last chunk of a sweep may be short. Ranks past the end exit 0 rather than
    failing, so a partial chunk does not mark the job FAILED.
    """

    try:
        rank = submitit.JobEnvironment().local_rank
    except Exception:
        # Not under SLURM (local testing): behave as a single-task job.
        rank = 0

    if rank >= len(cmds):
        sys.stderr.write(f"[task {rank}] no work in this chunk ({len(cmds)} run(s)); exiting.\n")
        return

    cmd = cmds[rank]               # `cmd` is an argv list
    cmd_command = shlex.join(cmd)  # join into the full command for quick copy-paste reproducibility
    sys.stderr.write(f"\n[task {rank}] {cmd_command}\n\n")
    sys.stderr.flush()

    code = subprocess.run(cmd).returncode
    if code != 0:
        # sys.exit, not raise: submitit catches Exception and dumps its own
        # six-frame traceback; SystemExit slips past it and SLURM still marks the
        # task FAILED.
        if code < 0:
            # Map negative signal codes to standard POSIX exit codes (128 + signal)
            code = 128 - code
        sys.exit(code)


# submitit sends run_sweep to the compute node with cloudpickle, and on the node
# every task starts a submitit parent process that unpickles it and launches
# main.py. cloudpickle sends a function in one of two ways:
#  - By value: the function's code is sent, and the parent only imports the
#    modules run_sweep uses (submitit, sys, shlex, subprocess).
#  - By reference: only the name "submit_jobs_slurm.run_sweep" is sent, and the
#    parent must import this whole module to find it, including src.utils and,
#    through it, hydra and omegaconf.
# When this file is run as a script, run_sweep lives in __main__, which cannot
# be imported on the node, so it is sent by value. When another script imports
# submit_jobs from this module, run_sweep would be sent by reference, and every
# task would hold those extra imports in memory for the whole run, on top of
# the per-seed overhead PER_SEED_MEM_MB is sized for.
# Registering this module for pickling by value makes both cases send run_sweep
# by value.
if __name__ != "__main__":
    cloudpickle.register_pickle_by_value(sys.modules[__name__])


def check_envs(jobs):
    """Stop before submitting anything if a configuration lacks `environment` or
    `algorithm`, or if its environment has no timeout or no per-seed memory."""

    missing_timeout = set()
    missing_memory = set()
    missing_group = []
    for hp_overrides, _ in jobs:
        env, algo = hp_overrides.get("environment"), hp_overrides.get("algorithm")
        if env is None or algo is None:
            missing_group.append(hp_overrides)
            continue
        if env not in TIMEOUTS_ENV:
            missing_timeout.add(env)
        if env not in PER_SEED_MEM_MB:
            missing_memory.add(env)

    problems = []
    if missing_group:
        problems.append(
            f"{len(missing_group)} configuration(s) without environment/algorithm, "
            f"e.g. {missing_group[0]}"
        )
    if missing_timeout:
        problems.append(f"no TIMEOUTS_ENV entry for: {sorted(missing_timeout)}")
    if missing_memory:
        problems.append(f"no PER_SEED_MEM_MB entry for: {sorted(missing_memory)}")
    if problems:
        raise SystemExit("Nothing submitted.\n" + "\n".join(f"  - {p}" for p in problems))


def submit_jobs(
    jobs,
    data_dir,
    cluster,
    debug=False,
    cuda=False,
    seeds_per_chunk=1,
    extra=(),
    skip_duplicates=False,
):
    """Submit (hp_overrides_dict, seeds) tuples to SLURM.

    `seeds` in each job may be either a list of seed ints or a single int N
    (interpreted as range(N)). Each job's seeds are split into chunks of
    `seeds_per_chunk` seeds; all chunks of a job go out as ONE array (one
    submission, one name), each chunk becoming one array element that runs
    each of its seeds as its own parallel srun task. `cluster` names the
    settings file in SLURM_CONFIG_DIR. A seed listed twice in one job is
    submitted once.

    An array's name covers every seed of the array, so with `skip_duplicates`
    a seed whose element failed counts as queued until every element of its
    array has left the queue.
    """

    settings = load_cluster(cluster)
    if cuda and settings.get("gres") is None:
        raise SystemExit(f"--cuda needs gres in {SLURM_CONFIG_DIR}/{cluster}.yaml")
    if not cuda and settings.get("gres") is not None:
        print(f"WARNING: {cluster} requests gres={settings['gres']} but --cuda is not "
              "set, so runs use the CPU")

    scratch = os.path.expandvars(settings["scratch_dir"])
    print(f"Cluster SCRATCH folder: {scratch}")

    data_root = f"{scratch}/{data_dir}"
    base_overrides = [
        f"wandb.dir={scratch}/{settings['wandb_out_dir']}",
        f"results.data_dir={data_root}",
        *DEFAULT_OVERRIDES,
    ]

    seeds_per_chunk = int(seeds_per_chunk)
    if seeds_per_chunk < 1:
        raise ValueError("seeds_per_chunk must be >= 1")

    check_envs(jobs)

    device = "cuda" if cuda else "cpu"

    # This is a snapshot, not an atomic reservation: two submit_jobs() invocations
    # started concurrently can both observe the same seed as free and submit it.
    # Preventing that race requires coordination outside this process.
    already_running = set()
    if skip_duplicates:
        already_running = queued_runs()
        if already_running is None:
            raise SystemExit(
                "Nothing submitted: squeue could not be queried, so pending and "
                "running seeds cannot be skipped."
            )

    # Count the queued jobs once, and only when max_submit_jobs is set, since
    # nothing else reads the count. Jobs this call submits are added through
    # `submitted` rather than by querying squeue again, which may not list a job
    # right after its submission. Like the snapshot above, this does not prevent
    # races: jobs submitted by anything else after this point are not counted.
    base_queued_jobs = queued_job_count() if settings.get("max_submit_jobs") is not None else 0

    skipped_configs = 0
    skipped_queued_seeds = 0
    skipped_completed_seeds = 0
    skipped_repeated_seeds = 0
    planned = set()
    arrays = []
    for hp_overrides, seeds in jobs:
        if isinstance(seeds, int):
            seeds = list(range(seeds))
        seeds = sorted({int(seed) for seed in seeds})
        if not seeds:
            print(f"WARNING: no seeds provided for {hp_overrides}, skipping")
            continue
        env, algo = hp_overrides["environment"], hp_overrides["algorithm"]

        config_hp = {**hp_overrides, "agent.critic.approximator.device": device}
        config_overrides = [
            format_override(k, v)
            for k, v in config_hp.items()
        ]
        config_id = job_config_id(config_hp, extra, defaults=base_overrides)

        # Drop the seeds an earlier entry of `jobs` already covers. Two entries
        # can resolve to the same config_id (e.g., one config_sweep entry pinning
        # a key to a value another one takes from global_sweep), and would
        # otherwise write to the same folder.
        repeated = [seed for seed in seeds if (config_id, seed) in planned]
        seeds = [seed for seed in seeds if (config_id, seed) not in planned]
        skipped_repeated_seeds += len(repeated)
        if not seeds:
            print(f"Skipped {env}/{algo}, config_id={config_id}: all {len(repeated)} seed(s) "
                  "already in this submission")
            skipped_configs += 1
            continue
        if repeated:
            print(f"Note {env}/{algo}, config_id={config_id}: seeds {repeated} already in "
                  "this submission, submitting the rest")

        if skip_duplicates:
            # Drop the seeds that are already in the queue. Every job name says
            # which config_id and which seeds it covers, so the queue is the record: a
            # half-submitted configuration gets only its missing seeds, and two
            # overlapping submissions (0-9 while 0-4 runs) never train a seed twice.
            queued = [seed for seed in seeds if (config_id, seed) in already_running]
            seeds = [seed for seed in seeds if (config_id, seed) not in already_running]
            # Drop the seeds that already produced data.npz.
            completed = [
                seed for seed in seeds
                if os.path.exists(os.path.join(data_root, config_id, str(seed), "data.npz"))
            ]
            seeds = [seed for seed in seeds if seed not in completed]
            skipped_queued_seeds += len(queued)
            skipped_completed_seeds += len(completed)
            if not seeds:
                print(f"Skipped {env}/{algo}, config_id={config_id}: all seeds "
                      f"already pending/running ({len(queued)}) or completed ({len(completed)})")
                skipped_configs += 1
                continue
            if queued:
                print(f"Note {env}/{algo}, config_id={config_id}: seeds {queued} "
                      "already pending or running, submitting the rest")
            if completed:
                print(f"Note {env}/{algo}, config_id={config_id}: seeds {completed} "
                      "already completed, submitting the rest")

        planned.update((config_id, seed) for seed in seeds)
        chunks = [seeds[i:i + seeds_per_chunk] for i in range(0, len(seeds), seeds_per_chunk)]
        if debug:
            # A debug submission sends only the first chunk
            chunks = chunks[:1]
        arrays.append((hp_overrides, config_overrides, config_id, chunks))

    tot_jobs = len(arrays)
    submitted = 0

    try:
        for job_idx, (hp_overrides, config_overrides, config_id, chunks) in enumerate(arrays, 1):
            env, algo = hp_overrides["environment"], hp_overrides["algorithm"]

            executor = submitit.AutoExecutor(
                folder=f"{scratch}/{settings['slurm_out_dir']}",  # you will find error logs here
                cluster="slurm",  # force SLURM rather than rely on auto-detection
            )

            # Scale tasks and memory according to seeds (each seed is one srun task).
            # All elements of an array share one resource request, so it must cover
            # the biggest chunk; a shorter last chunk simply leaves its extra ranks
            # idle (they exit 0 in `run_sweep`).
            tasks_per_element = max(len(chunk) for chunk in chunks)
            mem_mb = int(BASE_MEM_MB + PER_SEED_MEM_MB[env] * tasks_per_element)

            # Adjust timeouts
            timeout = TIMEOUTS_ENV[env]
            timeout = int(timeout * TIMEOUTS_ADJUST_ALGO.get(algo, 1))
            max_timeout = settings.get("max_timeout")
            if max_timeout is not None and timeout > max_timeout:
                print(f"WARNING: {env}/{algo} timeout of {timeout} min clamped to "
                      f"max_timeout={max_timeout} min")
                timeout = max_timeout

            # `tasks_per_node` is what runs the seeds of one chunk in parallel: one
            # srun task per seed, each with its own process and its own log file.
            executor.update_parameters(
                slurm_account=settings.get("account"),
                slurm_partition=settings.get("partition"),
                timeout_min=timeout,
                nodes=1,
                tasks_per_node=tasks_per_element,
                cpus_per_task=CPUS_PER_SEED,
                slurm_mem=f"{mem_mb}M",
                slurm_setup=[*slurm_setup(settings), *SLURM_SETUP],
                # With KillOnBadExit=1, the first task to exit non-zero would
                # otherwise terminate every other task of the step.
                slurm_srun_args=["--kill-on-bad-exit=0"],
            )
            if settings.get("max_running_per_array") is not None:
                executor.update_parameters(
                    slurm_array_parallelism=settings["max_running_per_array"],
                )

            # gres is requested per job, so all tasks (seeds) of an element share it.
            if settings.get("gres") is not None:
                executor.update_parameters(slurm_gres=settings["gres"])

            # Job command, one per seed: a plain single run, no Hydra multirun, so no
            # 'multirun.yaml' and (configs/hydra/default.yaml sets output_subdir to
            # null and disables logging) nothing else written into hydra.run.dir
            # either. Hydra creates that folder all the same, so every run points at
            # ONE shared directory: a per-run path would leave an empty folder per
            # (configuration, seed) behind forever. Nothing lands there to collide.
            # Where data actually goes is decided by the config_id, which depends on all
            # hyperparameters: <data_dir>/<config_id>/<seed>/data.npz.
            # The config_id is NOT the SLURM_ID -- that one defines log files
            # (slurm_out/<jobid>_<element>_<task>_log.*) and is assigned by SLURM at
            # submission. It is, however, a unique ID determined by all hyperparameters,
            # so the array name (<config_id>_<seeds>) is also a unique ID for the job.
            chunk_cmds = []
            for chunk in chunks:
                chunk_cmds.append([
                    [
                        "python", "main.py",
                        f"hydra.run.dir={scratch}/{settings['hydra_out_dir']}",
                        *base_overrides,     # before config_overrides
                        *config_overrides,   # can override DEFAULT_OVERRIDES
                        f"experiment.rng_seed={seed}",
                        *extra,
                    ]
                    for seed in chunk
                ])

            # SLURM counts array elements individually, so a very wide sweep goes out
            # as several arrays of at most max_elements_per_array elements each.
            # MaxSubmitJobs counts array elements from all arrays together.
            max_submit_jobs = settings.get("max_submit_jobs")
            max_elements_per_array = settings.get("max_elements_per_array")
            first = 0
            while first < len(chunk_cmds):
                batch_size = len(chunk_cmds) - first
                if max_submit_jobs is not None:
                    available_submit_slots = max(max_submit_jobs - base_queued_jobs - submitted, 0)
                    if available_submit_slots == 0:
                        raise SystemExit(
                            f"Submission of {config_id} stopped: no submit slots available "
                            f"(MaxSubmitJobs={max_submit_jobs}). {submitted} array element(s) "
                            "went out before this point and are still queued."
                        )
                    batch_size = min(batch_size, available_submit_slots)
                if max_elements_per_array is not None:
                    batch_size = min(batch_size, max_elements_per_array)
                batch = chunk_cmds[first:first + batch_size]

                batch_seeds = sorted(
                    seed
                    for chunk in chunks[first:first + batch_size]
                    for seed in chunk
                )
                name = job_name(config_id, batch_seeds)
                executor.update_parameters(name=name)

                try:
                    jobs_out = executor.map_array(run_sweep, batch)
                except Exception as exc:
                    msg = str(exc)
                    hint = ""
                    if "AssocMaxSubmitJobLimit" in msg or "QOSMax" in msg or "violates" in msg:
                        hint = ("\nSubmit limit hit. SLURM counts array elements "
                                "individually -- raise --seeds_per_chunk to use fewer.")
                    raise SystemExit(
                        f"Submission of {name} failed: {msg}{hint}\n"
                        f"{submitted} array element(s) went out before this point and "
                        "are still queued; scancel them if you are restarting."
                    )

                submitted += len(jobs_out)
                first_id = jobs_out[0].job_id if jobs_out else "?"
                print(f"Submitted job {job_idx}/{tot_jobs}: {env}/{algo}, name={name}, "
                      f"id={first_id} ({len(jobs_out)} element(s) x {tasks_per_element} task(s), "
                      f"{mem_mb} MB, {timeout} min)")

                first += batch_size

            if debug:
                print("Stopping, this was just a debug submission.")
                break
    finally:
        print(f"Skipped {skipped_repeated_seeds} run(s) repeated within this submission.")
        if skip_duplicates:
            print(f"Skipped {skipped_queued_seeds} run(s) already pending or running.")
            print(f"Skipped {skipped_completed_seeds} run(s) already completed.")
        print(f"Skipped {skipped_configs} configuration(s) with no seed left to submit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True,
        help="Sweep file in SWEEP_DIR (name without .yaml or a path). "
             "It defines environments, algorithms and hyperparameters.")
    parser.add_argument("--data_dir", required=True,
        help="Where runs write, under the cluster's scratch_dir.")
    parser.add_argument("--cluster", required=True,
        help="Cluster settings file in configs/slurm (name without .yaml).")
    parser.add_argument("--seeds", nargs="+", required=True,
        help="Seeds to run: ints are literal seeds, a-b an inclusive range. "
             "E.g. --seeds 0-9, --seeds 5, --seeds 0-4 7 9.")
    parser.add_argument("--seeds_per_chunk", type=int, default=1,
        help="Seeds per array element, run in parallel as separate SLURM tasks.")
    parser.add_argument("--debug", action="store_true",
        help="If True, it submits only one job.")
    parser.add_argument("--cuda", action="store_true",
        help="If True, jobs will run on CUDA.")
    parser.add_argument("--with", dest="extra", nargs="+", default=[],
        help="Overrides added to every command, e.g. wandb.mode=online.")
    parser.add_argument("--skip_duplicates", action="store_true",
        help="If True, seeds that are already in the queue, or that already "
             "produced data.npz, are skipped. Nothing is submitted if squeue "
             "cannot be queried.")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    jobs = [(hp, seeds) for hp in expand_sweep(*load_sweep(args.sweep))]

    submit_jobs(
        jobs,
        args.data_dir,
        args.cluster,
        debug=args.debug,
        cuda=args.cuda,
        seeds_per_chunk=args.seeds_per_chunk,
        extra=args.extra,
        skip_duplicates=args.skip_duplicates,
    )
