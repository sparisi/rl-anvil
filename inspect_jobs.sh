#!/usr/bin/env bash
#
# Usage:
#   bash inspect_jobs.sh                       # inspect every RUNNING job for $USER
#   bash inspect_jobs.sh <JOBID> [JOBID ...]   # inspect one or more specific jobs
#   bash inspect_jobs.sh {100..105}            # bash brace expansion -> 100 101 ... 105
#   bash inspect_jobs.sh 100..105              # same as above
#   bash inspect_jobs.sh --fit 101 102 103     # + fit BASE_MEM_MB / PER_SEED_MEM_MB
#   bash inspect_jobs.sh --progress_only       # one row per job: steps and time
#   bash inspect_jobs.sh --no-cgroup 101       # skip the overlap step (see below)
#   bash inspect_jobs.sh -s my_logs 101        # $SCRATCH subfolder for the cgroup
#                                              # cache (default: slurm_out)
#
# Any number of job ids may be given, space-separated (same arg style as
# `scancel`). A token of the form `<id1>..<id2>` is expanded internally
# to every integer id in [id1, id2].
#
# For each job it:
#   1. Resolves JobName / StdOut / StdErr / WorkDir via `scontrol show job` while
#      the job is in the controller's memory; once it's gone, falls back to
#      `sacct` + guessed filenames (slurm-<jid>.{err,out}, <jid>.{err,out}).
#      The name is "<run_id>_<seeds>" (see the submission script): it says which
#      configuration the job runs and where its data goes (<data_dir>/<run_id>).
#   2. Prints stderr (if non-empty) so failures surface immediately.
#   3. Prints an `sacct` summary -- state, exit code, memory, elapsed vs limit.
#   4. Prints a [progress] digest: steps done out of the configuration's
#      training_steps, and elapsed out of the wallclock limit. The step count is
#      read from the .out files, where experiment.py prints it at every
#      checkpoint when experiment.progress_report is "dict"; the total comes
#      from the run's cfg.yaml, located by the "Data will be saved at ..." line
#      the same .out carries among its first lines. Where each task has its own
#      .out, a task that has logged no step counts as zero, so the digest never
#      reports progress the slowest seed has not made. A step-less task whose
#      submitit result pickle (<jobid>_<task>_result.pkl) sits beside its .out
#      has ended rather than fallen behind, and is left out of the count,
#      provided another task has logged a step; where none has, the line
#      reports how many tasks have ended and nothing more. A task that ended
#      AFTER a checkpoint is not excluded, and holds the minimum at the step it
#      stopped on. Catching a task that wrote no file at all needs the task
#      count as well, which sacct and sstat do not always report; and a layout
#      that puts every task in ONE shared .out supports neither check, the
#      digest there reporting whichever task logged last. Wherever a check is
#      unavailable the line says so.
#   5. Prints an [analysis] block that says what all those numbers mean.
#
# PROGRESS ONLY (--progress_only):
#
#   Reduces every job to one row of a table -- job id, steps done out of the
#   configuration's training_steps, elapsed out of the wallclock limit, and how
#   much longer the remaining steps are projected to take:
#
#       Job        Steps                     Time                  Left
#       1445886_2  920000 / 1000000 (92%)    1d3h / 1d16h (69%)    2h21m
#       1445886_3  1000000 / 1000000 (100%)  1d16h / 1d16h (100%)  0s
#
#   Left divides the steps still to do by the rate the job has averaged over
#   its whole elapsed time. That rate includes container setup and warmup, when
#   no steps were running, so the estimate leans long -- and it assumes a rate
#   that in truth drifts with what the run is doing. It needs a step count to
#   divide by, so a job whose progress reads zero gets "?" rather than a
#   projection from nothing, as does one whose total is unknown.
#
#   Step 4 above is the whole of it: no stderr, no memory, no analysis, and no
#   cgroup overlap step. The caveats step 4 can raise still appear, trailing
#   the row they belong to, so a figure is never shown as more than it is.
#   Incompatible with --fit, which needs the memory numbers this skips.
#
#   The table prints once every job has been read, not a row at a time: both
#   columns are sized to the widest row, and a finished run is the widest of
#   all. Inspecting many jobs therefore shows nothing until the last is done.
#
# MEMORY, and why there are three numbers:
#
#   sacct/sstat MaxRSS is the largest SINGLE TASK of the step. Scaling it by
#   NTasks is the documented heuristic, but it over-counts twice over: it
#   assumes every seed peaked at the worst seed's value simultaneously, and RSS
#   counts the file-backed torch/numpy pages once PER TASK even though the
#   cgroup is charged for them once. It also excludes the .batch step, which
#   competes for the same --mem. So it is an upper bound, not a measurement.
#
#   AveRSS x NTasks fixes the first problem (it is the mean over the step's
#   tasks, not the max) but not the second. Reported as a second opinion.
#
#   The cgroup's ANON is the number to size from: it charges shared pages once,
#   includes every process in the job, and counts only memory that cannot be
#   reclaimed without swap -- which is what the OOM killer ends up fighting over.
#   It can only be read from the node while the job RUNS (the cgroup is
#   destroyed at the end), so this script reads it by launching a tiny
#   overlapping step into the live allocation:
#       srun --jobid=<id> --overlap --ntasks=1 ... cat .../memory.stat
#   That step costs a few MB of bash inside the same cgroup -- noise against a
#   multi-GB job, but pass --no-cgroup to skip it if your site forbids overlap
#   steps or you are inspecting many jobs at once.
#
#   Do NOT size from memory.peak, which is what this script used to report. It
#   is the high-water mark of memory.current, and memory.current counts page
#   cache: a run that faults libtorch off Lustre and writes .npz files fills
#   whatever --mem leaves free with cache that the kernel has no reason to drop,
#   so memory.peak pegs at 100% of the limit on jobs using a fraction of it. The
#   honest distress signal is memory.events `max` (reported as hit_max) -- the
#   count of times the kernel actually had to reclaim to stay under the limit.
#
#   anon is an instantaneous sample, not a high-water mark: the kernel exposes
#   no peak for anon alone. This script keeps the max across every poll, so run
#   it on a loop for a good estimate, and remember a spike between two polls is
#   invisible here. experiment.py records `mem_%` from inside the run, at every
#   checkpoint, for exactly that reason -- prefer it when you have it.
#
# FITTING (--fit):
#
#   Give several jobs of the SAME configuration that differ only in
#   --seeds_per_chunk. The script collects (tasks, total_peak) from each and
#   least-squares fits total = BASE + PER_SEED * tasks, then prints the two
#   constants to paste into submit_jobs.py. Rules for a fit worth trusting:
#     - same env/algorithm/overrides in every job (check the Name field: the
#       run_id must match, only the seed range should differ)
#     - jobs COMPLETED, or at least past the point where memory plateaus. A
#       replay buffer that is still filling makes a big chunk look cheap.
#     - request generous memory on the profiling runs (~2x production). A job
#       running near its cap gets reclaimed hard and reports back your own
#       limit rather than its demand.
#     - at least two distinct task counts; three or more also tells you whether
#       the relationship is actually linear.
#
# The sacct section runs even when the job wrote no logs at all: one that died
# before printing anything is exactly the one whose accounting you want to read.
#
# Exit status is 0 when every inspected job succeeded, otherwise the non-zero
# return of the last failing inspection.

set -u

DO_FIT=0
USE_CGROUP=1
PROGRESS_ONLY=0

# One --progress_only row per inspected job, collected by inspect_job and
# printed by progress_table once every job has been read. Nothing can be
# printed earlier: the column widths are measured over the rows themselves,
# and a width guessed in advance is a column that the first row to outgrow it
# either collides with or pushes out of alignment.
ROW_JOB=()
ROW_STEPS=()
ROW_TIME=()
ROW_LEFT=()
ROW_EXTRA=()

# Append one row: job id, steps cell, time cell, time-left cell, trailing caveats.
#
# Every job asked about gets one, including the ones that could not be read at
# all. A table with fewer rows than the ids given is a job dropped in silence,
# and the reader has no way to tell which.
progress_row() {
    ROW_JOB+=("$1")
    ROW_STEPS+=("$2")
    ROW_TIME+=("$3")
    ROW_LEFT+=("$4")
    ROW_EXTRA+=("${5:-}")
}

# $SCRATCH subfolder the cgroup cache is written to.
SLURM_OUT="slurm_out"

# Collected by inspect_job, consumed by fit_memory_model.
FIT_TASKS=()
FIT_KB=()
FIT_SRC=()
FIT_STATE=()
FIT_JOB=()
FIT_NAME=()
FIT_DROPPED=()
FIT_NO_TASKS=()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
get_field() {
    # extract a "Key=Value" field from scontrol output (values have no spaces)
    local key="$1" text="$2"
    echo "$text" | grep -oE "${key}=[^[:space:]]+" | head -n1 | cut -d= -f2-
}

# SLURM memory strings ("989688K", "1.50G", "8G", or bare digits meaning KB)
# to kilobytes, so requested and peak can be compared numerically.
mem_to_kb() {
    local v="$1" num unit
    [[ -z "$v" || "$v" == "?" ]] && return 1
    num="${v//[^0-9.]/}"
    unit="${v//[0-9.]/}"
    [[ -z "$num" ]] && return 1
    # Always an integer: bash arithmetic cannot parse a fraction, and sacct
    # reports one whenever a value is an average over tasks that does not divide
    # evenly (AveRSS=576654.50K over 8 tasks). Left as-is it aborts the caller
    # mid-job, which silently drops that job from --fit.
    awk -v n="$num" -v u="$unit" 'BEGIN{
        u = toupper(substr(u,1,1))
        if (u=="T") v = n*1024*1024*1024
        else if (u=="G") v = n*1024*1024
        else if (u=="M") v = n*1024
        else v = n
        printf "%.0f", v
    }'
}

kb_to_mb() { awk -v k="$1" 'BEGIN{ printf "%.0f", k/1024 }'; }

# SLURM durations (D-HH:MM:SS, HH:MM:SS or MM:SS) to seconds. Anything
# non-numeric (UNLIMITED, Partition_Limit, "?") fails, so an unknown limit is
# never reported as a job about to be killed.
hms_to_s() {
    local t="$1" d=0 h=0 m=0 s=0
    [[ "$t" == *-* ]] && { d="${t%%-*}"; t="${t#*-}"; }
    case "$t" in
        *:*:*) IFS=: read -r h m s <<< "$t" ;;
        *:*)   IFS=: read -r m s <<< "$t" ;;
        *)     return 1 ;;
    esac
    [[ "$d$h$m$s" =~ ^[0-9]+$ ]] || return 1
    echo $(( 10#$d*86400 + 10#$h*3600 + 10#$m*60 + 10#$s ))
}

pct() {  # $1 as a percentage of $2, rounded
    awk -v a="$1" -v b="$2" 'BEGIN{ printf "%.0f", (b>0 ? 100*a/b : 0) }'
}

# Seconds as a compact two-unit duration: "1d2h", "2h3m", "13m4s", "45s".
s_to_compact() {
    awk -v s="$1" 'BEGIN{
        s = int(s)
        d = int(s/86400); s -= d*86400
        h = int(s/3600);  s -= h*3600
        m = int(s/60);    s -= m*60
        if (d)      printf "%dd%dh", d, h
        else if (h) printf "%dh%dm", h, m
        else if (m) printf "%dm%ds", m, s
        else        printf "%ds", s
    }'
}

# Environment steps the configuration asks for, as "<steps>\t<ambiguous>", read
# from the cfg.yaml the run wrote beside its data. <ambiguous> is 1 when the
# file sets training_steps more than once with different values: the first is
# returned and the caller says so, rather than one of them being picked
# silently.
#
# Where that is comes out of the job's own log: prepare_run() prints
# "Data will be saved at <data_dir>/<config_id>/<rng_seed>" among the first
# lines of the run, and it is the only record of the location. results.data_dir
# reaches the job as a Hydra override, nothing the scheduler can be asked for
# keeps it, and cfg.yaml itself does not carry it. The file sits one level up
# from the seed folder, in the <config_id> directory the seeds share.
#
# rich wraps at 80 columns when stdout is a file, so a long path arrives folded
# over several lines. Continuations are glued back on one at a time until the
# result names a directory that really holds a cfg.yaml.
training_steps_from_out() {
    local out="$1" workdir="$2" cand="" dir line parts n=0
    [[ -s "$out" ]] || return 1

    # The message is among the first lines of the run: reading the whole log of
    # a job that has been checkpointing for hours to find it would be waste.
    parts=$(head -n 50 "$out" | awk '
        found { print; next }
        /Data will be saved at/ {
            sub(/.*Data will be saved at[[:space:]]*/, "")
            print; found = 1
        }')
    [[ -n "$parts" ]] || return 1

    while IFS= read -r line; do
        cand+="${line//[[:space:]]/}"
        dir="${cand%/*}"
        [[ "$dir" == /* ]] || dir="$workdir/$dir"
        if [[ -r "$dir/cfg.yaml" ]]; then
            awk '
                $1 == "training_steps:" {
                    v = $2
                    sub(/#.*/, "", v)
                    gsub(/_/, "", v)
                    if (v !~ /^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$/) next
                    if (found) {
                        if (v + 0 != val) multi = 1
                        next
                    }
                    val = v + 0
                    found = 1
                }
                END {
                    if (!found) exit 1
                    printf "%.0f\t%d\n", val, multi + 0
                }
            ' "$dir/cfg.yaml"
            return
        fi
        (( ++n < 4 )) || break
    done <<< "$parts"
    return 1
}

# Report checkpoint progress over a job's .out files as
# "<files with a step>\t<min>\t<max>[\t<step-less path>...]".
#
# The step count exists nowhere else: no accounting command knows what the run
# is doing, only experiment.py does, and it prints the count in the dict it
# writes at every checkpoint under progress_report="dict". A log written in any
# other mode carries no such line and contributes nothing here.
#
# Only the LAST such line of each file is wanted, so each file is read from the
# end. These logs grow for the whole run, and scanning them forward is the most
# expensive thing this script does.
#
# submitit writes one .out per task, so a job running several seeds reports a
# range. A file with no step line at all is counted in none of the three
# numbers: the caller compares the count against the number of TASKS -- which
# is not the number of files, since a task that has yet to start has written
# none -- and treats the shortfall, less any task that has already ended, as
# zero, because the allocation lives until the slowest task is done.
#
# The files with no step line are named in the trailing fields, so a caller
# that needs them does not have to find them again: they are the only ones this
# function reads end to end.
steps_from_outs() {
    local f s n=0 mn=0 mx=0 nostep=()
    for f in "$@"; do
        # head -n1: -m1 stops after the first matching LINE, and -o then prints
        # every match on it. Two step counts reaching one line -- interleaved
        # writes to a shared log -- would otherwise have their digits run
        # together into a number neither task ever reached.
        s=$(tac "$f" 2>/dev/null | grep -m1 -oE "'steps': *[0-9]+" | head -n1 | tr -dc '0-9')
        if [[ -z "$s" ]]; then
            nostep+=("$f")
            continue
        fi
        s=$(( 10#$s ))
        if (( ++n == 1 )); then
            mn=$s
            mx=$s
            continue
        fi
        (( s < mn )) && mn=$s
        (( s > mx )) && mx=$s
    done
    printf '%d\t%d\t%d' "$n" "$mn" "$mx"
    (( ${#nostep[@]} > 0 )) && printf '\t%s' "${nostep[@]}"
    printf '\n'
}

# ---------------------------------------------------------------------------
# Memory of the whole job's cgroup as
# "anon|peak|hit_max|oom_kill|file|kernel|seed_kb|seed_n|ovh_kb", or "" if
# unavailable. Sizes are KiB, hit_max/oom_kill/seed_n are counts.
#
# Walks up from the overlap step's own cgroup to the nearest `job_<id>`
# ancestor -- the level SLURM applies --mem at, and therefore the level whose
# numbers are the ones that matter.
#
# Four numbers, because the obvious one is a trap:
#
#   anon      Current anonymous bytes (memory.stat `anon`, v1 `rss`): heap,
#             numpy buffers, torch tensors. This is the memory that CANNOT be
#             reclaimed without swap, which compute nodes do not have, so it is
#             what --mem has to cover. Note it is a sample, not a high-water
#             mark -- cgroup exposes no peak for anon alone, so a spike between
#             two reads is invisible here (experiment.py records `mem_%` from
#             inside the run for exactly that reason).
#
#   peak      memory.peak (v1 memory.max_usage_in_bytes): the high-water mark
#             of memory.current, which counts page cache, kernel memory and
#             socket buffers as well as anon. A run that faults a few hundred MB
#             of libtorch off Lustre and writes .npz files fills whatever is
#             left of its limit with cache, and the kernel has no reason to
#             reclaim it until something asks -- so this pegs at 100% of --mem
#             on jobs that are nowhere near trouble. Reported for completeness.
#             Never size from it.
#
#   hit_max   memory.events `max` (v1 memory.failcnt): how many times the cgroup
#             hit its limit and the kernel had to reclaim to stay under it.
#             Nonzero is the real "--mem is too low" signal -- unlike `peak`,
#             cache filling up harmlessly does not move it.
#
#   oom_kill  memory.events `oom_kill`: processes actually killed. Nonzero means
#             the job died of memory, full stop.
#
# Plus the split that actually sizes the two constants, walked from the cgroup's
# own process list (find -name cgroup.procs) rather than pgrep, which scans the
# whole node and happily returns another job's python:
#
#   file      memory.stat `file` (v1 `cache`): page cache and mapped library
#             text. Charged ONCE however many seeds run, which is what makes it
#             BASE_MEM_MB rather than a per-seed cost.
#   kernel    memory.stat `kernel`: slab, pagetables, percpu. Also roughly
#             seed-count independent.
#   seed_kb   summed Anonymous of the processes running main.py, and seed_n how
#             many there are: seed_kb/seed_n is the per-seed anon cost.
#   ovh_kb    summed Anonymous of everything else in the cgroup -- the submitit
#             parent, the Apptainer squashfuse_ll mounts, srun and the wrapper
#             shells. Roughly 40 MB PER TASK that no one budgets for, so it
#             belongs in PER_SEED_MEM_MB, not in the base.
# ---------------------------------------------------------------------------
cgroup_stats() {
    local jobid="$1"
    (( USE_CGROUP )) || return 1
    command -v srun >/dev/null 2>&1 || return 1

    local out rc
    out=$(timeout 20 srun --jobid="$jobid" --overlap --ntasks=1 --cpus-per-task=1 --quiet --job-name=cgpeek bash -c '
        rel=$(awk -F: "{print \$NF}" /proc/self/cgroup | head -n1)
        p="/sys/fs/cgroup${rel}"
        while [ -n "$p" ] && [ "$p" != "/sys/fs/cgroup" ]; do
            case "${p##*/}" in
                job_*)
                    anon=$(grep -m1 "^anon " "$p/memory.stat" 2>/dev/null | cut -d" " -f2)
                    [ -n "$anon" ] || anon=$(grep -m1 "^rss " "$p/memory.stat" 2>/dev/null | cut -d" " -f2)
                    file=$(grep -m1 "^file " "$p/memory.stat" 2>/dev/null | cut -d" " -f2)
                    [ -n "$file" ] || file=$(grep -m1 "^cache " "$p/memory.stat" 2>/dev/null | cut -d" " -f2)
                    kern=$(grep -m1 "^kernel " "$p/memory.stat" 2>/dev/null | cut -d" " -f2)
                    peak=""
                    for f in memory.peak memory.max_usage_in_bytes; do
                        if [ -r "$p/$f" ]; then
                            peak=$(cat "$p/$f")
                            [ -n "$peak" ] && break
                        fi
                    done
                    hitmax=$(grep -m1 "^max " "$p/memory.events" 2>/dev/null | cut -d" " -f2)
                    [ -n "$hitmax" ] || hitmax=$(cat "$p/memory.failcnt" 2>/dev/null)
                    oomk=$(grep -m1 "^oom_kill " "$p/memory.events" 2>/dev/null | cut -d" " -f2)
                    seed=0; seedn=0; ovh=0
                    for pid in $(find "$p" -name cgroup.procs -exec cat {} + 2>/dev/null | sort -un); do
                        a=$(sed -n "s/^Anonymous: *\([0-9]*\).*/\1/p" /proc/$pid/smaps_rollup 2>/dev/null)
                        [ -n "$a" ] || continue
                        # TODO: "main.py" is hardcoded. If the entry point is renamed, processes will be misclassified as overhead.
                        if tr -d "\000" < /proc/$pid/cmdline 2>/dev/null | grep -q "main\.py"; then
                            seed=$(( seed + a )); seedn=$(( seedn + 1 ))
                        else
                            ovh=$(( ovh + a ))
                        fi
                    done
                    if [ -n "$anon" ] || [ -n "$peak" ]; then
                        echo "$(( ${anon:-0} / 1024 ))|$(( ${peak:-0} / 1024 ))|${hitmax:-0}|${oomk:-0}|$(( ${file:-0} / 1024 ))|$(( ${kern:-0} / 1024 ))|$seed|$seedn|$ovh"
                        exit 0
                    fi
                    ;;
            esac
            p="${p%/*}"
        done
        exit 1' 2>/dev/null)
    rc=$?
    out="${out//[[:space:]]/}"

    if (( rc == 124 )); then
        echo "TIMEOUT"
        return 1
    fi

    [[ "$out" =~ ^([0-9]+\|){8}[0-9]+$ ]] || return 1
    echo "$out"
}

cgroup_cache_file() { echo "${SCRATCH:-/tmp}/${SLURM_OUT}/$1.cgpeak"; }

# The cgroup dies with the job, but the fit wants COMPLETED jobs -- without a
# cache those two requirements are mutually exclusive and --fit silently falls
# back to MaxRSS x tasks.
#
# `peak` and the event counters are monotonic, so their last write is their true
# value. `anon` is NOT: it is an instantaneous sample that falls as well as
# rises, so keeping the last write would report whatever the job happened to
# hold at the final poll. Every field is therefore merged as a running MAX, and
# polling on a loop only improves the anon estimate.
#
# A cache file written by an older version holds a bare memory.peak, which fails
# the tuple check below and is treated as absent -- the right outcome, since it
# cannot be told apart from an anon reading and sizing from it is the bug this
# format change fixes.
cgroup_cache_write() {
    local f prev
    f=$(cgroup_cache_file "$1")
    mkdir -p "$(dirname "$f")" 2>/dev/null || return 0
    # Keep the WHOLE record from the poll with the largest anon, never a per-field
    # maximum. Maxing fields independently mixes timepoints: anon from one poll,
    # seed_kb from another, file from a third. The fields then stop adding up,
    # and `anon - seed_kb - ovh_kb` reports hundreds of MB of "unattributed"
    # memory that never existed.
    if prev=$(cgroup_cache_read "$1"); then
        local new_anon old_anon
        new_anon=${2%%|*}
        old_anon=${prev%%|*}
        (( new_anon < old_anon )) && return 0
    fi
    echo "$2" > "$f" 2>/dev/null
}

cgroup_cache_read() {
    local f v; f=$(cgroup_cache_file "$1")
    [[ -r "$f" ]] || return 1
    v=$(tr -d '[:space:]' < "$f")
    [[ "$v" =~ ^([0-9]+\|){8}[0-9]+$ ]] || return 1
    echo "$v"
}

inspect_job() {
    local JOBID="$1"
    local STDERR="" STDOUT="" WORKDIR="" NAME="" ARRAY_ID=""

    # -------------------------------------------------------------------
    # 1) Resolve StdErr / StdOut / WorkDir.
    #    scontrol works while the job is in the controller's memory;
    #    once it's gone we fall back to sacct + reconstruction from
    #    the work dir.
    # -------------------------------------------------------------------
    local SCONTROL_OUT
    if SCONTROL_OUT=$(scontrol show job "$JOBID" 2>/dev/null); then
        STDERR=$(get_field "StdErr"  "$SCONTROL_OUT")
        STDOUT=$(get_field "StdOut"  "$SCONTROL_OUT")
        WORKDIR=$(get_field "WorkDir" "$SCONTROL_OUT")
        NAME=$(get_field "JobName" "$SCONTROL_OUT")
        local AJID ATID
        AJID=$(get_field "ArrayJobId" "$SCONTROL_OUT")
        ATID=$(get_field "ArrayTaskId" "$SCONTROL_OUT")
        [[ -n "$AJID" && -n "$ATID" ]] && ARRAY_ID="${AJID}_${ATID}"
    fi

    if [[ -z "$NAME" ]] && command -v sacct >/dev/null 2>&1; then
        NAME=$(sacct -j "$JOBID" -o JobName%200 -n -P </dev/null 2>/dev/null | head -n1 | xargs)
    fi

    if [[ -z "$STDERR" || -z "$STDOUT" ]]; then
        local SACCT_OUT
        SACCT_OUT=$(sacct -j "$JOBID" -o WorkDir%500 -n -P </dev/null 2>/dev/null | head -n1 | xargs)
        if [[ -n "$SACCT_OUT" ]]; then
            WORKDIR="$SACCT_OUT"
            local cand
            for cand in "$WORKDIR/slurm-$JOBID.err" "$WORKDIR/slurm-${JOBID}.out" "$WORKDIR/${JOBID}.err" "$WORKDIR/${JOBID}.out"; do
                [[ -z "$STDERR" && "$cand" == *.err && -f "$cand" ]] && STDERR="$cand"
                [[ -z "$STDOUT" && "$cand" == *.out && -f "$cand" ]] && STDOUT="$cand"
            done
        fi
    fi

    if (( ! PROGRESS_ONLY )); then
        echo "Job:    $JOBID"
        [[ -n "$NAME"    ]] && echo "Name:    $NAME"
        [[ -n "$STDOUT"  ]] && echo "StdOut:  $STDOUT"
        [[ -n "$STDERR"  ]] && echo "StdErr:  $STDERR"
        [[ -n "$WORKDIR" ]] && echo "WorkDir: $WORKDIR"
        echo
    fi

    # Print stderr if non-empty -- useful to surface errors / warnings.
    if (( ! PROGRESS_ONLY )) && [[ -n "$STDERR" && -f "$STDERR" && -s "$STDERR" ]]; then
        echo "===== stderr ($STDERR) ====="
        cat "$STDERR"
        echo "============================"
        echo
    fi

    # -------------------------------------------------------------------
    # 2) sacct summary for the job: state, exit code, and the memory/time
    #    digest (requested vs allocated vs peak, elapsed vs limit),
    #    then what those numbers mean.
    #    For a RUNNING job these are live/partial (peak not yet final).
    # -------------------------------------------------------------------
    if ! command -v sacct >/dev/null 2>&1; then
        (( PROGRESS_ONLY )) && progress_row "$JOBID" "?" "?" "?" "sacct not on PATH, so nothing about this job could be read"
        return 0
    fi

    local SACCT_RAW
    SACCT_RAW=$(sacct -j "$JOBID" -S now-90days --format=JobID,State,ExitCode,ReqMem,MaxRSS,MaxVMSize,Elapsed,Timelimit,ReqTRES%40,AllocTRES%40,NTasks,AveRSS,TRESUsageInTot%80 -P </dev/null 2>/dev/null)

    if [[ -z "$SACCT_RAW" ]] || (( $(printf '%s\n' "$SACCT_RAW" | tail -n +2 | grep -c .) == 0 )); then
        if (( PROGRESS_ONLY )); then
            progress_row "$JOBID" "?" "?" "?" "no sacct records (older than 90d or purged)"
        else
            echo "[sacct $JOBID: no records (older than 90d or purged)]"
        fi
        return 0
    fi

    # One awk pass emits the fields as TSV; the digest line and the analysis are
    # both formatted from them, so the table is parsed once. The fields are
    # spread across rows: State/ExitCode/ReqMem and the TRES columns sit on the
    # job row, while MaxRSS/AveRSS/MaxVMSize populate only on the .batch/.0 rows.
    local ST EC REQ ALLOC PEAK CPUS ELAPSED TLIMIT NTASKS AVE BATCH TRESTOT_KB
    IFS=$'\t' read -r ST EC REQ ALLOC PEAK CPUS ELAPSED TLIMIT NTASKS AVE BATCH TRESTOT_KB < <(
        printf '%s\n' "$SACCT_RAW" | awk -F'|' '
            function tres(s, key,   i, a, n) {
                n = split(s, a, ",")
                for (i = 1; i <= n; i++)
                    if (a[i] ~ "^" key "=") { sub("^" key "=", "", a[i]); return a[i] }
                return ""
            }
            function tokb(m,   mn, u) {
                mn = m + 0; u = m; gsub(/[0-9.]/, "", u)
                return (u ~ /^T/ ? mn*1024*1024*1024 : (u ~ /^G/ ? mn*1024*1024 : (u ~ /^M/ ? mn*1024 : mn)))
            }
            NR == 1 { next }
            {
                if ($2 != "" && state == "")  state = $2
                if ($3 != "" && ecode == "")  ecode = $3
                if ($4 != "" && reqmem == "") reqmem = $4
                if (reqtres == "")   reqtres = tres($9, "mem")
                if (alloctres == "") alloctres = tres($10, "mem")
                if (alloccpu == "")  alloccpu = tres($10, "cpu")
                # MaxRSS/AveRSS must come from the srun step (.0, .1, ...) ONLY.
                # Taking the max over every row lets a .batch step that peaked
                # higher than the seeds win, and it then gets multiplied by
                # NTasks -- inventing memory that was never held. .batch is a
                # single process; it is counted once, separately, below.
                if ($1 ~ /\.batch$/ && $5 != "") batchrss = $5
                # TRESUsageInTot mem is the SUM over the tasks of the step, not
                # the biggest one -- the number MaxRSS cannot give. With
                # JobAcctGatherType=jobacct_gather/cgroup it comes from the
                # per-task cgroups, so a page shared by several tasks is charged
                # once and the sum is a real total. Add .batch: a separate step,
                # but the same --mem pool.
                if ($1 ~ /\.batch$/) trestot_kb += tokb(tres($13, "mem"))
                if ($1 ~ /\.[0-9]+$/) {
                    trestot_kb += tokb(tres($13, "mem"))
                    if ($5 != "") {
                        kb = tokb($5)
                        if (kb + 0 > maxrss_kb + 0) { maxrss_kb = kb; maxrss = $5 }
                    }
                    if ($12 != "") {
                        kb = tokb($12)
                        if (kb + 0 > averss_kb + 0) { averss_kb = kb; averss = $12 }
                    }
                }
                # Elapsed comes from the job row, whose 00:00:00 is a real
                # reading. Rejecting it everywhere drops the time digest from
                # the jobs it is most wanted on, the ones that just started.
                if ($1 !~ /\./) { if (elapsed == "") elapsed = $7 }
                else if (elapsed_step == "" && $7 != "" && $7 != "00:00:00") elapsed_step = $7
                if ($8 != "" && tlimit == "") tlimit = $8
                # NTasks comes from the srun steps (.0, .1, ...) only, the same
                # rule MaxRSS follows above. .batch and .extern say 1 whatever
                # the job is doing, and a 1 taken from them cannot be told apart
                # from a single-task job reporting its real count -- which is
                # the distinction everything below this leans on.
                if ($1 ~ /\.[0-9]+$/ && $11 + 0 > ntasks + 0) ntasks = $11 + 0
            }
            END {
                if (elapsed == "") elapsed = elapsed_step
                req = (reqtres != "" ? reqtres : reqmem)
                printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                       (state != "" ? state : "?"), (ecode != "" ? ecode : "?"),
                       (req != "" ? req : "?"), (alloctres != "" ? alloctres : "?"),
                       (maxrss != "" ? maxrss : "?"), (alloccpu != "" ? alloccpu : "?"),
                       (elapsed != "" ? elapsed : "?"), (tlimit != "" ? tlimit : "?"),
                       (ntasks + 0), (averss != "" ? averss : "?"),
                       (batchrss != "" ? batchrss : "?"), (trestot_kb + 0)
            }')

    # sacct writes MaxRSS only when a step ENDS, so a running job always reports
    # "?" no matter how long it has been going. sstat reads the same counters
    # live from the running steps -- it works only while the job runs, and only
    # on steps that exist (.batch, .extern, and the srun step .0).
    #
    # Only the srun steps (.0, .1, ...) count, the same rule the sacct path
    # above follows. .batch is the wrapper shell and .extern the container step:
    # taking the max across every row reports bash's ~20 MB as the job's peak
    # whenever the python step has not registered its counters yet, which reads
    # as "this run uses no memory" instead of "no measurement yet".
    local PEAK_SRC="sacct"
    if [[ "$PEAK" == "?" ]] && command -v sstat >/dev/null 2>&1; then
        local LIVE LIVE_PEAK LIVE_AVE LIVE_NTASKS
        LIVE=$(sstat -a -j "$JOBID" --format=JobID,MaxRSS,AveRSS,NTasks -P -n </dev/null 2>/dev/null | awk -F'|' '
                $1 ~ /\.[0-9]+$/ && $2 != "" && $2 != "N/A" {
                    m = $2; mn = m + 0; u = m; gsub(/[0-9.]/, "", u)
                    kb = (u ~ /^G/ ? mn*1024*1024 : (u ~ /^M/ ? mn*1024 : mn))
                    if (kb + 0 > best + 0) { best = kb; bs = m; ba = $3; bn = $4 + 0 }
                }
                END { printf "%s\t%s\t%s\n", bs, ba, (bn + 0) }')
        IFS=$'\t' read -r LIVE_PEAK LIVE_AVE LIVE_NTASKS <<< "$LIVE"
        if [[ -n "$LIVE_PEAK" ]]; then
            PEAK="$LIVE_PEAK"
            PEAK_SRC="sstat"
            # Only when sstat reported one: overwriting with a default would
            # discard whatever sacct managed to read.
            [[ "$LIVE_NTASKS" =~ ^[0-9]+$ ]] && (( LIVE_NTASKS > 0 )) && NTASKS="$LIVE_NTASKS"
            [[ -n "$LIVE_AVE" && "$LIVE_AVE" != "N/A" ]] && AVE="$LIVE_AVE"
        fi
    fi

    # One task (the joblib layout, where the seeds are workers inside a single
    # task) is the fallback whenever NTasks is missing or unreadable. The two
    # are kept apart: a count nothing reported carries none of the authority of
    # a count that says 1, and the progress digest needs to know which it has.
    local NTASKS_KNOWN=1
    [[ "$NTASKS" =~ ^[0-9]+$ ]] && (( NTASKS >= 1 )) || { NTASKS=1; NTASKS_KNOWN=0; }

    # What this job contributes to the script's exit status, so a caller can
    # gate on it without parsing the report.
    #
    # A job that reached a terminal state other than COMPLETED failed, whatever
    # killed it. The states that have not finished are not failures and must
    # not read as any: inspecting a healthy sweep mid-flight is the ordinary
    # use, and a non-zero status there would make the check useless. Nor is a
    # state nothing reported -- "?" is the absence of a verdict, not a bad one.
    local JOB_RC=0
    case "$ST" in
        COMPLETED*|PENDING*|RUNNING*|COMPLETING*|SUSPENDED*|RESIZING*) : ;;
        CONFIGURING*|REQUEUED*|REQUEUE_*|SPECIAL_EXIT*)                : ;;
        STAGE_OUT*|SIGNALING*|STOPPED*|RESV_DEL_HOLD*)                 : ;;
        "?")                                                           : ;;
        *)                                                             JOB_RC=1 ;;
    esac

    # The cgroup numbers. Read live if the job is still up, otherwise recovered
    # from the cache a previous inspection wrote. CG_SRC is set on the branch
    # actually taken: a RUNNING job whose overlap step is refused still falls
    # back to the cache, and saying "live" then would hide that it is stale.
    local CG="" CG_KB="" CG_PEAK_KB="" CG_HITMAX="" CG_OOM="" CG_SRC=""
    local CG_FILE_KB="" CG_KERN_KB="" CG_SEED_KB="" CG_SEED_N="" CG_OVH_KB=""
    if [[ "$ST" == RUNNING* ]]; then
        CG=$(cgroup_stats "$JOBID")
        if [[ "$CG" == "TIMEOUT" ]]; then
            echo "[cgroup $JOBID] WARNING: srun overlap step timed out after 20s. Falling back to cache." >&2
            CG=""
        fi
        if [[ -n "$CG" ]]; then
            CG_SRC="live"
            cgroup_cache_write "$JOBID" "$CG"
            CG=$(cgroup_cache_read "$JOBID") || CG=""   # merged max over every poll
        else
            CG=$(cgroup_cache_read "$JOBID") || CG=""
            [[ -n "$CG" ]] && CG_SRC="cached"
        fi
    else
        CG=$(cgroup_cache_read "$JOBID") || CG=""
        [[ -n "$CG" ]] && CG_SRC="cached"
    fi

    [[ -n "$CG" ]] && IFS='|' read -r CG_KB CG_PEAK_KB CG_HITMAX CG_OOM CG_FILE_KB CG_KERN_KB CG_SEED_KB CG_SEED_N CG_OVH_KB <<< "$CG"

    local PEAK_LABEL="$PEAK"
    [[ "$PEAK_SRC" == "sstat" ]] && PEAK_LABEL="$PEAK (live, sstat)"

    local TASKS_LABEL="$NTASKS"
    (( NTASKS_KNOWN )) || TASKS_LABEL="$NTASKS (assumed, none reported)"

    (( PROGRESS_ONLY )) || echo "[sacct $JOBID] state=$ST exit=$EC | req_mem=$REQ alloc_mem=$ALLOC peak_rss=$PEAK_LABEL ave_rss=$AVE tasks=$TASKS_LABEL cpus=$CPUS | elapsed=$ELAPSED limit=$TLIMIT"
    if [[ -n "$CG_KB" ]] && (( ! PROGRESS_ONLY )); then
        echo "[cgroup $JOBID] anon = $(kb_to_mb "$CG_KB")M | file = $(kb_to_mb "$CG_FILE_KB")M | kernel = $(kb_to_mb "$CG_KERN_KB")M | hit_max = $CG_HITMAX | oom_kill = $CG_OOM  ($CG_SRC, max over polls)"
        echo "[cgroup $JOBID] memory.peak = $(kb_to_mb "$CG_PEAK_KB")M -- counts page cache, pegs at the limit on healthy jobs, NOT for sizing"
        if (( CG_SEED_N > 0 )); then
            # A misclassified process moves memory between seed and overhead
            # without changing what neither accounts for, so this figure holds
            # whether or not the split can be trusted.
            local unattr_kb=$(( CG_KB - CG_SEED_KB - CG_OVH_KB ))
            (( unattr_kb < 0 )) && unattr_kb=0
            if (( ! NTASKS_KNOWN || CG_SEED_N != NTASKS )); then
                if (( NTASKS_KNOWN )); then
                    echo "[anon   $JOBID] WARNING: classified $CG_SEED_N seed process(es) but sacct reports $NTASKS task(s)." >&2
                    echo "                 The main.py match missed some, or the job is not laid out as assumed. The" >&2
                    echo "                 per-seed split is unreliable and has been omitted to prevent incorrect sizing." >&2
                else
                    echo "[anon   $JOBID] WARNING: classified $CG_SEED_N seed process(es) and neither sacct nor sstat" >&2
                    echo "                 reported a task count to check them against. The per-seed split is" >&2
                    echo "                 unverifiable and has been omitted to prevent incorrect sizing." >&2
                fi
                echo "[anon   $JOBID] unattributed $(kb_to_mb "$unattr_kb")M (does not depend on the withheld split)"
            else
                echo "[anon   $JOBID] $CG_SEED_N seed(s) x ~$(kb_to_mb $(( CG_SEED_KB / CG_SEED_N )))M = $(kb_to_mb "$CG_SEED_KB")M | overhead $(kb_to_mb "$CG_OVH_KB")M (submitit + squashfuse + srun, ~$(kb_to_mb $(( CG_OVH_KB / CG_SEED_N )))M per task) | unattributed $(kb_to_mb "$unattr_kb")M"
            fi
        else
            # Every other unavailable check in this script announces itself.
            # Without this one a renamed entry point reads as a job whose cgroup
            # was never readable, which is the one thing it is not.
            echo "[anon   $JOBID] WARNING: no process in the cgroup matched the entry point, so every" >&2
            echo "                 one of them counted as overhead and there is no split to report. The" >&2
            echo "                 classifier looks for main.py; if the entry point was renamed, this" >&2
            echo "                 line is the only sign of it." >&2
        fi
    fi

    # -------------------------------------------------------------------
    # 3) Progress: how much of the run is done, and how much of the wallclock
    #    it took to get there. The two are printed together because neither
    #    bounds the other -- a job at 40% of its steps and 80% of its limit is
    #    one that will be killed before it finishes, and only the pair says so.
    # -------------------------------------------------------------------
    # Each branch below fills both a short cell for the --progress_only table
    # and the long line the default output prints, so the two modes can never
    # disagree about what was measured.
    local el_s lim_s steps_cell="" steps_long="" time_cell="?" eta_cell="?" extra="" ts_multi=""
    local total_caveat="" time_caveat=""
    el_s=$(hms_to_s "$ELAPSED") || el_s=""
    lim_s=$(hms_to_s "$TLIMIT") || lim_s=""

    local OUTS=() cand prefix prefixes per_task_logs=0
    if [[ -n "$STDOUT" ]]; then
        # An array element's logs are named after the array job and task, which
        # is not the id scontrol reports as JobId and not necessarily the id
        # this job was asked about. The array id is tried first: when the id
        # asked about is the array JOB id, the plain glob also matches every
        # other element of the array, and pooling those into one progress line
        # would contradict the sacct and cgroup lines above it, which describe
        # a single element.
        prefixes=()
        [[ -n "$ARRAY_ID" && "$ARRAY_ID" != "$JOBID" ]] && prefixes+=("$ARRAY_ID")
        prefixes+=("$JOBID")
        for prefix in "${prefixes[@]}"; do
            for cand in "${STDOUT%/*}/${prefix}"_*_log.out; do
                [[ -f "$cand" ]] && OUTS+=("$cand")
            done
            # Where the two ids overlap at all, the narrower one is tried first
            # and already holds every file the other would add.
            (( ${#OUTS[@]} > 0 )) && { per_task_logs=1; break; }
        done
        # Non-submitit layouts write a single file under the name scontrol gave.
        # That one file carries every task, so its existence says nothing about
        # how many of them started.
        (( ${#OUTS[@]} == 0 )) && [[ -f "$STDOUT" ]] && OUTS=("$STDOUT")
    fi

    if [[ "$ST" == PENDING* ]]; then
        steps_cell="queued"
        steps_long="still queued -- nothing has run, so there is no log to read"
    elif (( ${#OUTS[@]} > 0 )); then
        local n_out=${#OUTS[@]} n_seen min_logged max_logged n_task unit n_ended=0 ended_note=""
        local nostep_fields="" NOSTEP=()
        IFS=$'\t' read -r n_seen min_logged max_logged nostep_fields < <(steps_from_outs "${OUTS[@]}")
        [[ -n "$nostep_fields" ]] && IFS=$'\t' read -ra NOSTEP <<< "$nostep_fields"
        # With one .out per task, a file missing from the glob is a task that
        # has not started, and sacct is the only thing that knows how many the
        # step has. The shared-log fallback has a single file whatever the task
        # count, so there the file count is all there is to count against.
        n_task=$n_out
        (( per_task_logs && NTASKS > n_task )) && n_task=$NTASKS
        # What n_task counts depends on where it came from. Calling logs "tasks"
        # would lend the figure an authority nothing gave it.
        unit="task(s)"
        (( NTASKS_KNOWN )) || unit=".out file(s)"
        # submitit writes the result pickle when a task returns, so one sitting
        # beside a step-less .out marks a task that has ended rather than one
        # still short of its first checkpoint. Counted once here for the two
        # branches below, which put the figure to different uses.
        if (( per_task_logs && ${#NOSTEP[@]} > 0 )); then
            for cand in "${NOSTEP[@]}"; do
                [[ -f "${cand%_log.out}_result.pkl" ]] && (( ++n_ended ))
            done
        fi
        if (( n_ended > 0 && n_seen > 0 )); then
            # Holding the job's progress at zero for an ended task would report
            # a slowest task that is no longer running. Dropping it is sound
            # only where a sibling .out HAS logged a step, the one thing that
            # shows this run prints step counts at all: with none anywhere, an
            # ended task and a run launched with experiment.progress_report !=
            # "dict" leave the same pair of files behind, and the pickle cannot
            # tell them apart. That case is handled where nothing was logged,
            # below.
            #
            # "task(s)" rather than $unit, which may be counting .out files: the
            # pickle is written per task, so this count is over tasks whatever
            # the totals beside it are counted in.
            ended_note="$n_ended task(s) ended before any checkpoint and are not counted"
            n_task=$(( n_task - n_ended ))
        fi
        if (( n_seen > 0 )); then
            local total_steps="" ts_out="" job_steps="$min_logged"
            # A task that has logged nothing is either short of its own first
            # checkpoint or dead. Checkpoints begin early in a run, so the
            # longer the job has been going the likelier the second is. Either
            # way the job is no further along than that task, whatever ground
            # the others have covered.
            (( n_seen < n_task )) && job_steps=0
            # Every task of a job runs the same configuration, so any .out that
            # got far enough to print the location answers for all of them.
            for cand in "${OUTS[@]}"; do
                ts_out=$(training_steps_from_out "$cand" "$WORKDIR") && break
                ts_out=""
            done
            IFS=$'\t' read -r total_steps ts_multi <<< "$ts_out"
            if [[ -n "$total_steps" ]] && (( total_steps > 0 )); then
                steps_cell="$job_steps / $total_steps ($(pct "$job_steps" "$total_steps")%)"
                steps_long="$steps_cell"
                # Steps remaining at the rate the job has averaged since it
                # started. Needs a step count to divide by, so a job whose
                # progress is zero -- including one held at zero because a task
                # has logged nothing -- has no rate and gets no estimate.
                if [[ -n "$el_s" ]] && (( el_s > 0 && job_steps > 0 )); then
                    local remaining=$(( total_steps - job_steps ))
                    (( remaining < 0 )) && remaining=0
                    eta_cell=$(s_to_compact $(( remaining * el_s / job_steps )))
                fi
            else
                steps_cell="$job_steps / ?"
                total_caveat="no \"Data will be saved at\" line in the .out, or no cfg.yaml where it points, so the total is unknown"
                steps_long="$steps_cell -- $total_caveat"
            fi
            if (( ! per_task_logs && (NTASKS > 1 || ! NTASKS_KNOWN) )); then
                # Every task writes to this one file, so the last step line in
                # it belongs to whichever flushed most recently. There is no
                # way to tell the others apart, and no way to find the slowest.
                # An unreported count means the same caveat: the 1 standing in
                # for it cannot be told from a job that really has one task.
                if (( NTASKS_KNOWN )); then
                    extra="one .out shared by $NTASKS task(s): this is whichever logged last, not the slowest"
                else
                    extra="one .out for the whole job and no task count reported: this is whichever task logged last, not necessarily the slowest"
                fi
            elif (( n_seen < n_task )); then
                extra="$(( n_task - n_seen )) of $n_task $unit with no checkpoint yet"
                if (( min_logged == max_logged )); then
                    extra+="; the other $n_seen at $min_logged"
                else
                    extra+="; the other $n_seen at $min_logged-$max_logged"
                fi
            elif (( max_logged > min_logged )); then
                extra="slowest of $n_task $unit, fastest at $max_logged"
            fi
            if [[ -n "$ended_note" ]]; then
                [[ -n "$extra" ]] && extra+="; "
                extra+="$ended_note"
            fi
            if (( per_task_logs && ! NTASKS_KNOWN )); then
                # n_task is the number of logs here, and a task that has not
                # written one leaves no trace in that count. Nothing reported
                # how many tasks the step has, so there is no way to notice --
                # the figure above is over the tasks that did write.
                [[ -n "$extra" ]] && extra+="; "
                extra+="task count not reported, so a task that has written no log at all is not counted"
            fi
            [[ -n "$extra" ]] && steps_long+="  [$extra]"
            if [[ "$ts_multi" == "1" ]]; then
                echo "[progress $JOBID] WARNING: cfg.yaml sets training_steps more than once, with different" >&2
                echo "                  values. The first is the total reported for this job; it may not be the" >&2
                echo "                  one this run obeys. Check the file before reading the percentage." >&2
            fi
        else
            steps_cell="no checkpoint"
            # Nothing here has logged a step, so nothing establishes that this
            # run logs them, and the two explanations below stand together. A
            # result pickle still proves that its task has ended, so those are
            # counted -- but not excluded, and not said to have ended before a
            # checkpoint.
            steps_long="nothing logged in $n_out .out file(s) -- the run has not reached its first checkpoint, or it was launched with experiment.progress_report != \"dict\""
            # No denominator: the count is over tasks, one pickle each, while
            # n_task may be a number of .out files, and this branch has no
            # clause qualifying that the way the stepped one does. The long
            # line states the file count immediately before this.
            if (( n_ended > 0 )); then
                extra="$n_ended task(s) have ended"
                steps_long+="  [$extra]"
            fi
        fi
    else
        steps_cell="no logs"
        steps_long="no .out to read (StdOut unresolved, or the files are gone)"
    fi

    local time_long=""
    if [[ -n "$el_s" ]]; then
        if [[ -n "$lim_s" ]]; then
            time_cell="$(s_to_compact "$el_s") / $(s_to_compact "$lim_s") ($(pct "$el_s" "$lim_s")%)"
            time_long="$time_cell"
        else
            time_cell="$(s_to_compact "$el_s") / $TLIMIT"
            time_caveat="not a duration, so no percentage"
            time_long="$time_cell ($time_caveat)"
        fi
    fi

    if (( PROGRESS_ONLY )); then
        # Handed to progress_table rather than printed: the columns cannot be
        # sized until every row is in. The long lines carry these two caveats
        # inline, where a cell has no room for them, so they join the rest in
        # the trailing bracket instead of being dropped.
        local row_extra="$extra" c
        for c in "$total_caveat" "$time_caveat"; do
            [[ -n "$c" ]] || continue
            [[ -n "$row_extra" ]] && row_extra+="; "
            row_extra+="$c"
        done
        progress_row "$JOBID" "$steps_cell" "$time_cell" "$eta_cell" "$row_extra"
        return $JOB_RC
    fi

    echo "[progress $JOBID] steps: $steps_long"
    [[ -n "$time_long" ]] && echo "[progress $JOBID] time:  $time_long"

    local notes=() req_kb peak_kb ave_kb total_kb mem_pct seen time_pct
    local best_kb="" best_src=""

    req_kb=$(mem_to_kb "$REQ") || req_kb=""
    peak_kb=$(mem_to_kb "$PEAK") || peak_kb=""
    ave_kb=$(mem_to_kb "$AVE") || ave_kb=""

    # Three estimates of what the job actually holds, best first. They disagree
    # on purpose: see the header. Whichever is available and tightest is what
    # --fit consumes.
    if [[ -n "$CG_KB" ]]; then
        # anon + file + kernel, NOT anon alone. --mem limits the cgroup's whole
        # charge, so that is what BASE_MEM_MB + PER_SEED_MEM_MB * n has to cover.
        # Fitting anon by itself under-requests by the size of `file`, which for
        # this stack is several hundred MB.
        best_kb=$(( CG_KB + CG_FILE_KB + CG_KERN_KB )); best_src="cgroup"
        if [[ -n "$req_kb" ]]; then
            notes+=("memory (cgroup anon, size from this): $(kb_to_mb "$CG_KB")M is $(pct "$CG_KB" "$req_kb")% of the $REQ requested -- unreclaimable memory, shared pages once, every process included")
        fi
        # The constants, read straight off this job. `file` and `kernel` do not
        # grow with the seed count, so they are the base; the seeds' own anon and
        # the per-task overhead do, so they are the per-seed term. Whatever anon
        # no process accounts for goes to the base, which is the safe direction.
        # Only when the seed count agrees with a REPORTED NTasks: see the
        # warning above. Against the 1 that stands in for a count nothing
        # gave, the comparison proves nothing, and a job whose classifier
        # found one process would pass it without being checked at all.
        if (( CG_SEED_N > 0 && NTASKS_KNOWN && CG_SEED_N == NTASKS )); then
            local base_kb=$(( CG_FILE_KB + CG_KERN_KB + CG_KB - CG_SEED_KB - CG_OVH_KB ))
            (( base_kb < CG_FILE_KB + CG_KERN_KB )) && base_kb=$(( CG_FILE_KB + CG_KERN_KB ))
            notes+=("model: this job implies BASE_MEM_MB ~= $(kb_to_mb "$base_kb") and PER_SEED_MEM_MB ~= $(kb_to_mb $(( (CG_SEED_KB + CG_OVH_KB) / CG_SEED_N ))) -- one sample, from ONE seed count. Fit over several with --fit, and add 10-20%.")
        elif (( CG_SEED_N > 0 && NTASKS_KNOWN )); then
            notes+=("model: not stated -- $CG_SEED_N seed process(es) classified against $NTASKS task(s) from sacct. Fix the mismatch before sizing anything from this job.")
        elif (( CG_SEED_N > 0 )); then
            notes+=("model: not stated -- $CG_SEED_N seed process(es) classified, and no task count was reported to check them against. Poll the job live with sstat, or read NTasks off a COMPLETED run, before sizing anything from this job.")
        else
            notes+=("model: not stated -- no process in the cgroup matched the entry point (main.py), so there are no seed processes to divide the anon between and every one of them landed in overhead. Check what the job actually runs before sizing anything from it.")
        fi
        # memory.peak sitting at the limit says nothing on its own: page cache
        # expands to fill whatever --mem leaves free. hit_max is what separates
        # "cache filled up harmlessly" from "the kernel had to fight for room".
        if (( CG_OOM > 0 )); then
            notes+=("memory: cgroup reports $CG_OOM oom_kill event(s) -- this job was killed for memory, raise --mem")
        elif (( CG_HITMAX > 0 )); then
            notes+=("memory: cgroup hit its limit $CG_HITMAX time(s) and had to reclaim to stay under it -- --mem is too low even if nothing was killed yet")
        elif [[ -n "$req_kb" && -n "$CG_PEAK_KB" ]] && (( CG_PEAK_KB * 100 >= req_kb * 95 )); then
            notes+=("memory: memory.peak is at $(pct "$CG_PEAK_KB" "$req_kb")% of $REQ but hit_max is 0 -- that is page cache filling the headroom, which is normal and harmless. Ignore it and read the anon line.")
        fi
    fi

    # Second choice after the cgroup, and the only one that survives the job:
    # summed over the step's tasks rather than the largest of them, and
    # cgroup-derived when JobAcctGatherType is jobacct_gather/cgroup (check with
    # `scontrol show config | grep -i JobAcctGather`). It is sampled every
    # JobAcctGatherFrequency seconds, so a spike between samples is still
    # missed, and it is RSS -- no anon/file split, no hit_max.
    if [[ -z "$best_kb" && "${TRESTOT_KB:-0}" =~ ^[0-9]+$ ]] && (( TRESTOT_KB > 0 )); then
        best_kb="$TRESTOT_KB"; best_src="tres_tot"
        if [[ -n "$req_kb" ]]; then
            notes+=("memory (TRESUsageInTot, summed over tasks + batch): $(kb_to_mb "$TRESTOT_KB")M is $(pct "$TRESTOT_KB" "$req_kb")% of the $REQ requested -- post-mortem, and sampled every JobAcctGatherFrequency seconds, so a spike between samples is missed")
        fi
    fi

    local batch_kb
    batch_kb=$(mem_to_kb "$BATCH") || batch_kb=0

    if [[ -n "$peak_kb" ]]; then
        # + .batch: one process, counted once, but charged to the same --mem.
        total_kb=$(( peak_kb * NTASKS + batch_kb ))
        [[ -z "$best_kb" ]] && { best_kb="$total_kb"; best_src="maxrss_x_tasks"; }
        seen="peak_rss $PEAK"
        [[ "$PEAK_SRC" == "sstat" ]] && seen="peak so far $PEAK"
        if [[ -n "$req_kb" ]]; then
            mem_pct=$(pct "$total_kb" "$req_kb")
            if (( ! NTASKS_KNOWN )); then
                notes+=("memory (MaxRSS, ONE TASK): $seen, and no task count was reported to scale it by. This is one task's peak, not the job's, so it cannot be read against the $REQ requested -- multiply it by the number of tasks for that. Poll live with sstat, or read NTasks off a COMPLETED run.")
            elif (( NTASKS > 1 )); then
                notes+=("memory (MaxRSS x tasks + batch, UPPER BOUND): $seen x $NTASKS + $BATCH ~= $(kb_to_mb "$total_kb")M, ${mem_pct}% of $REQ -- assumes all tasks peaked at once and counts library pages $NTASKS times")
            else
                notes+=("memory (MaxRSS): $seen is ${mem_pct}% of the $REQ requested")
            fi
        fi
    elif [[ "$PEAK" == "?" && -z "$CG_KB" ]]; then
        notes+=("memory: no peak available -- sacct fills MaxRSS only when a step ends, and sstat (live) returned nothing. Check directly with: sstat -a -j $JOBID --format=JobID,MaxRSS,AveRSS,MaxVMSize")
    fi

    if [[ -n "$ave_kb" ]] && (( NTASKS_KNOWN && NTASKS > 1 )) && [[ -n "$req_kb" ]]; then
        total_kb=$(( ave_kb * NTASKS + batch_kb ))
        notes+=("memory (AveRSS x tasks + batch): $AVE x $NTASKS + $BATCH ~= $(kb_to_mb "$total_kb")M, $(pct "$total_kb" "$req_kb")% of $REQ -- mean over tasks rather than max, still counts library pages $NTASKS times")
    fi

    if [[ -z "$CG_KB" && "$ST" == RUNNING* ]] && (( USE_CGROUP )); then
        notes+=("memory: cgroup unavailable (no overlap step permitted, or memory.stat not exposed). Everything above is an over-estimate.")
    elif [[ -z "$CG_KB" && "$ST" != RUNNING* ]]; then
        notes+=("memory: the cgroup can only be read while the job RUNS -- it is destroyed when the job ends. Poll the next one live (see the sizing recipe in submit_jobs.py).")
    fi

    if [[ -n "$el_s" && -n "$lim_s" ]]; then
        time_pct=$(pct "$el_s" "$lim_s")
        if (( time_pct >= 80 )); then
            notes+=("time: ${time_pct}% of the $TLIMIT wallclock is gone. submitit is signalled (SIGUSR2) shortly before the limit and the run dies without a traceback. Raise this env's timeout.")
        fi
    fi

    case "$ST" in
        PENDING*)       notes+=("state: still queued -- nothing has run yet, so there is no usage to read") ;;
        RUNNING*)       notes+=("state: running -- the numbers above are partial. Memory that still grows (a replay buffer filling) has not peaked yet; do not size from this.") ;;
        COMPLETED*)     notes+=("state: completed cleanly (exit $EC) -- these numbers are final and safe to size from") ;;
        OUT_OF_MEMORY*) notes+=("state: OOM-killed -- raise the memory per seed, or run fewer seeds per chunk") ;;
        TIMEOUT*)       notes+=("state: killed at the wallclock -- raise this env's timeout") ;;
        NODE_FAIL*)     notes+=("state: node failure, nothing to do with the code -- resubmit") ;;
        BOOT_FAIL*)     notes+=("state: node failed to boot -- resubmit") ;;
        PREEMPTED*)     notes+=("state: preempted by a higher-priority job -- resubmit") ;;
        DEADLINE*)      notes+=("state: hit the partition deadline ($ST)") ;;
        CANCELLED*)
            # CANCELLED is ambiguous: a real scancel and a wallclock kill both
            # land here. If elapsed reached the limit, call it what it is.
            if [[ -n "$el_s" && -n "$lim_s" ]] && (( el_s >= lim_s )); then
                notes+=("state: cancelled at the time limit ($ELAPSED of $TLIMIT) -- a wallclock kill, not a scancel; raise this env's timeout")
            else
                notes+=("state: cancelled early ($ST) -- scancel, by you or by the scheduler")
            fi ;;
        FAILED*)        notes+=("state: failed with exit $EC -- read the stderr above; if it is empty the job died before writing anything, which usually means the batch prologue (the module/venv setup lines of the submission script)") ;;
        *)              notes+=("state: $ST (exit $EC)") ;;
    esac

    echo
    echo "[analysis]"
    printf '  %s\n' "${notes[@]}"

    # Feed the fit. Running jobs are still collected but flagged, because a
    # half-filled replay buffer is the single easiest way to fit a slope that is
    # too shallow and then OOM the production sweep.
    if (( DO_FIT )) && [[ -z "$best_kb" ]]; then
        # Dropping it quietly would leave --fit reporting a clean fit over fewer
        # jobs than were asked for, with nothing to say which one went missing.
        FIT_DROPPED+=("$JOBID")
    elif (( DO_FIT )) && (( ! NTASKS_KNOWN )); then
        # The fit is over task count, so a job with no count has no x. The 1
        # standing in for it is not a measurement, and fitting it would place
        # the job at a task count it may never have had.
        FIT_NO_TASKS+=("$JOBID")
    elif (( DO_FIT )); then
        FIT_TASKS+=("$NTASKS")
        FIT_KB+=("$best_kb")
        FIT_SRC+=("$best_src")
        FIT_STATE+=("$ST")
        FIT_JOB+=("$JOBID")
        FIT_NAME+=("${NAME:-?}")
    fi

    return $JOB_RC
}

# ---------------------------------------------------------------------------
# The --progress_only table, printed once every job has been inspected.
#
# Every column but the last is measured over the rows rather than fixed. Ids
# vary in length within one sweep, and a steps cell grows with the digits of
# training_steps -- a finished run prints the widest cell of all. Two literal
# spaces separate the columns on top of the padding, so a cell that outgrows
# its column pushes its own row out instead of touching the next one.
#
# Caveats trail the row rather than crowd a cell, so a job with nothing to
# qualify prints the four columns and stops.
# ---------------------------------------------------------------------------
progress_table() {
    local n=${#ROW_JOB[@]} i job_w=3 steps_w=5 time_w=4
    (( n > 0 )) || return 0

    for (( i = 0; i < n; i++ )); do
        (( ${#ROW_JOB[i]}   > job_w   )) && job_w=${#ROW_JOB[i]}
        (( ${#ROW_STEPS[i]} > steps_w )) && steps_w=${#ROW_STEPS[i]}
        (( ${#ROW_TIME[i]}  > time_w  )) && time_w=${#ROW_TIME[i]}
    done

    printf '%-*s  %-*s  %-*s  %s\n' "$job_w" "Job" "$steps_w" "Steps" "$time_w" "Time" "Left"
    for (( i = 0; i < n; i++ )); do
        printf '%-*s  %-*s  %-*s  %s%s\n' "$job_w" "${ROW_JOB[i]}" "$steps_w" "${ROW_STEPS[i]}" "$time_w" "${ROW_TIME[i]}" "${ROW_LEFT[i]}" "${ROW_EXTRA[i]:+  [${ROW_EXTRA[i]}]}"
    done
}

# ---------------------------------------------------------------------------
# Least-squares fit of total = BASE + PER_SEED * tasks over the collected jobs.
# ---------------------------------------------------------------------------
fit_memory_model() {
    local n=${#FIT_TASKS[@]}
    echo
    echo "########################################################################"
    echo "# Memory model fit"
    echo "########################################################################"

    if (( ${#FIT_DROPPED[@]} > 0 )); then
        echo "WARNING: ${#FIT_DROPPED[@]} job(s) contributed nothing and are NOT in the fit below:" >&2
        printf '           %s\n' "${FIT_DROPPED[*]}" >&2
        echo "         No memory figure was available for them -- no cgroup cache (never polled" >&2
        echo "         while running), no TRESUsageInTot and no MaxRSS. A job that never started," >&2
        echo "         or one still pending, looks exactly like this. Check with sacct." >&2
        echo >&2
    fi

    if (( ${#FIT_NO_TASKS[@]} > 0 )); then
        echo "WARNING: ${#FIT_NO_TASKS[@]} job(s) have a memory figure but no task count, and are NOT" >&2
        echo "         in the fit below:" >&2
        printf '           %s\n' "${FIT_NO_TASKS[*]}" >&2
        echo "         Neither sacct nor sstat reported NTasks for them. Task count is the axis" >&2
        echo "         being fitted, so there is nothing to place these jobs on it: the 1 the rest" >&2
        echo "         of this script stands in with would pull BASE up and the slope down. Poll" >&2
        echo "         them live with sstat while they run, or read NTasks off a COMPLETED run." >&2
        echo >&2
    fi

    if (( n < 2 )); then
        echo "Need at least 2 usable jobs; got $n." >&2
        return 1
    fi

    local i distinct
    printf '  %-12s %-24s %6s %12s %s\n' "JOBID" "NAME" "TASKS" "TOTAL(M)" "SOURCE"
    for (( i = 0; i < n; i++ )); do
        printf '  %-12s %-24s %6s %12s %s\n' "${FIT_JOB[i]}" "${FIT_NAME[i]:0:24}" "${FIT_TASKS[i]}" "$(kb_to_mb "${FIT_KB[i]}")" "${FIT_SRC[i]} (${FIT_STATE[i]})"
    done
    echo

    # The job name is "<run_id>_<seeds>". Fitting across different run_ids fits
    # a slope through two different configurations, which is exactly how you get
    # four scattered points and no explanation.
    local runids
    runids=$(printf '%s\n' "${FIT_NAME[@]}" | sed 's/_[^_]*$//' | sort -u)
    if (( $(printf '%s\n' "$runids" | wc -l) > 1 )); then
        echo "WARNING: these jobs are NOT the same configuration. run_ids present:" >&2
        printf '           %s\n' $runids >&2
        echo "         The slope is meaningless across configs. Refit within one run_id." >&2
        echo >&2
    fi

    distinct=$(printf '%s\n' "${FIT_TASKS[@]}" | sort -u | wc -l)
    if (( distinct < 2 )); then
        echo "All $n jobs have the same task count -- the slope is unidentifiable." >&2
        echo "Resubmit the same config at a different --seeds_per_chunk." >&2
        return 1
    fi

    # Mixed sources make the fit meaningless: a cgroup anon figure and a
    # MaxRSS x tasks figure are not the same quantity, and the difference grows
    # with tasks, which is exactly the axis being fitted.
    if (( $(printf '%s\n' "${FIT_SRC[@]}" | sort -u | wc -l) > 1 )); then
        echo "WARNING: the totals above come from different sources. cgroup anon and" >&2
        echo "         MaxRSS x tasks are not comparable, and the gap widens with the" >&2
        echo "         task count -- the slope will absorb it. Refit on one source." >&2
        echo >&2
    fi
    if printf '%s\n' "${FIT_STATE[@]}" | grep -q '^RUNNING'; then
        echo "WARNING: some jobs are still RUNNING. Their peak has not settled, so the" >&2
        echo "         slope is a lower bound. Refit once they COMPLETE." >&2
        echo >&2
    fi

    paste <(printf '%s\n' "${FIT_TASKS[@]}") <(printf '%s\n' "${FIT_KB[@]}") | awk '
            { x = $1 + 0; y = ($2 + 0) / 1024; n++; sx += x; sy += y; sxx += x*x; sxy += x*y
              px[n] = x; py[n] = y }
            END {
                den = n*sxx - sx*sx
                slope = (n*sxy - sx*sy) / den
                icept = (sy - slope*sx) / n

                # R^2, as a check that the two-constant model is the right shape
                ybar = sy / n
                for (i = 1; i <= n; i++) {
                    f = icept + slope*px[i]
                    ssr += (py[i] - f)^2
                    sst += (py[i] - ybar)^2
                }
                r2 = (sst > 0 ? 1 - ssr/sst : 1)

                printf "  fitted   PER_SEED = %.0f M/seed\n", slope
                printf "  fitted   BASE     = %.0f M\n", icept
                printf "  R^2               = %.4f\n\n", r2

                # Two points define a line exactly, so R^2 is 1 whatever the
                # data says. Printing it next to a two-point fit reads as
                # confirmation when nothing has been confirmed.
                if (n < 3)
                    printf "  NOTE: only %d points -- a line through 2 points always fits perfectly, so\n        the R^2 above means nothing. Add a third task count to test linearity.\n\n", n

                if (icept < 0)
                    printf "  NOTE: negative intercept -- the shared component is below the noise.\n        Clamp BASE to ~300M and treat the slope as the whole story.\n\n"
                if (r2 < 0.95 && n > 2)
                    printf "  NOTE: R^2 is low; the relationship is not linear. Something you assumed\n        was shared is being allocated per task. Do not extrapolate this fit.\n\n"

                # 1.3x covers seed-to-seed variance: the tasks share one cgroup, so
                # one runaway takes the whole chunk with it. No batch-step term
                # here -- both total sources already include .batch (the cgroup
                # natively, the RSS path via batch_kb), so it is in the intercept.
                base = icept * 1.3
                if (base < 300) base = 300
                printf "  Paste into submit_jobs.py:\n"
                printf "    BASE_MEM_MB = %d\n", int(base + 0.5)
                printf "    PER_SEED_MEM_MB[env] = %d\n", int(slope * 1.3 + 0.5)
            }'
}

# ---------------------------------------------------------------------------
# Dispatch: arbitrary list of job ids (with id1..id2 range expansion),
# or all running jobs for $USER when no args are given.
# ---------------------------------------------------------------------------
JOB_IDS=()
ARGS=()
while (( $# )); do
    case "$1" in
        --fit)          DO_FIT=1; shift ;;
        # Nothing but the .out files and sacct's clock is read, so the overlap
        # step that samples the cgroup would be paid for output never printed.
        --progress_only) PROGRESS_ONLY=1; USE_CGROUP=0; shift ;;
        --no-cgroup)    USE_CGROUP=0; shift ;;
        -s|--slurm-out) SLURM_OUT="${2:?--slurm-out needs a name}"; shift 2 ;;
        --slurm-out=*)  SLURM_OUT="${1#*=}"; shift ;;
        -h|--help)      sed -n '2,/^set -u/p' "$0" | sed 's/^# \{0,1\}//;$d'; exit 0 ;;
        *)              ARGS+=("$1"); shift ;;
    esac
done

if (( PROGRESS_ONLY && DO_FIT )); then
    echo "--progress_only and --fit are incompatible: the fit is over memory totals," >&2
    echo "which --progress_only does not collect. Run them as two commands." >&2
    exit 2
fi

if (( ${#ARGS[@]} >= 1 )); then
    for tok in "${ARGS[@]}"; do
        if [[ "$tok" =~ ^([0-9]+)\.\.([0-9]+)$ ]]; then
            a="${BASH_REMATCH[1]}"
            b="${BASH_REMATCH[2]}"
            if (( a > b )); then t=$a; a=$b; b=$t; fi
            for (( i=a; i<=b; i++ )); do JOB_IDS+=("$i"); done
        else
            JOB_IDS+=("$tok")
        fi
    done
else
    # squeue: only this user, only RUNNING, just the job id, no header
    mapfile -t JOB_IDS < <(squeue -h -u "${USER:-$(id -un)}" -t RUNNING -o '%i' 2>/dev/null)
    if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
        echo "No running jobs for user ${USER:-$(id -un)}." >&2
        exit 0
    fi
    if (( ! PROGRESS_ONLY )); then
        echo "Found ${#JOB_IDS[@]} running job(s) for ${USER:-$(id -un)}: ${JOB_IDS[*]}"
        echo
    fi
fi

RC=0
for jid in "${JOB_IDS[@]}"; do
    if (( ! PROGRESS_ONLY )) && [[ ${#JOB_IDS[@]} -gt 1 ]]; then
        echo "########################################################################"
        echo "# Job $jid"
        echo "########################################################################"
    fi
    inspect_job "$jid" || RC=$?
    (( PROGRESS_ONLY )) || echo
done

(( PROGRESS_ONLY )) && progress_table
(( DO_FIT )) && { fit_memory_model || RC=$?; }

exit $RC
