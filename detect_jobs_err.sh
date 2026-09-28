#!/usr/bin/env bash
# Scan slurm *.err logs in a folder and summarize the ones that failed, as one
# section per file:
#
#   JOB: 842523_0_log          (job 842523, task 0)
#   NAME: a1b2c3d4_0-4         (job name: <run_id>_<seeds>)
#   FILE: .../842523_0_log.err
#   [diagnosis]   why it died, when sacct or the log says so
#   [note]        caveats: clean task in a failed job, log/sacct disagreement
#   [command]     the exact command, to copy-paste and reproduce
#   [progress]    speed and time-left history, read back from the sibling .out
#   [sacct]       requested vs allocated vs peak memory, elapsed vs limit
#   [.err]        the log itself
#
# Jobs that never wrote a log get their own section at the top: submitit leaves a
# marker (_submission.sh / _submitted.pkl) at submission time, so a marker with no
# *_log.out/.err is a job that died or is queued, and the per-file scan below --
# which iterates over .err files -- cannot see it.
#
# One job runs a whole seed chunk as parallel srun TASKS -- one plain
# `python main.py` per seed, each writing its own log
# (<jobid>_<element>_<task>_log.err). A seed's traceback therefore does NOT
# appear in its siblings' logs: the same bug shows up once per task file, not
# interleaved. Which seed a file belongs to is in its [command] line
# (experiment.rng_seed=N).
#
# WHAT COUNTS AS A FAILURE:
#
#   Two independent sources, and a file is reported if EITHER fires:
#
#     - sacct's state for the job (FAILED, TIMEOUT, OUT_OF_MEMORY, ...). This is
#       the authority. A task killed hard enough to write nothing still lands here.
#     - error patterns in the log itself, for the cases sacct cannot see (a job
#       that exits 0 after swallowing a traceback, or no sacct at all).
#
#   The log side is deliberately conservative. Every line is checked against a
#   BENIGN list first, and anything matching it can never raise an error --
#   wandb's own `wandb: WARNING ...` banner, Python's UserWarning/
#   DeprecationWarning/FutureWarning family, Lmod's module-reload chatter, the
#   `[task N]` command banner. A log containing nothing but warnings is NOT a
#   failure and is not reported; the run counter at the end says how many such
#   files were skipped.
#
#   Corollary: matching is CASE-SENSITIVE. "Error"/"ERROR" mean something,
#   "error" inside prose does not, and a case-insensitive scan is what turns
#   every warning banner into a false positive.
#
# MEMORY, and why sacct always understates it:
#
#   MaxRSS is the largest SINGLE TASK of the step, sampled every
#   JobAcctGatherFrequency seconds. Against a job that was OOM-killed it is
#   wrong in three directions at once:
#
#     - it misses the sibling tasks. All tasks of a job share ONE cgroup and
#       one --mem pool, so what the OOM killer weighs is their SUM, not the
#       biggest one.
#     - it misses everything that is not the srun step. The submitit parent
#       (~25 MB per task) and the Apptainer squashfuse_ll mounts (~18 MB per
#       task) are charged to the same pool and appear in no MaxRSS.
#     - it misses file-backed memory entirely. Mapped libtorch text plus page
#       cache runs ~1 GB for this stack, is charged ONCE per job however many
#       seeds run, and is exactly what BASE_MEM_MB exists to cover.
#
#   Together that is why an OOM routinely shows a MaxRSS at half of ReqMem or
#   less, which reads as "this job used no memory" and is not true. The numbers
#   are reported as-is and NOT extrapolated into a verdict: MaxRSS x n_tasks is
#   not the job's footprint (it double-counts nothing and misses three things),
#   so it cannot tell a job that was starved of memory from one that was simply
#   slower than its wallclock. The full split (anon / file / kernel, seeds vs
#   per-task overhead) comes from inspect_jobs.sh, which can only read it while
#   the job RUNS -- size from that, and use these logs to confirm what died.

set -uo pipefail

# Where submitit writes the logs. It is SLURM_OUT in submit_jobs.py, which
# builds the same path as $SCRATCH/<name>: keep the two in step, or point this
# at the folder directly with --folder.
slurm_out="slurm_out"
folder=""          # full path; when set it overrides $SCRATCH/$slurm_out
out=""             # report path; default <parent of folder>/jobs_err.txt
log_full_lines=400 # a log shorter than this is printed whole
log_head_lines=80  # lines from the top of a log too long to print whole
log_tail_lines=80  # ... and from the bottom, with the gap between them marked
ctx_before=15      # context around each matched line in [matched patterns]
ctx_after=25
include_ok=0       # also report jobs sacct calls COMPLETED but whose log has errors
within_min=""      # only look at logs touched in the last N minutes (empty: all of them)
cutoff=0           # the same thing as an epoch second; 0 disables the filter

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--folder)     folder="${2:?--folder needs a directory}";  shift 2 ;;
        -s|--slurm-out)  slurm_out="${2:?--slurm-out needs a name}"; shift 2 ;;
        -o|--out)        out="${2:?--out needs a file path}";        shift 2 ;;
        -m|--minutes)    within_min="${2:?--minutes needs a number}";  shift 2 ;;
        --include-completed) include_ok=1; shift ;;
        -h|--help)
            echo "Usage: $0 [-f|--folder DIR] [-s|--slurm-out NAME] [-o|--out FILE]"
            echo "          [-m|--minutes N] [--include-completed]"
            echo "  -f  slurm .err folder as a full path (default: \$SCRATCH/\$slurm_out)"
            echo "  -s  name of that folder under \$SCRATCH (default: slurm_out)"
            echo "  -o  where to write the report (default: <parent of folder>/jobs_err.txt)"
            echo "  -m  only scan logs written in the last N minutes (default: all of them)."
            echo "      The test is the file's mtime, i.e. when the task last wrote a line:"
            echo "      for a dead task that is when it died, for a live one it is now, so a"
            echo "      long run started before the window still counts as recent."
            echo "  --include-completed  also report jobs sacct calls COMPLETED whose log"
            echo "                       matched an error pattern (off by default: sacct wins)"
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

folder="${folder:-${SCRATCH:-.}/${slurm_out}}"
folder="${folder%/}"

if [[ ! -d "$folder" ]]; then
    echo "Folder not found: $folder" >&2
    exit 1
fi

out="${out:-$(dirname "$folder")/jobs_err.txt}"
if ! ( : > "$out" ) 2>/dev/null; then
    echo "Cannot write report to $out -- pass a writable path with --out" >&2
    exit 1
fi
# Absolute, so the report can be re-read and so the scan below can skip it if it
# happens to live inside the scanned folder.
[[ "$out" = /* ]] || out="$(cd "$(dirname "$out")" && pwd)/$(basename "$out")"

# --- the --minutes window ---------------------------------------------------
#
# The filter is on the file's MTIME, not on sacct: a per-file stat costs nothing
# and can prune thousands of logs before any of the expensive work (the awk
# passes, one sacct query per job) starts, which is the point of asking for a
# window in the first place.
#
# What that timestamp means depends on the task. A task that is dead stopped
# writing when it died, so its mtime is its time of death and "the last 30
# minutes" reads exactly as expected. A task still RUNNING keeps touching its
# log, so it stays in the window however long ago it was submitted -- also what
# you want, since a live job is the one you are most likely watching.
#
# The exception is a job that never wrote a log at all: it has no mtime to test,
# so the NO LOGS scan below filters on the submitit MARKER instead, which is
# stamped at submission time. For those entries the window means "submitted in
# the last N minutes", and a job queued for longer than the window drops out of
# the report even though it is still pending.
if [[ -n "$within_min" ]]; then
    if ! [[ "$within_min" =~ ^[0-9]+$ ]] || (( within_min == 0 )); then
        echo "--minutes needs a positive whole number of minutes, got: $within_min" >&2
        exit 1
    fi
    cutoff=$(( $(date +%s) - within_min * 60 ))
fi

# GNU stat first, BSD/macOS second, so this keeps working off the cluster.
file_mtime() {
    stat -c %Y -- "$1" 2>/dev/null || stat -f %m -- "$1" 2>/dev/null
}

# True when the file is inside the window. A file whose mtime cannot be read is
# kept: dropping a log because stat hiccuped would hide a failure silently,
# which is the one outcome this script exists to prevent.
recent_enough() {
    local t
    (( cutoff == 0 )) && return 0
    t=$(file_mtime "$1") || return 0
    [[ "$t" =~ ^[0-9]+$ ]] || return 0
    (( t >= cutoff ))
}

if (( cutoff > 0 )); then
    cutoff_str=$(date -d "@$cutoff" 2>/dev/null || date -r "$cutoff" 2>/dev/null || echo "epoch $cutoff")
    echo "Looking for logs in $folder (last $within_min min, i.e. since $cutoff_str)" >&2
    {
        echo "FILTER: only logs written in the last $within_min minute(s), i.e. since $cutoff_str."
        echo "        Older files are not scanned and are not counted below."
        echo ""
    } >> "$out"
else
    echo "Looking for logs in $folder" >&2
fi

# nullglob for the whole script rather than toggled per use: toggling clobbers
# the caller's setting when this file is sourced, and an unmatched glob leaking
# through as a literal path is the worse failure mode here.
shopt -s nullglob

# ---------------------------------------------------------------------------
# What a failure looks like in a log
# ---------------------------------------------------------------------------
#
# BENIGN is checked first and wins. Anything here is noise by construction: it
# is what a healthy run of this stack prints on stderr. Add to this list, not
# to the error list, when a clean job shows up in the report.
benign='^\[task [0-9]+\] '
benign+='|^wandb:'
benign+='|^submitit (INFO|WARNING) '
benign+='|(User|Deprecation|Future|Pending|Pending[A-Za-z]*|Runtime|Resource|Import|Unicode|Bytes|Encoding|Syntax)Warning:'
benign+='|^ *warnings\.warn'
benign+='|^ *from pkg_resources import'
benign+='|^(Inactive|Currently Loaded|Loading) Modules'
benign+='|^The following (have|has) been reloaded'
benign+='|^Due to MODULEPATH changes'
benign+='|^ *[0-9]+\) [A-Za-z0-9_+.-]+/'
benign+='|^Lmod (has|is|Warning)'
benign+='|^-{10,}$'
benign+='|^https?://'
benign+='|^ *[A-Za-z]+Warning: '

# ERR is what actually kills a run. Case-sensitive, and anchored wherever the
# shape of the line allows it. The old catch-all `^<word>: <Capital>` is gone:
# it matched `wandb: WARNING ...` and flagged every healthy job in the folder.
# Nothing is lost by dropping it -- any Python exception that terminates a
# process prints the Traceback header, which is the first rule below.
err='Traceback \(most recent call last\):'
err+='|^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception|Interrupt)(: |$)'
err+='|^[A-Za-z_][A-Za-z0-9_.]*(SystemExit|StopIteration)(: |$)'
err+='|^ *raise [A-Z][A-Za-z0-9_.]*'
err+='|Error executing job'
err+='|HYDRA_FULL_ERROR'
err+='|Could not (find|load|override|append) '
err+='|In .config.* validation error'
err+='|(MissingConfig|ConfigComposition|ConfigAttribute|Instantiation)Exception'
err+='|submitit ERROR'
err+='|CUDA error|CUDA out of memory|cuDNN error|NCCL error'
err+='|oom[-_]kill|Out Of Memory Killed|OUT_OF_MEMORY|Killed process'
err+='|slurmstepd: error|srun: error|sbatch: error|CANCELLED AT .* DUE TO'
err+='|core dumped|Segmentation fault|Bus error|Aborted$'
err+='|^fatal: |FATAL ERROR|Fatal Python error'

# One pass over a log: how many real error lines, how many benign ones, whether
# SLURM's wallclock signal or an OOM message is in there, and the [task N]
# command banner. Folding these into one awk instead of four greps matters when
# the folder holds thousands of multi-MB logs.
#
# The pattern strings go through the environment, not -v: awk expands escape
# sequences in a -v assignment, which mangles the \( in the Traceback pattern.
US=$'\x1f'
scan_log() {
    ERRPAT="$err" BENIGNPAT="$benign" awk -v SEP="$US" '
        BEGIN { e = ENVIRON["ERRPAT"]; b = ENVIRON["BENIGNPAT"] }
        {
            if (cmd == "" && $0 ~ /^\[task [0-9]+\] /) cmd = substr($0, index($0, "] ") + 2)
            if ($0 ~ b) { warn++; next }
            if ($0 ~ e) err++
            if ($0 ~ /User defined signal 2|DUE TO TIME LIMIT/) sig2 = 1
            l = tolower($0)
            if (l ~ /oom[-_]kill|out of memory|out_of_memory|killed process/) oom = 1
        }
        END { printf "%d%s%d%s%d%s%d%s%s\n",
                     err+0, SEP, warn+0, SEP, sig2+0, SEP, oom+0, SEP, cmd }
    ' "$1"
}

# Matched lines with context, for logs too long to print whole. Two passes over
# the file rather than buffering it: these logs can be enormous.
show_context() {
    ERRPAT="$err" BENIGNPAT="$benign" awk -v B="$ctx_before" -v A="$ctx_after" '
        BEGIN { e = ENVIRON["ERRPAT"]; b = ENVIRON["BENIGNPAT"] }
        FNR == NR {
            if ($0 ~ b) next
            if ($0 ~ e) { lo = FNR - B; if (lo < 1) lo = 1
                          for (i = lo; i <= FNR + A; i++) want[i] = 1 }
            next
        }
        want[FNR] {
            if (last && FNR > last + 1) print "        ..."
            printf "%6d: %s\n", FNR, $0
            last = FNR
        }
    ' "$1" "$1"
}

# Reconstruct a dead run's speed history from its stdout.
#
# experiment.py with progress_report="dict" prints one `{...}` line per
# checkpoint to log.out, carrying
#   - step/sec (rate over the last interval): a decreasing rate may be a sign of
#     underlying training problems;
#   - eta_sec (projected time to finish at that rate, tests included): to
#     recalibrate timeout.
progress_from_out() {
    local out_file="$1"
    [[ -s "$out_file" ]] || return 1
    awk '
        # The dict lines are Python reprs, so every key is wrapped in single
        # quotes. Building that quote from its code point keeps this whole
        # program inside one shell-quoted string.
        BEGIN { q = sprintf("%c", 39) }

        function num(line, key,   s, pat) {
            pat = q key q ": *(-?[0-9]+\\.?[0-9]*([eE][-+]?[0-9]+)?|-?inf|nan)"
            if (!match(line, pat)) return "NA"
            s = substr(line, RSTART, RLENGTH)
            sub("^" q "[^" q "]*" q ": *", "", s)
            return s
        }
        function bad(x) { return (x == "NA" || x ~ /inf|nan/) }

        # What a projection column MEANS, ignoring how it is spelled. eta_sec
        # and eta_s are the same quantity under two names; left_s excluded the
        # remaining test() calls and is a different one.
        function kind(k) { return (k ~ /^left/) ? "left" : "eta" }

        # First key in the alias list that is present, so one parser reads every
        # generation of the log format. Sets PICKED to whichever it used --
        # the projection column has to be labelled with the name it came from,
        # since the alternatives do not mean the same thing.
        function pick(line, keys,   i, a, n, v) {
            n = split(keys, a, " ")
            for (i = 1; i <= n; i++) {
                v = num(line, a[i])
                if (!bad(v)) { PICKED = a[i]; return v }
            }
            PICKED = a[1]
            return "NA"
        }
        function hms(x,   h, m, s) {
            if (bad(x)) return "" x
            x = int(x + 0.5); h = int(x/3600); m = int((x%3600)/60); s = x%60
            return sprintf("%d:%02d:%02d", h, m, s)
        }
        function med(arr, lo, hi,   i, j, k, n, t) {
            n = 0
            for (i = lo; i <= hi; i++) { n++; t[n] = arr[i] }
            if (n == 0) return 0
            for (i = 2; i <= n; i++) { k = t[i]; j = i-1
                while (j > 0 && t[j] > k) { t[j+1] = t[j]; j-- }
                t[j+1] = k }
            return (n % 2) ? t[(n+1)/2] : (t[n/2] + t[n/2+1]) / 2
        }

        index($0, q "step/sec" q) > 0 || index($0, q "step/s" q) > 0 {
            sp = pick($0, "step/sec step/s")
            if (bad(sp)) next
            n++
            step[n] = pick($0, "steps");            spd[n] = sp + 0
            tst[n]  = pick($0, "test_sec test_s")
            trn[n]  = pick($0, "train_sec train_s")

            # eta is the projection to compare against the wallclock. left_s is
            # a pre-rename log that carried no eta column at all; it omits the
            # remaining test() calls and reads low, so record which key each
            # sample came from and never mix the two inside one trend.
            e = pick($0, "eta_sec eta_s left_s"); src[n] = PICKED
            # A rename is not a change of meaning: eta_s -> eta_sec is the same
            # quantity and its trend is still readable end to end. left_s is a
            # different quantity, and only that difference suppresses the trend.
            if (n > 1 && kind(src[n]) != kind(src[n-1])) mixed = 1
            eta[n] = e
        }

        END {
            if (n == 0) { print "NONE"; exit }

            key = src[n]
            printf "last checkpoint: step %s, %.1f step/s (%d checkpoint(s) logged)\n",
                   step[n], spd[n], n
            if (!bad(eta[n])) {
                printf "the run itself said %s left (%s)", hms(eta[n]), key
                if (key ~ /^left/) printf " -- training only, the remaining test() calls are NOT in that figure"
                printf "\n"
            }

            third = int(n/3); if (third < 1) third = 1
            early = med(spd, 1, third)
            late  = med(spd, n-third+1, n)
            ratio = (late > 0) ? early/late : 0

            printf "speed: first %.1f -> last %.1f step/s", spd[1], spd[n]
            if (n >= 3) printf "  (median of first third %.1f -> last third %.1f)", early, late
            printf "\n"

            if (n >= 3 && ratio >= 1.25)
                printf "SLOWDOWN: ended %.1fx slower than it started. A longer wallclock does not fix a rate that is still falling -- find what grows with steps (replay memory, visit counts, a graph never freed, a cgroup at its limit refaulting pages) before resubmitting.\n", ratio
            else if (n >= 3 && ratio <= 0.80)
                printf "SPEEDUP: ended %.1fx faster than it started -- the early checkpoints paid a warm-up cost the rest did not.\n", 1/ratio
            else if (n >= 3)
                printf "rate was stable end to end (within %.0f%%): this run was simply longer than its wallclock, so the timeout is the thing to raise.\n", (ratio > 1 ? ratio-1 : 1-ratio) * 100

            # A trend needs at least two samples, and both ends must come from
            # the same column: an eta against a left_s is not a comparison.
            if (n >= 2 && mixed)
                printf "the projection column changed meaning mid-run (%s -> %s): one of them excludes the remaining test() calls, so no trend is reported.\n", src[1], src[n]
            else if (n >= 2 && !bad(eta[1]) && !bad(eta[n])) {
                mid = int((n+1)/2)
                if (eta[n]+0 >= eta[1]+0)
                    printf "NOT CONVERGING: %s went %s -> %s. It has to FALL as steps are consumed; rising means the run was losing ground and would not have finished at any wallclock.\n", key, hms(eta[1]), hms(eta[n])
                else if (n >= 3 && !bad(eta[mid]) && eta[n]+0 > eta[mid]+0)
                    printf "%s fell then rose again (%s -> %s -> %s): the slowdown started mid-run.\n", key, hms(eta[1]), hms(eta[mid]), hms(eta[n])
            }

            if (!bad(tst[n]) && !bad(trn[n]) && trn[n]+0 > 0)
                printf "last interval: %.1fs training + %.1fs test (test was %.0f%% of the checkpoint)\n",
                       trn[n], tst[n], 100*tst[n]/(trn[n]+tst[n])
        }
    ' "$out_file"
}

# Print a log into the report: whole when short enough, otherwise head AND tail
# with the gap marked. Never the tail alone -- the command banner and the first
# import errors are at the top -- and never the head alone, which drops the
# slurmstepd lines at the end that say what killed the job.
dump_log() {
    local file="$1" label="$2" n gap
    n=$(wc -l < "$file")
    if (( n <= log_full_lines || n <= log_head_lines + log_tail_lines )); then
        echo "[$label in full ($n lines)]"
        cat "$file"
    else
        gap=$(( n - log_head_lines - log_tail_lines ))
        echo "[$label head (1-$log_head_lines of $n lines)]"
        head -n "$log_head_lines" "$file"
        echo ""
        echo "        ... $gap lines omitted ..."
        echo ""
        echo "[$label tail (last $log_tail_lines of $n lines)]"
        tail -n "$log_tail_lines" "$file"
    fi
    echo ""
}

have_sacct=0
command -v sacct >/dev/null 2>&1 && have_sacct=1

# SLURM duration -> seconds. Accepts D-HH:MM:SS, HH:MM:SS or MM:SS. Any
# non-numeric limit (UNLIMITED, Partition_Limit, empty) fails, so an unknown
# limit never gets reported as a wallclock kill.
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

# True if elapsed is close enough to the limit to call it a wallclock kill.
# Exact >= is wrong: submitit asks SLURM for a SIGUSR2 signal_delay_s before the
# hard limit, so a timed-out job characteristically dies at 01:28:23 of 01:30:00
# and a strict comparison files it under "cancelled early". Grace is the larger
# of 5% of the limit and 120s, which covers the default 90s delay plus teardown.
at_time_limit() {
    local a b grace pct
    a=$(hms_to_s "$1") || return 1
    b=$(hms_to_s "$2") || return 1
    (( b > 0 )) || return 1
    pct=$(( b * 5 / 100 ))
    grace=120; (( pct > grace )) && grace=$pct
    (( a + grace >= b ))
}

# Cached: one .err per job means one query per job, but an array-style submission
# puts several .err files under the same id and would repeat the query for each.
declare -A sacct_cache

sacct_query() {
    local id="$1" o
    if [[ -n "${sacct_cache[$id]+x}" ]]; then
        printf '%s' "${sacct_cache[$id]}"
        return 0
    fi
    o=$(sacct -j "$id" -S now-90days \
        --format=JobID,State,ExitCode,ReqMem,MaxRSS,MaxVMSize,Elapsed,Timelimit,ReqTRES%40,AllocTRES%40,JobName%200 \
        -P </dev/null 2>/dev/null) || o=""
    sacct_cache["$id"]="$o"
    printf '%s' "$o"
}

# --- jobs that never wrote a log -------------------------------------------
#
# submitit writes its markers before SLURM opens the log files: <job>_submission.sh
# for the submission itself and <array>_<element>_submitted.pkl for each array
# element. A marker whose *_log.out / *_log.err never appeared is therefore a job
# that never got far enough to write anything -- rejected at startup, cancelled
# while pending, node failure -- or one still sitting in the queue. The scan below
# iterates over .err files, so it cannot see any of these.
markers=( "$folder"/*_submitted.pkl "$folder"/*_submission.sh )

n_missing=0
n_old_markers=0
declare -A marker_seen
missing_txt=""

for m in ${markers[@]+"${markers[@]}"}; do
    # Outside the window. Skipped before marker_seen is written, so an id with
    # one stale marker and one fresh one is still reported through the fresh one.
    if ! recent_enough "$m"; then
        n_old_markers=$((n_old_markers+1))
        continue
    fi

    base=$(basename "$m")
    id="${base%_submitted.pkl}"
    id="${id%_submission.sh}"

    [[ -n "${marker_seen[$id]+x}" ]] && continue
    marker_seen["$id"]=1

    # An array leaves both a master marker (823523_submission.sh) and one per
    # element (823523_0_submitted.pkl). The elements are what own logs, so when
    # they exist the master is skipped: it would only repeat them.
    if [[ $id =~ ^[0-9]+$ ]]; then
        elems=( "$folder/${id}"_*_submitted.pkl )
        (( ${#elems[@]} > 0 )) && continue
    fi

    # One .err and one .out per task -- one of each for a joblib job, several when
    # the job was an array element; any of them says the job started writing.
    errs=( "$folder/${id}"_*_log.err )
    outs=( "$folder/${id}"_*_log.out )
    (( ${#errs[@]} > 0 && ${#outs[@]} > 0 )) && continue

    n_missing=$((n_missing+1))

    if (( ${#errs[@]} == 0 && ${#outs[@]} == 0 )); then
        what="neither ${id}_*_log.out nor ${id}_*_log.err exists"
    elif (( ${#errs[@]} == 0 )); then
        what="${id}_*_log.err is missing (stdout is there: $(basename "${outs[0]}"))"
    else
        what="${id}_*_log.out is missing (stderr is there: $(basename "${errs[0]}"))"
    fi

    m_state=""; m_elapsed=""; m_name=""
    if (( have_sacct )); then
        m_sacct=$(sacct_query "$id")
        m_state=$(printf '%s\n' "$m_sacct" \
            | awk -F'|' 'NR>1 && $1 !~ /\./ {print $2; exit}')
        m_elapsed=$(printf '%s\n' "$m_sacct" \
            | awk -F'|' 'NR>1 && $1 !~ /\./ {print $7; exit}')
        m_name=$(printf '%s\n' "$m_sacct" \
            | awk -F'|' 'NR>1 && $1 !~ /\./ {print $11; exit}')
    fi

    case "$m_state" in
        PENDING*)   m_diag="still queued -- the logs are created when the job starts, so nothing is wrong yet" ;;
        RUNNING*)   m_diag="running but nothing written yet (elapsed $m_elapsed) -- the log normally appears at once, so check that \$SCRATCH resolves to the scanned folder on the compute node" ;;
        COMPLETED*) m_diag="sacct says COMPLETED -- the logs were deleted, or written to a folder other than the one scanned" ;;
        CANCELLED*) m_diag="cancelled before producing any output (sacct: $m_state, elapsed $m_elapsed) -- scancel while pending, or the scheduler dropped it" ;;
        NODE_FAIL*) m_diag="node failure before the job could write anything -- resubmit" ;;
        BOOT_FAIL*) m_diag="node failed to boot -- resubmit" ;;
        PREEMPTED*) m_diag="preempted before writing anything -- resubmit" ;;
        TIMEOUT*)   m_diag="hit the wallclock without writing a log (sacct: TIMEOUT, elapsed $m_elapsed)" ;;
        FAILED*)    m_diag="failed before opening its logs (sacct: FAILED) -- usually the batch prologue, i.e. the module/venv setup lines of the submission script" ;;
        "")         m_diag="no sacct record -- the marker is written before sbatch returns, so a submission that errored out leaves exactly this; otherwise the job is older than 90 days and has been purged" ;;
        *)          m_diag="sacct: $m_state, elapsed $m_elapsed" ;;
    esac

    missing_txt+="JOB: $id"$'\n'
    [[ -n "$m_name" ]] && missing_txt+="NAME: $m_name"$'\n'
    missing_txt+="MARKER: $m"$'\n'
    missing_txt+="[missing]     $what"$'\n'
    missing_txt+="[diagnosis]   $m_diag"$'\n'
    missing_txt+=$'\n'
done

if (( n_missing > 0 )); then
    {
        echo "================================================================"
        echo "NO LOGS: $n_missing submitted job(s) never wrote .out/.err"
        echo "----------------------------------------------------------------"
        printf '%s' "$missing_txt"
    } >> "$out"
    echo "$n_missing submitted job(s) have no .out/.err -- see $out" >&2
fi

files=( "$folder"/*.err )
n_old=0

# Prune to the window up front rather than inside the loop: the progress bar,
# the per-file counters and the summary then all describe the same set of files.
if (( cutoff > 0 && ${#files[@]} > 0 )); then
    recent_files=()
    for f in "${files[@]}"; do
        if recent_enough "$f"; then
            recent_files+=("$f")
        else
            n_old=$((n_old+1))
        fi
    done
    files=( ${recent_files[@]+"${recent_files[@]}"} )
fi

if (( ${#files[@]} == 0 )); then
    if (( n_old > 0 )); then
        echo "No *.err files written in the last $within_min minute(s) in $folder ($n_old older file(s) skipped)" | tee -a "$out"
    else
        echo "No *.err files in $folder" | tee -a "$out"
    fi
    (( n_missing > 0 )) && echo "$n_missing submitted job(s) with no logs are reported above" | tee -a "$out"
    exit 0
fi

total=0
errored=0
n_clean=0      # no error patterns and sacct is happy -- not reported
n_warnonly=0   # of those, the ones that did print warnings
n_total=${#files[@]}

bar_width=40
draw_bar() {
    [[ -t 2 ]] || return 0
    local cur=$1 tot=$2 width=$bar_width
    local filled=$(( cur * width / tot ))
    local empty=$(( width - filled ))
    local pct=$(( cur * 100 / tot ))
    printf '\r[' >&2
    (( filled > 0 )) && printf '#%.0s' $(seq 1 $filled) >&2
    printf '%*s] %3d%%  %d/%d  err:%d' "$empty" '' "$pct" "$cur" "$tot" "$errored" >&2
}

for f in "${files[@]}"; do
    [[ "$f" -ef "$out" ]] && continue   # never scan our own report
    total=$((total+1))
    draw_bar "$total" "$n_total"

    job=$(basename "$f" .err)

    # Filename -> sacct id. submitit names logs <job_id>_<task>_log.err, where
    # job_id is <array>_<element> for an array task but a bare id for a plain job:
    #   842523_0_1_log.err -> job 842523_0, task 1
    #   842523_0_log.err   -> job 842523,   task 0
    # Dropping the trailing _<task>_log gets both right. Reading that last field as
    # an array element instead makes every sacct query for a plain job ask for
    # 842523_0, which was never an array element and returns nothing.
    elem_id=""
    stem="${job%_log}"
    cand="${stem%_*}"
    if [[ $cand =~ ^[0-9]+(_[0-9]+)?$ ]]; then
        elem_id="$cand"
    elif [[ $job =~ ^([0-9]+) ]]; then
        elem_id="${BASH_REMATCH[1]}"
    fi

    # sacct first: it is the authority on whether the job failed, so a task with
    # an unhelpful log still gets reported.
    sacct_out=""; job_state=""; job_elapsed=""; job_tlimit=""; job_jobname=""
    if (( have_sacct )) && [[ -n "$elem_id" ]]; then
        sacct_out=$(sacct_query "$elem_id")
        # First row without a dot in JobID is the job itself, not a step.
        job_state=$(printf '%s\n' "$sacct_out" \
            | awk -F'|' 'NR>1 && $1 !~ /\./ {print $2; exit}')
        job_jobname=$(printf '%s\n' "$sacct_out" \
            | awk -F'|' 'NR>1 && $1 !~ /\./ {print $11; exit}')
        job_elapsed=$(printf '%s\n' "$sacct_out" \
            | awk -F'|' 'NR>1 && $1 !~ /\./ {print $7; exit}')
        job_tlimit=$(printf '%s\n' "$sacct_out" \
            | awk -F'|' 'NR>1 && $1 !~ /\./ {print $8; exit}')
        # Requested memory and peak RSS are read once, inside the digest awk
        # below. They are reported, never turned into a verdict, so nothing up
        # here needs them.
    fi

    # One pass over the log for everything the log itself can tell us.
    IFS="$US" read -r n_err n_warn has_sig2 has_oom cmd_line < <(scan_log "$f")

    # Whether a task failed is not a question the log can answer on its own: a
    # traceback the patterns happen not to match, or a hard kill that wrote
    # nothing, would be dropped silently. Take failure from either source --
    # but never from warnings alone, which is the whole point of the benign list.
    failed=0
    log_only=0
    case "$job_state" in
        FAILED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|CANCELLED*|PREEMPTED*|BOOT_FAIL*|DEADLINE*|REVOKED*)
            failed=1 ;;
    esac
    if (( ! failed && n_err > 0 )); then
        # sacct calls it COMPLETED but the log says otherwise. Usually a wrapper
        # swallowing a non-zero exit; occasionally a stale pattern. Off by
        # default so a healthy sweep does not produce a report full of noise.
        case "$job_state" in
            COMPLETED*) (( include_ok )) && { failed=1; log_only=1; } ;;
            *)          failed=1; log_only=1 ;;
        esac
    fi

    if (( ! failed )); then
        n_clean=$((n_clean+1))
        (( n_warn > 0 )) && n_warnonly=$((n_warnonly+1))
        continue
    fi

    errored=$((errored+1))

    {
        echo "================================================================"
        echo "JOB: $job"
        [[ -n "$job_jobname" ]] && echo "NAME: $job_jobname"
        echo "FILE: $f"
        echo "----------------------------------------------------------------"
    } >> "$out"

    # Kills by SLURM leave no traceback, only a cryptic srun line, so name them
    # up front. "User defined signal 2" is SIGUSR2: SLURM warning submitit that
    # the wallclock limit is about to expire.
    diag=""
    (( has_sig2 )) && diag="wallclock limit reached (SIGUSR2) -- raise this env's timeout"
    (( has_oom ))  && diag="out of memory -- the tasks share one --mem pool. Run inspect_jobs.sh against a live job to know how memory is allocated. This will tell you the right amount of memory to request."
    if [[ -z "$diag" ]]; then
        # Kills that leave no message in the log at all: only sacct knows.
        case "$job_state" in
            OUT_OF_MEMORY*)
                diag="out of memory (sacct) -- the tasks share one --mem pool. Run inspect_jobs.sh against a live job to know how memory is allocated. This will tell you the right amount of memory to request." ;;
            CANCELLED*)
                # CANCELLED is ambiguous: a real scancel and a wallclock kill
                # both land here. submitit's SIGUSR2 arrives signal_delay_s
                # BEFORE the limit, so "reached the limit" means "close to it".
                if [[ -n "$job_elapsed" && -n "$job_tlimit" ]] \
                   && at_time_limit "$job_elapsed" "$job_tlimit"; then
                    diag="wallclock limit reached (cancelled at time limit, elapsed $job_elapsed of limit $job_tlimit) -- raise this env's timeout"
                else
                    diag="cancelled early -- scancel, by you or by the scheduler (sacct: $job_state, elapsed $job_elapsed of limit $job_tlimit)"
                fi ;;
            TIMEOUT*)   diag="wallclock limit reached (sacct: TIMEOUT) -- raise this env's timeout" ;;
            NODE_FAIL*) diag="node failure, nothing to do with the code -- resubmit" ;;
            PREEMPTED*) diag="preempted by a higher-priority job -- resubmit" ;;
            BOOT_FAIL*) diag="node failed to boot -- resubmit" ;;
            DEADLINE*)  diag="hit the partition deadline (sacct: $job_state)" ;;
        esac
    fi
    [[ -n "$diag" ]] && { echo "[diagnosis] $diag"; echo ""; } >> "$out"

    # Caveats that change how the section should be read.
    notes=()
    if (( failed && n_err == 0 )); then
        notes+=("this task's own log has no errors in it (${n_warn} benign/warning line(s)); it is reported because sacct says the JOB is $job_state. On a chunked job the cause is usually a SIBLING task -- check the other ${elem_id}_*_log.err files.")
    fi
    (( log_only )) && [[ -n "$job_state" ]] && notes+=("sacct reports $job_state but the log matched an error pattern -- something between the run and the exit code is swallowing the failure.")
    if (( ${#notes[@]} > 0 )); then
        for nline in "${notes[@]}"; do echo "[note] $nline"; done >> "$out"
        echo "" >> "$out"
    fi

    # The exact command, so it can be copy-pasted to reproduce interactively.
    # run_sweep echoes "[task N] <cmd>" to stderr before launching, flushed, so
    # it survives a hard kill.
    [[ -n "$cmd_line" ]] && { echo "[command]"; echo "$cmd_line"; echo ""; } >> "$out"

    # Speed history, from the .out next to this .err. Nothing in .err says how
    # fast the run was going or how much it thought was left; that lives in the
    # per-checkpoint dict lines, and it is the difference between "raise the
    # timeout" and "the timeout was never the problem".
    out_file="${f%.err}.out"
    if [[ -s "$out_file" ]]; then
        prog=$(progress_from_out "$out_file")
        if [[ -n "$prog" && "$prog" != "NONE" ]]; then
            {
                echo "[progress] from $(basename "$out_file")"
                printf '%s\n' "$prog" | sed 's/^/           /'
                echo ""
            } >> "$out"
        elif [[ "$prog" == "NONE" ]]; then
            {
                echo "[progress] $(basename "$out_file") has no checkpoint lines -- the run died before its first checkpoint, or it was launched with progress_report != \"dict\"."
                echo ""
            } >> "$out"
        fi
    else
        {
            echo "[progress] no ${job}.out to read (empty or missing) -- run with experiment.progress_report=dict to get a speed history."
            echo ""
        } >> "$out"
    fi

    # State + memory requested vs used.
    if (( have_sacct )) && [[ -n "$elem_id" ]]; then
        n_rows=$(printf '%s\n' "$sacct_out" | tail -n +2 | grep -c . || true)
        if (( n_rows > 0 )); then
            # Digest first: the raw table spreads these across rows (ReqMem and
            # the allocated TRES sit on the job row; MaxRSS/MaxVMSize populate
            # only on the .batch/.0 sub-step rows), so pull the max/first of each
            # into one line that answers "requested vs allocated vs peak, elapsed
            # vs limit" without reading the table.
            digest=$(printf '%s\n' "$sacct_out" | awk -F'|' '
                function tres_mem(s,   i,a,n) {
                    n=split(s,a,","); for(i=1;i<=n;i++) if(a[i]~/^mem=/){sub(/^mem=/,"",a[i]); return a[i]} return ""
                }
                function tres_cpu(s,   i,a,n) {
                    n=split(s,a,","); for(i=1;i<=n;i++) if(a[i]~/^cpu=/){sub(/^cpu=/,"",a[i]); return a[i]} return ""
                }
                NR==1 { next }
                {
                    if ($4!="" && reqmem=="")   reqmem=$4          # ReqMem, job row
                    if (reqtres=="")   reqtres=tres_mem($9)        # ReqTRES
                    if (alloctres=="") alloctres=tres_mem($10)     # AllocTRES
                    if (allalloc=="")  allalloc=tres_cpu($10)      # allocated cpus
                    if ($5!="" && $1 !~ /\.(batch|extern)$/) {
                        m=$5; mn=m+0; u=m; gsub(/[0-9.]/,"",u)
                        kb = (u~/^G/? mn*1024*1024 : (u~/^M/? mn*1024 : mn))
                        if (kb+0 > maxrss_kb+0) { maxrss_kb=kb; maxrss=m }
                    }
                    if ($7!="" && elapsed=="") elapsed=$7
                    if ($8!="" && tlimit=="") tlimit=$8
                }
                END {
                    req = (reqtres!=""? reqtres : reqmem)
                    printf "req_mem=%s  alloc_mem=%s  peak_rss=%s  cpus=%s  |  elapsed=%s  limit=%s",
                           (req!=""?req:"?"), (alloctres!=""?alloctres:"?"),
                           (maxrss!=""?maxrss:"?"), (allalloc!=""?allalloc:"?"),
                           (elapsed!=""?elapsed:"?"), (tlimit!=""?tlimit:"?")
                }')
            {
                [[ -n "$digest" ]] && echo "[sacct $elem_id] $digest"
                echo "(peak_rss is the largest SINGLE TASK of the compute steps -- not the job's total. It excludes the sibling tasks, .batch, the submitit parents, the squashfuse mounts, and all file-backed memory, so expect it well below ReqMem even on a genuine OOM. Do not read it as a memory verdict; use inspect_jobs.sh on a live job for that.)"
                echo "$sacct_out"
                echo ""
            } >> "$out"
        else
            echo "[sacct $elem_id: no records (older than 90d or purged)]" >> "$out"
            echo "" >> "$out"
        fi
    fi

    # The log itself. One seed per file, so no slicing: whatever is here belongs
    # to this run. Short logs go in whole; long ones get the pattern matches with
    # context first, then head and tail.
    if [[ ! -s "$f" ]]; then
        echo "[.err] the file is empty -- the task died before writing anything" >> "$out"
        echo "" >> "$out"
    else
        n_lines=$(wc -l < "$f")
        if (( n_lines > log_full_lines && n_err > 0 )); then
            {
                echo "[matched patterns ($n_err line(s), +-${ctx_before}/${ctx_after} context)]"
                show_context "$f"
                echo ""
            } >> "$out"
        fi
        dump_log "$f" ".err" >> "$out"
    fi
done

# Clear progress bar line.
[[ -t 2 ]] && printf '\r\033[K' >&2

{
    echo "================================================================"
    echo "SUMMARY: $errored / $total .err file(s) reported as failed"
    echo "         $n_clean file(s) clean and skipped ($n_warnonly of them had warnings only)"
    (( n_missing > 0 )) && echo "         $n_missing submitted job(s) never wrote .out/.err (listed at the top)"
    if (( cutoff > 0 )); then
        echo "         window: last $within_min minute(s), since $cutoff_str"
        (( n_old > 0 ))         && echo "         $n_old .err file(s) older than the window and not scanned"
        (( n_old_markers > 0 )) && echo "         $n_old_markers submission marker(s) older than the window and not checked"
    fi
    echo "Folder scanned: $folder"
    echo "Written: $out"
} | tee -a "$out"
