#!/bin/bash
# Shared body of cluster/cpu.sbatch and cluster/gpu.sbatch (they exec into it, so this shell keeps
# the batch PID and receives --signal=B:USR1).
#   job_body.sh <profile> <stage> <grid> [extra run_sweep.jl args...]
#
# Requeue and resume:
# - Preemption: Slurm requeues the job itself (--requeue, PreemptMode=REQUEUE).
# - Time limit: Slurm does not requeue, so USR1 (5 min before the limit) triggers scontrol requeue.
# - Each attempt reruns run_sweep.jl, which skips keys already in the shard CSVs and retries the
#   run that was in flight (an in-flight sentinel records it).
# - If Julia itself dies (segfault, OOM kill, cuDSS crash) the loop below relaunches it in the
#   same allocation; the driver then marks the run it died on as CRASHED and moves on.
# - Hang watchdog: MadNLP checks max_wall_time only between iterations, and one MUMPS
#   factorization can run for hours. Julia logs a line after every run, so if the log has not
#   changed for max-wall + 15 min, the watchdog kills Julia and the next launch records CRASHED.
set -uo pipefail
PROFILE=${1:?profile}; STAGE=${2:-sweep}; GRID=${3:-full}
shift $(( $# < 3 ? $# : 3 ))
ATTEMPT=${SLURM_RESTART_COUNT:-0}
TASK=${SLURM_ARRAY_TASK_ID:-0}
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]; then
    NTASKS=$SLURM_ARRAY_TASK_COUNT
elif [ -n "${SLURM_ARRAY_TASK_MAX:-}" ]; then
    NTASKS=$(( SLURM_ARRAY_TASK_MAX - ${SLURM_ARRAY_TASK_MIN:-0} + 1 ))
else
    NTASKS=1
fi
THREADS=${SLURM_CPUS_PER_TASK:-1}
MAX_WALL=1800                      # run_sweep.jl's default --max-wall
args=("$@")
for (( i = 0; i + 1 < ${#args[@]}; i++ )); do
    [ "${args[i]}" = "--max-wall" ] && MAX_WALL=${args[i+1]%.*}
done
HANG_LIMIT=${SWEEP_HANG_LIMIT:-$(( MAX_WALL + 900 ))}
LOG=$(readlink -f /proc/$$/fd/2 2>/dev/null || true)
WATCH_POLL=${SWEEP_WATCH_POLL:-60}

cd "${SLURM_SUBMIT_DIR:?}"
source cluster/env.sh
export JULIA_PKG_OFFLINE=true      # never touch the registry from a job
export OPENBLAS_NUM_THREADS=$THREADS
mkdir -p cluster/logs results

echo "=== $(date -Is) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-}_${TASK} attempt=$ATTEMPT" \
     "host=$(hostname) profile=$PROFILE stage=$STAGE grid=$GRID task=$TASK/$NTASKS threads=$THREADS hang_limit=${HANG_LIMIT}s"
echo "=== cpu: $(lscpu | sed -n 's/^Model name:[[:space:]]*//p')"
[ "$PROFILE" = gpu ] && { nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; }

trap 'echo "=== $(date -Is) near the time limit: requeueing job $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"' USR1

watchdog() {
    local pid=$1 age
    [ -f "$LOG" ] || return 0
    while sleep "$WATCH_POLL"; kill -0 "$pid" 2>/dev/null; do
        age=$(( $(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || date +%s) ))
        if (( age > HANG_LIMIT )); then
            echo "=== $(date -Is) watchdog: no log output for ${age}s (limit ${HANG_LIMIT}s), killing julia $pid"
            kill -TERM "$pid" 2>/dev/null; sleep 30     # SIGTERM makes Julia print a backtrace
            kill -KILL "$pid" 2>/dev/null
            return 0
        fi
    done
}

rc=1
for launch in $(seq 1 20); do       # every relaunch records the run it died on, so this always progresses
    julia --project=. --startup-file=no scripts/run_sweep.jl --profile "$PROFILE" --stage "$STAGE" \
        --grid "$GRID" --task "$TASK" --ntasks "$NTASKS" --threads "$THREADS" --launch "$launch" "$@" &
    pid=$!
    watchdog "$pid" & wd=$!
    wait "$pid"; rc=$?
    while kill -0 "$pid" 2>/dev/null; do wait "$pid"; rc=$?; done   # a trapped signal interrupts `wait`
    kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
    echo "=== $(date -Is) julia launch $launch exited with $rc"
    [ "$rc" -eq 0 ] && break
done
exit "$rc"
