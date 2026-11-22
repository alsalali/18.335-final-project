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

cd "${SLURM_SUBMIT_DIR:?}"
source cluster/env.sh
export JULIA_PKG_OFFLINE=true      # never touch the registry from a job
export OPENBLAS_NUM_THREADS=$THREADS
mkdir -p cluster/logs results

echo "=== $(date -Is) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-}_${TASK} attempt=$ATTEMPT" \
     "host=$(hostname) profile=$PROFILE stage=$STAGE grid=$GRID task=$TASK/$NTASKS threads=$THREADS"
echo "=== cpu: $(lscpu | sed -n 's/^Model name:[[:space:]]*//p')"
[ "$PROFILE" = gpu ] && { nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; }

trap 'echo "=== $(date -Is) near the time limit: requeueing job $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"' USR1

rc=1
for launch in 1 2 3 4 5; do
    julia --project=. --startup-file=no scripts/run_sweep.jl --profile "$PROFILE" --stage "$STAGE" \
        --grid "$GRID" --task "$TASK" --ntasks "$NTASKS" --threads "$THREADS" --launch "$launch" "$@" &
    pid=$!
    wait "$pid"; rc=$?
    while kill -0 "$pid" 2>/dev/null; do wait "$pid"; rc=$?; done   # a trapped signal interrupts `wait`
    echo "=== $(date -Is) julia launch $launch exited with $rc"
    [ "$rc" -eq 0 ] && break
done
exit "$rc"
