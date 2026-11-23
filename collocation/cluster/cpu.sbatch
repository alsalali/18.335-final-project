#!/bin/bash
#SBATCH --job-name=coll-cpu
#SBATCH --partition=mit_preemptable
#SBATCH --array=0-31
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue                    # preemption on mit_preemptable requeues the job
#SBATCH --signal=B:USR1@300          # 5 min before the time limit, signal this script (trap below)
#SBATCH --open-mode=append           # one log across attempts (job ids are stable across requeues)
#SBATCH --output=cluster/logs/%x_%A_%a.out
#
# CPU-only job: the exact (accuracy) and lifted (timing) configs on the CPU.
# Submit from the collocation folder (log paths are relative to it):
#   sbatch cluster/cpu.sbatch sweep full                              # the screen (32 tasks)
#   sbatch --array=0-9 -J coll-ref cluster/cpu.sbatch reference full --max-wall 14400   # references, one task per problem
#   sbatch -p mit_quicktest --time=00:15:00 --array=0-1 --no-requeue cluster/cpu.sbatch sweep smoke
# Extra arguments go to scripts/run_sweep.jl (e.g. --problems gasoil,rocket).
exec bash "${SLURM_SUBMIT_DIR:?submit from the collocation folder}/cluster/job_body.sh" cpu "$@"
