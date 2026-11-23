#!/bin/bash
#SBATCH --job-name=coll-gpu
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:1            # FP64-capable only (a100, h100, h200, b200); keep one type per sweep
#SBATCH --array=0-3                  # the QoS allows 4 GPUs per user
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue                    # preemption on mit_preemptable requeues the job
#SBATCH --signal=B:USR1@300          # 5 min before the time limit, signal this script (trap below)
#SBATCH --open-mode=append           # one log across attempts (job ids are stable across requeues)
#SBATCH --output=cluster/logs/%x_%A_%a.out
#
# GPU job: the lifted config on the GPU, plus the headline subset on this node's CPU for a
# same-hardware comparison. Submit from the collocation folder (log paths are relative to it):
#   sbatch cluster/gpu.sbatch sweep full
#   sbatch --time=00:30:00 --array=0 cluster/gpu.sbatch sweep smoke
#   sbatch --time=02:00:00 --array=0 cluster/gpu.sbatch sweep pilot
#   sbatch --gres=gpu:a100:1 cluster/gpu.sbatch sweep full           # another GPU type
exec bash "${SLURM_SUBMIT_DIR:?submit from the collocation folder}/cluster/job_body.sh" gpu "$@"
