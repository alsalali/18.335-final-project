#!/bin/bash
# Full screen: reference solutions (10 CPU tasks), the CPU sweep (32 tasks) and the GPU sweep
# (4 GPUs). They are independent: errors are computed afterwards by scripts/collect.jl.
#   bash cluster/submit.sh [grid]       grid defaults to full
set -euo pipefail
cd "$(dirname "$0")/.."
GRID=${1:-full}
mkdir -p cluster/logs results
# References are few but large (nonsmooth problems: radau K=4 at N=20480), so allow 4 h per solve.
ref=$(sbatch --parsable --array=0-9 -J coll-ref cluster/cpu.sbatch reference "$GRID" --max-wall 14400)
cpu=$(sbatch --parsable cluster/cpu.sbatch sweep "$GRID")
gpu=$(sbatch --parsable cluster/gpu.sbatch sweep "$GRID")
echo "reference=$ref cpu=$cpu gpu=$gpu"
echo "watch:   squeue -u $USER; tail -f cluster/logs/coll-cpu_${cpu}_0.out"
echo "status:  sacct -j $cpu,$gpu,$ref --format=JobID,State,Restarts,Elapsed,MaxRSS"
echo "collect: source cluster/env.sh && julia --project=. scripts/collect.jl --grid $GRID"
