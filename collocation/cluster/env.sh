# Sourced by cluster/setup.sh and both sbatch scripts. Paths can be overridden from the environment.
module load julia/1.12.6
# The depot holds CUDA artifacts (several GB); home is close to its quota, so keep it on pool storage.
export JULIA_DEPOT_PATH="${SCREEN_DEPOT:-$HOME/orcd/pool/julia_depot}:"
# Compile package images for every x86 CPU family (same targets as the official binaries), so caches built on
# the AMD login node also load on the Intel/AMD compute nodes instead of recompiling inside each job.
export JULIA_CPU_TARGET='generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)'
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_THREADS=1
