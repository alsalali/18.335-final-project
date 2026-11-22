#!/bin/bash
# One-time setup on the login node (it needs internet; re-running is safe):
#   bash cluster/setup.sh
# Instantiates the pinned Project/Manifest into the depot on pool storage and precompiles for
# every CPU target in JULIA_CPU_TARGET. CUDA's runtime is resolved on the first GPU job.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/env.sh
mkdir -p cluster/logs results
julia --project=. --startup-file=no -e '
    using Pkg
    Pkg.instantiate()
    Pkg.precompile()
    Pkg.status()'
julia --project=. --startup-file=no -e '
    include("src/Screen.jl"); using .Screen
    println("Screen loads. CPU target: ", unsafe_string(Base.JLOptions().cpu_target))'
echo "setup done: depot ${JULIA_DEPOT_PATH%:}"
