"""
Collocation screen on the COPS ODE problems: transcription, solver configurations, the
K x N grid, and resumable result storage. Device-specific packages (CUDA, MadNLPGPU) are
loaded by the driver script, not here.
"""
module Screen

using ExaModels
using MadNLP
using LinearAlgebra
using Printf
using Dates
using Statistics
import SparseArrays

include(joinpath(@__DIR__, "..", "collocation_schemes.jl"))
include(joinpath(@__DIR__, "..", "cops_problems.jl"))
using .CollocationSchemes
using .COPSProblems
using .COPSProblems: ODEProblemSpec

include("transcription.jl")
include("solvers.jl")
include("grid.jl")
include("store.jl")
include("runner.jl")

export CollocationSchemes, COPSProblems, COPS_PROBLEMS, collocation_scheme, interpolation_weights,
       build_collocation_model, extract_solution, sample_states, ModelLayout, estimate_nvar, time_scale,
       SolverConfig, CONFIGS, Device, cpu_device, solve_point, sgm,
       Group, groups_for, shard, run_key, group_id, scheme_family, reference_plan, PROBLEM_ORDER, N_LADDER,
       Store, RunContext, load_done, read_shard, run_group!, parse_floats, COLUMNS

end # module Screen
