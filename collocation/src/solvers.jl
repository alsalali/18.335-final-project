# Solver configurations, devices, and the timed solve of one grid point.

"""
A MadNLP configuration.
- `exact`: augmented KKT, equalities enforced, tight tol. For accuracy measurements.
- `lifted`: condensed KKT with relaxed equalities (LiftedKKT, as in Choi & Shin). For timing.
  Defects only hold to about tol, so its errors bottom out near 1e-7.
- `reference`: `exact` with a tighter tol, for the reference solutions.
"""
struct SolverConfig
    name::String
    kkt_system::Type
    equality_treatment::Type
    fixed_variable_treatment::Type
    tol::Float64
    acceptable_tol::Float64
    timing_reruns::Bool
end

const CONFIGS = Dict(
    "exact" => SolverConfig("exact", MadNLP.SparseKKTSystem, MadNLP.EnforceEquality,
                            MadNLP.MakeParameter, 1e-10, 1e-8, false),
    "lifted" => SolverConfig("lifted", MadNLP.SparseCondensedKKTSystem, MadNLP.RelaxEquality,
                             MadNLP.RelaxBound, 1e-8, 1e-6, true),
    "reference" => SolverConfig("reference", MadNLP.SparseKKTSystem, MadNLP.EnforceEquality,
                                MadNLP.MakeParameter, 1e-11, 1e-9, false),
)

"""
Where a run executes. `cpu_gpunode` is the CPU of the node that hosts the GPU job, for a
same-hardware CPU vs GPU comparison. GPU hooks are filled in by the driver script, which
loads CUDA; this module never does.
"""
struct Device
    name::String
    backend::Any
    linear_solver::Type
    blas_threads::Int
    sync::Function        # wait for queued device work before stopping a clock
    gc::Function          # between reruns: free dead allocations, keep pools warm
    reclaim::Function     # between grid points: also return pooled device memory
    gpu_name::String
    driver::String
end

cpu_device(name, linear_solver, threads) =
    Device(name, nothing, linear_solver, threads, () -> nothing, () -> GC.gc(), () -> GC.gc(), "", "")

const COST_STATUSES = ("MAXIMUM_WALLTIME_EXCEEDED", "CRASHED", "KILLED", "ERROR_OOM")
const CONVERGED_STATUSES = ("SOLVE_SUCCEEDED", "SOLVED_TO_ACCEPTABLE_LEVEL")

function solver_options(cfg::SolverConfig, dev::Device; max_iter, max_wall_time)
    return (; kkt_system=cfg.kkt_system, equality_treatment=cfg.equality_treatment,
            fixed_variable_treatment=cfg.fixed_variable_treatment, linear_solver=dev.linear_solver,
            tol=cfg.tol, acceptable_tol=cfg.acceptable_tol, acceptable_iter=15,
            max_iter, max_wall_time=Float64(max_wall_time), blas_num_threads=dev.blas_threads,
            print_level=MadNLP.ERROR)
end

# Same as the reference repo: construct and solve, the way `madnlp(model)` does.
function timed_solve(model, opts, dev::Device)
    dev.sync()
    t0 = time_ns()
    solver = MadNLP.MadNLPSolver(model; opts...)
    stats = MadNLP.solve!(solver)
    dev.sync()
    return stats, solver, (time_ns() - t0) / 1e9
end

"Shifted geometric mean, as in the reference benchmark (delta = 0.01 s)."
sgm(times; delta=0.01) = exp(mean(log.(times .+ delta))) - delta

rerun_count(t_first) = t_first < 10 ? 5 : t_first < 120 ? 3 : 0

function kkt_size(solver)
    try
        A = solver.kkt.aug_com
        return SparseArrays.nnz(A), size(A, 1)
    catch
        return -1, -1
    end
end

"""
    solve_point(spec, problem, family, K, N, cfg, dev; max_iter, max_wall_time)

Build, solve and time one grid point. Returns a NamedTuple with the stats, the host-side
solution, timings and sizes. Exceptions propagate to the caller.
"""
function solve_point(spec, problem::Symbol, family::Symbol, K::Int, N::Int, cfg::SolverConfig,
                     dev::Device; max_iter=3000, max_wall_time=1800.0)
    scheme = collocation_scheme(family, K)
    dev.reclaim()
    t0 = time_ns()
    model, layout = build_collocation_model(spec, scheme, N; backend=dev.backend, problem)
    dev.sync()
    t_build = (time_ns() - t0) / 1e9

    opts = solver_options(cfg, dev; max_iter, max_wall_time)
    stats, solver, t_first = timed_solve(model, opts, dev)
    nnz_kkt, kkt_dim = kkt_size(solver)
    sol = extract_solution(layout, stats.solution)
    status = string(stats.status)
    objective = Float64(stats.objective)
    iters = stats.iter
    primal_inf, dual_inf = Float64(stats.primal_feas), Float64(stats.dual_feas)
    solver = nothing
    stats = nothing

    nre = cfg.timing_reruns && string(status) in CONVERGED_STATUSES ? rerun_count(t_first) : 0
    times = Float64[]
    for _ in 1:nre
        dev.gc()                      # outside the timed region
        _, _, t = timed_solve(model, opts, dev)
        push!(times, t)
    end
    t_sgm = isempty(times) ? t_first : sgm(times)

    return (; status, objective, iters, primal_inf, dual_inf, layout, scheme, sol,
            t_build, t_first, t_sgm, n_reruns=nre, nnz_kkt, kkt_dim,
            nvar=model.meta.nvar, ncon=model.meta.ncon, nnzj=model.meta.nnzj, nnzh=model.meta.nnzh)
end
