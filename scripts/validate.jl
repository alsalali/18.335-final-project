# Checks before the sweep (HANDOFF.md "Validation order"):
#   1. gasoil (smooth): mesh-point state error vs h; expect order ~2K (legendre), ~2K-1 (radau).
#   2. rocket (bang-bang): low order whatever K is.
#   3. K = 4 Legendre objectives against COPSBenchmark.jl's expected values at the same N.
#      The estimation problems use 4-point Gauss collocation in COPS too, so they should agree
#      closely; the control problems use other COPS discretizations, so expect O(h) agreement.
#   4. (--gpu) GPU and CPU lifted objectives agree to about tol.
#
#   julia --project=. scripts/validate.jl [--steps 1,2,3] [--gpu]

include(joinpath(@__DIR__, "..", "src", "Screen.jl"))
using .Screen
using MadNLP, Printf, Statistics, LinearAlgebra

const STEPS = let i = findfirst(==("--steps"), ARGS)
    i === nothing ? [1, 2, 3] : parse.(Int, split(ARGS[i+1], ','))
end
const USE_GPU = "--gpu" in ARGS
if USE_GPU
    using CUDA, CUDSS, MadNLPGPU
end

const CPU = cpu_device("cpu", MadNLP.MumpsSolver, 4)

solve(problem, family, K, N; config="exact", dev=CPU) =
    solve_point(COPS_PROBLEMS[problem], problem, family, K, N, CONFIGS[config], dev; max_wall_time=600.0)

states(r) = sample_states(r.layout, r.scheme, r.sol, 200)
const MESH = [1 + 40m for m in 0:5]

function mesh_error(r, ref)
    X, R = states(r), states(ref)
    scale = max.(1.0, vec(maximum(abs.(R); dims=1)))
    return maximum((abs.(X .- R) ./ permutedims(scale))[MESH, :])
end

function convergence_table(problem, families, Ks, Ns, ref)
    @printf("%-9s %-9s %2s %6s %11s %7s %s\n", "problem", "family", "K", "N", "err_mesh", "order", "status")
    for family in families, K in Ks
        prev = NaN
        for N in Ns
            r = solve(problem, family, K, N)
            e = mesh_error(r, ref)
            order = isfinite(prev) ? log2(prev / e) : NaN
            @printf("%-9s %-9s %2d %6d %11.3e %7.2f %s\n", problem, family, K, N, e, order, r.status)
            prev = e
            e < 1e-10 && break   # at the solver floor
        end
    end
end

if 1 in STEPS
    println("\n== 1. gasoil: expect ~2K (legendre) and ~2K-1 (radau) at mesh points")
    ref = solve(:gasoil, :legendre, 8, 160; config="reference")
    @printf("reference legendre K=8 N=160: %s objective %.10e p = %s\n", ref.status, ref.objective,
            join((@sprintf("%.8f", v) for v in ref.sol.p), ", "))
    convergence_table(:gasoil, (:legendre, :radau), 1:4, [5, 10, 20, 40, 80], ref)
end

# Step 1 mixes two error sources: the ODE discretization and the parameter estimate, whose
# least-squares fit interpolates states at measurement times that are not mesh points. Fixing
# the parameters at the reference values (lvar = uvar) leaves a pure initial value problem, where
# mesh-point superconvergence should be clean.
function with_fixed_parameters(spec, p)
    COPSProblems.ODEProblemSpec(spec.name, spec.nx, spec.nu, spec.np, spec.rhs, spec.t0, spec.tf, spec.x0,
        spec.xbounds, spec.ubounds, [(v, v) for v in p], spec.objective, spec.tmeas, spec.ymeas,
        merge(spec.guess, (p=collect(p),)), spec.xf, spec.terminal_cost)
end

if 4 in STEPS
    println("\n== 1b. gasoil as an IVP (parameters fixed at the reference): expect 2K (legendre), 2K-1 (radau)")
    ref = solve(:gasoil, :legendre, 8, 160; config="reference")
    spec = with_fixed_parameters(COPS_PROBLEMS[:gasoil], ref.sol.p)
    refivp = solve_point(spec, :gasoil, :legendre, 10, 160, CONFIGS["reference"], CPU)
    @printf("%-9s %2s %6s %11s %7s\n", "family", "K", "N", "err_mesh", "order")
    for family in (:legendre, :radau), K in 1:4
        prev = NaN
        for N in [5, 10, 20, 40, 80]
            r = solve_point(spec, :gasoil, family, K, N, CONFIGS["exact"], CPU)
            e = mesh_error(r, refivp)
            @printf("%-9s %2d %6d %11.3e %7.2f %s\n", family, K, N, e, isfinite(prev) ? log2(prev / e) : NaN, r.status)
            prev = e
            e < 1e-11 && break
        end
    end
end

if 2 in STEPS
    println("\n== 2. rocket: expect low order for every K")
    ref = solve(:rocket, :radau, 4, 1280; config="reference")
    @printf("reference radau K=4 N=1280: %s objective %.10f tf %.8f\n", ref.status, ref.objective, ref.sol.tf)
    convergence_table(:rocket, (:radau,), 1:4, [10, 20, 40, 80, 160], ref)
end

# COPSBenchmark.jl test/runtests.jl: (problem, nh, expected objective). Sign conventions differ
# for maximization problems, so magnitudes are compared.
const COPS_EXPECTED = [
    (:gasoil, 100, 5.2366e-3), (:methanol, 100, 9.02229e-3), (:pinene, 100, 1.98721e1),
    (:marine, 100, 1.97462e7), (:catmix, 100, -4.80556e-2), (:glider, 100, -1.25505e3),
    (:robot, 200, 9.14138), (:rocket, 400, 1.01283), (:steering, 200, 5.54577e-1), (:chain, 800, 5.06891),
]

if 3 in STEPS
    println("\n== 3. K=4 Legendre vs COPSBenchmark.jl expected objectives (magnitudes)")
    @printf("%-9s %5s %14s %14s %10s %s\n", "problem", "N", "ours", "COPS", "rel diff", "status")
    for (problem, N, expected) in COPS_EXPECTED
        r = solve(problem, :legendre, 4, N)
        d = abs(abs(r.objective) - abs(expected)) / abs(expected)
        @printf("%-9s %5d %14.6e %14.6e %10.2e %s\n", problem, N, r.objective, expected, d, r.status)
    end
end

if USE_GPU
    println("\n== 4. GPU vs CPU, lifted config")
    gpu = Device("gpu", CUDABackend(), MadNLPGPU.CUDSSSolver, 1, () -> CUDA.synchronize(), () -> GC.gc(),
                 () -> (GC.gc(); CUDA.reclaim()), CUDA.name(CUDA.device()), string(CUDA.driver_version()))
    for problem in (:gasoil, :rocket, :robot, :marine), (family, K, N) in ((:radau, 2, 40), (:legendre, 4, 80))
        rc = solve(problem, family, K, N; config="lifted")
        rg = solve(problem, family, K, N; config="lifted", dev=gpu)
        @printf("%-9s %-9s K=%d N=%-4d cpu %.10e (%s)  gpu %.10e (%s)  rel diff %.1e\n", problem, family, K, N,
                rc.objective, rc.status, rg.objective, rg.status,
                abs(rc.objective - rg.objective) / max(1, abs(rc.objective)))
    end
end
