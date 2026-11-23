# Driver for both SLURM jobs (and for local runs).
#
#   julia --project=. scripts/run_sweep.jl --profile cpu|gpu --stage sweep|reference \
#         --grid full|smoke|pilot [--task i --ntasks n --launch k --threads t] \
#         [--problems gasoil,rocket] [--cpu-solver mumps] [--max-wall 1800] [--max-iter 3000] \
#         [--results dir] [--dry-run]          (--results defaults to results/ in this folder)
#
# Resumable: finished keys (in any shard file) are skipped, so rerunning the same command
# after preemption, a time limit or a crash continues where it stopped. Exits 0 when the
# task's shard is complete.

const ARGS_DEFAULT = Dict(
    "profile" => "cpu", "stage" => "sweep", "grid" => "smoke", "task" => "0", "ntasks" => "1",
    "launch" => "1", "threads" => "1", "problems" => "", "cpu-solver" => "mumps",
    "max-wall" => "1800", "max-iter" => "3000", "results" => normpath(joinpath(@__DIR__, "..", "results")),
    "dry-run" => "false",
)

function parse_cli(args)
    opts = copy(ARGS_DEFAULT)
    i = 1
    while i <= length(args)
        a = args[i]
        startswith(a, "--") || error("unexpected argument $a")
        name = a[3:end]
        haskey(opts, name) || error("unknown option $a")
        if name == "dry-run"
            opts[name] = "true"; i += 1
        else
            i + 1 <= length(args) || error("$a needs a value")
            opts[name] = args[i+1]; i += 2
        end
    end
    return opts
end

const OPTS = parse_cli(ARGS)
const PROFILE = OPTS["profile"]
PROFILE in ("cpu", "gpu") || error("--profile must be cpu or gpu")

include(joinpath(@__DIR__, "..", "src", "Screen.jl"))
using .Screen
using ExaModels, MadNLP, Printf, Dates

if PROFILE == "gpu"
    using CUDA, CUDSS, MadNLPGPU
end

function cpu_linear_solver(name)
    name == "mumps" && return MadNLP.MumpsSolver
    if name == "ma57"
        # Needs a licensed HSL_jll developed into the project, then MadNLPHSL added.
        @eval using MadNLPHSL
        return Base.invokelatest(() -> MadNLPHSL.Ma57Solver)
    end
    error("unknown --cpu-solver $name (mumps or ma57)")
end

function make_devices(profile, threads, cpu_solver)
    devs = Dict{String,Screen.Device}()
    devs["cpu"] = cpu_device("cpu", cpu_solver, threads)
    devs["cpu_gpunode"] = cpu_device("cpu_gpunode", cpu_solver, threads)
    if profile == "gpu"
        CUDA.functional() || error("CUDA is not functional on $(gethostname())")
        gpu_name = CUDA.name(CUDA.device())
        driver = string(CUDA.driver_version())
        devs["gpu"] = Screen.Device("gpu", CUDABackend(), MadNLPGPU.CUDSSSolver, 1,
                                    () -> CUDA.synchronize(), () -> GC.gc(),
                                    () -> (GC.gc(); CUDA.reclaim()), gpu_name, driver)
    end
    return devs
end

function versions_string(profile)
    v = ["julia=$(VERSION)", "ExaModels=$(pkgversion(ExaModels))", "MadNLP=$(pkgversion(MadNLP))"]
    if profile == "gpu"
        push!(v, "MadNLPGPU=$(pkgversion(MadNLPGPU))", "CUDA=$(pkgversion(CUDA))", "CUDSS=$(pkgversion(CUDSS))",
              "cuda_runtime=$(CUDA.runtime_version())")
    end
    return join(v, ';')
end

function main()
    stage, grid = OPTS["stage"], OPTS["grid"]
    task, ntasks = parse(Int, OPTS["task"]), parse(Int, OPTS["ntasks"])
    problems = isempty(OPTS["problems"]) ? PROBLEM_ORDER : Tuple(Symbol.(split(OPTS["problems"], ',')))
    groups = groups_for(PROFILE, stage, grid; problems)
    mine = shard(groups, task, ntasks)
    store = Store(joinpath(OPTS["results"], "$(stage)_$(grid)"))
    done = load_done(store)
    todo = sum(count(N -> !haskey(done, run_key(g, N)), g.Ns) for g in mine; init=0)
    @info "profile=$PROFILE stage=$stage grid=$grid task=$task/$ntasks: $(length(mine)) of $(length(groups)) groups, $todo runs left"
    flush(stderr)

    if OPTS["dry-run"] == "true"
        for g in mine
            println(rpad(group_id(g), 44), " tier=", rpad(g.tier, 8), " N=", join(g.Ns, ' '))
        end
        return 0
    end

    threads = parse(Int, OPTS["threads"])
    devs = make_devices(PROFILE, threads, cpu_linear_solver(OPTS["cpu-solver"]))
    ctx = RunContext(stage, grid, task, ntasks, parse(Int, OPTS["launch"]),
                     parse(Int, get(ENV, "SLURM_RESTART_COUNT", "0")), get(ENV, "SLURM_JOB_ID", "local"),
                     gethostname(), Sys.cpu_info()[1].model, versions_string(PROFILE))
    @info "host=$(ctx.hostname) cpu=$(ctx.cpu_model) blas_threads=$threads versions=$(ctx.versions)"
    haskey(devs, "gpu") && @info "gpu=$(devs["gpu"].gpu_name) driver=$(devs["gpu"].driver)"
    flush(stderr)

    warmed = Dict{Any,Float64}()
    max_iter, max_wall = parse(Int, OPTS["max-iter"]), parse(Float64, OPTS["max-wall"])
    for g in mine
        run_group!(store, done, g, devs[g.device], ctx, warmed; max_iter, max_wall_time=max_wall)
    end
    @info "task $task finished at $(now())"
    return 0
end

exit(main())
