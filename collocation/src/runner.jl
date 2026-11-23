# Run one group's N ladder: resume, prune, warm up, solve, record.

const SAMPLE_M = 200          # states sampled at s = m / 200
const MESH_M = 5              # shared mesh points s = 0, 0.2, ..., 1 (every N in the ladder)
const SATURATION_TOL = 1e-10  # exact config: stop the ladder after two consecutive changes below this

kkt_label(cfg::SolverConfig) = string(nameof(cfg.kkt_system))
solver_label(dev::Device) = string(nameof(dev.linear_solver))

function base_row(g::Group, N, cfg::SolverConfig, dev::Device, ctx::RunContext)
    spec = COPS_PROBLEMS[g.problem]
    return Dict{String,String}(
        "key" => run_key(g, N), "stage" => g.stage, "grid" => ctx.grid, "problem" => string(g.problem),
        "family" => g.family, "K" => string(g.K), "N" => string(N), "device" => g.device,
        "config" => g.config, "tier" => g.tier, "linear_solver" => solver_label(dev),
        "kkt_system" => kkt_label(cfg), "tol" => string(cfg.tol),
        "block_size" => string((g.K + 1) * spec.nx),
        "hostname" => ctx.hostname, "cpu_model" => ctx.cpu_model, "blas_threads" => string(dev.blas_threads),
        "gpu_name" => dev.gpu_name, "driver" => dev.driver, "versions" => ctx.versions,
        "slurm_job_id" => ctx.job_id, "array_task" => string(ctx.task),
        "restart_count" => string(ctx.restart_count), "launch" => string(ctx.launch),
        "timestamp" => string(now()),
    )
end

"Pruning state of a ladder, rebuilt from finished rows on resume."
mutable struct LadderState
    prune::String                          # "", "SKIPPED_COST" or "SATURATED"
    last::Union{Nothing,Dict{String,String}}
    saturated_steps::Int
end

function relative_change(a::Dict{String,String}, b::Dict{String,String})
    rel(x, y) = abs(x - y) / (1 + abs(y))
    d = rel(parse(Float64, a["objective"]), parse(Float64, b["objective"]))
    for key in ("p_hat", "x_mesh")
        va, vb = parse_floats(a[key]), parse_floats(b[key])
        length(va) == length(vb) && !isempty(va) && (d = max(d, maximum(rel.(va, vb))))
    end
    return d
end

function update!(state::LadderState, row::Dict{String,String}, cfg::SolverConfig)
    status = row["status"]
    if status in COST_STATUSES
        state.prune = "SKIPPED_COST"
    elseif cfg.name == "exact" && status in CONVERGED_STATUSES
        if state.last !== nothing
            change = relative_change(row, state.last)
            state.saturated_steps = change < SATURATION_TOL ? state.saturated_steps + 1 : 0
            state.saturated_steps >= 2 && (state.prune = "SATURATED")
        end
        state.last = row
    end
    return state
end

# Job logs are files, where Julia buffers stderr; flush so a killed job still shows its progress.
flushlogs() = (flush(stderr); flush(stdout); nothing)

is_oom(e) = e isa OutOfMemoryError || occursin("OutOfGPUMemory", string(typeof(e))) ||
            occursin("out of memory", lowercase(sprint(showerror, e)))

"""
    run_group!(store, done, g, dev, ctx, warmed; max_iter, max_wall_time)

Run the N ladder of `g` in ascending order, skipping keys already in `done`.
`warmed` maps (problem, config, device) to the untimed warmup time this process paid.
"""
function run_group!(st::Store, done::Dict, g::Group, dev::Device, ctx::RunContext, warmed::Dict;
                    max_iter=3000, max_wall_time=1800.0)
    spec = COPS_PROBLEMS[g.problem]
    cfg = CONFIGS[g.config]
    family = scheme_family(g.family)
    path = shard_path(st, g.device, g.config, ctx.task)
    state = LadderState("", nothing, 0)
    scale = time_scale(g.problem)

    for N in g.Ns
        key = run_key(g, N)
        if haskey(done, key)
            update!(state, done[key], cfg)
            continue
        end
        row = base_row(g, N, cfg, dev, ctx)
        if !isempty(state.prune)
            row["status"] = state.prune
            append_row(path, row); done[key] = row
            continue
        end
        claim = claim!(st, key, ctx)
        if claim !== :run
            row["status"] = claim === :crashed ? "CRASHED" : "KILLED"
            row["note"] = "previous attempt never finished"
            append_row(path, row); done[key] = row; release!(st, key)
            update!(state, row, cfg)
            @info "$(row["status"]) $key"
            flushlogs()
            continue
        end

        # Untimed warmup on the smallest grid point: compile cost depends on the problem,
        # config and device, not on K or N (scheme coefficients are data).
        wkey = (g.problem, g.config, g.device)
        if !haskey(warmed, wkey)
            t0 = time_ns()
            try
                solve_point(spec, g.problem, family, g.K, 5, cfg, dev; max_iter=200, max_wall_time)
            catch e
                @warn "warmup failed for $wkey" exception = (e, catch_backtrace())
            end
            warmed[wkey] = (time_ns() - t0) / 1e9
        end
        row["t_compile"] = string(warmed[wkey])

        try
            r = solve_point(spec, g.problem, family, g.K, N, cfg, dev; max_iter, max_wall_time)
            mesh = sample_states(r.layout, r.scheme, r.sol, MESH_M)
            p_phys = r.sol.p ./ scale
            tf_phys = r.layout.free_tf ? r.sol.tf : spec.tf * scale
            merge!(row, Dict(
                "status" => r.status, "iters" => string(r.iters), "objective" => repr(r.objective),
                "tf" => repr(tf_phys), "p_hat" => join_floats(p_phys),
                "x_mesh" => join_floats(vec(permutedims(mesh))),
                "primal_inf" => repr(r.primal_inf), "dual_inf" => repr(r.dual_inf),
                "nvar" => string(r.nvar), "ncon" => string(r.ncon), "nnzj" => string(r.nnzj),
                "nnzh" => string(r.nnzh), "nnz_kkt" => string(r.nnz_kkt), "kkt_dim" => string(r.kkt_dim),
                "t_build" => repr(r.t_build), "t_first" => repr(r.t_first), "t_sgm" => repr(r.t_sgm),
                "n_reruns" => string(r.n_reruns), "timestamp" => string(now()),
            ))
            S = sample_states(r.layout, r.scheme, r.sol, SAMPLE_M)
            write_solution(st, key, ["key" => key, "objective" => repr(r.objective), "tf" => repr(tf_phys),
                                     "p_hat" => join_floats(p_phys), "status" => r.status],
                           [m / SAMPLE_M for m in 0:SAMPLE_M], S)
        catch e
            e isa InterruptException && rethrow()
            row["status"] = is_oom(e) ? "ERROR_OOM" : "ERROR"
            row["note"] = first(sprint(showerror, e), 300)
            @warn "run failed: $key" exception = (e, catch_backtrace())
        end
        append_row(path, row)
        done[key] = row
        release!(st, key)
        update!(state, row, cfg)
        @info @sprintf("%-48s %-28s it=%-4s t_first=%-9s t_sgm=%s", key, row["status"], get(row, "iters", ""),
                       get(row, "t_first", ""), get(row, "t_sgm", ""))
        flushlogs()
    end
    return nothing
end
