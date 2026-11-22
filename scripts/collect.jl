# Merge sweep shards, compute errors against the reference solutions, fit convergence orders.
#
#   julia --project=. scripts/collect.jl [--grid full] [--results results]
#
# Writes results/sweep_<grid>.csv (every run, with error columns) and results/rates_<grid>.csv.
#
# Errors, all compared in normalized time s in [0, 1] (free-time problems differ in tf):
#   err_obj    |J - J_ref| / max(1, |J_ref|)
#   err_p      ||p - p_ref|| / ||p_ref||                       (estimation problems)
#   err_tf     |tf - tf_ref| / max(1, |tf_ref|)                (free-time problems)
#   err_x_mesh max over states and s in {0, 0.2, ..., 1} of |x - x_ref| / scale_state
#   err_x_unif same over s = m / 200
# where scale_state = max(1, max_s |x_ref|). Controls are never compared (bang-bang problems).
# The reference uncertainty is the same metric between the two reference resolutions; errors
# below 10x that are flagged below_ref_floor = 1.

include(joinpath(@__DIR__, "..", "src", "Screen.jl"))
using .Screen
using Printf, Statistics, LinearAlgebra

function cli(args)
    o = Dict("grid" => "full", "results" => "results")
    for i in 1:2:length(args)
        o[args[i][3:end]] = args[i+1]
    end
    return o
end

function rows_under(root)
    rows = Dict{String,Dict{String,String}}()
    isdir(root) || return rows
    for d in readdir(root; join=true)
        isdir(d) && !(basename(d) in ("solutions", "inflight")) || continue
        for f in readdir(d; join=true)
            startswith(basename(f), "shard_") || continue
            for r in read_shard(f)
                rows[r["key"]] = r   # a later duplicate (resharded rerun) wins
            end
        end
    end
    return rows
end

"Sampled states and metadata from a solution file, or nothing."
function read_solution(path)
    isfile(path) || return nothing
    meta = Dict{String,String}()
    data = Vector{Vector{Float64}}()
    for line in eachline(path)
        if startswith(line, "# ")
            k, v = split(line[3:end], '='; limit=2)
            meta[k] = v
        elseif startswith(line, "s,")
            continue
        else
            push!(data, parse.(Float64, split(line, ',')))
        end
    end
    M = reduce(vcat, permutedims.(data))
    return (; meta, s=M[:, 1], X=M[:, 2:end])
end

converged(r) = r["status"] in ("SOLVE_SUCCEEDED", "SOLVED_TO_ACCEPTABLE_LEVEL")
num(r, k) = (v = get(r, k, ""); isempty(v) ? NaN : parse(Float64, v))

const MESH_IDX = [1 + 40m for m in 0:5]   # s = 0, 0.2, ..., 1 within s = m / 200

function errors(obj, p, tf, X, ref)
    scale = max.(1.0, vec(maximum(abs.(ref.X); dims=1)))
    D = abs.(X .- ref.X) ./ permutedims(scale)
    return (
        err_obj=abs(obj - ref.obj) / max(1, abs(ref.obj)),
        err_p=isempty(ref.p) ? NaN : norm(p .- ref.p) / max(norm(ref.p), eps()),
        err_tf=ref.free_tf ? abs(tf - ref.tf) / max(1, abs(ref.tf)) : NaN,
        err_x_mesh=maximum(D[MESH_IDX, :]),
        err_x_unif=maximum(D),
    )
end

function load_reference(refrows, refroot, problem)
    rs = sort([r for r in values(refrows) if r["problem"] == problem && converged(r)]; by=r -> parse(Int, r["N"]))
    isempty(rs) && return nothing
    function pack(r)
        sol = read_solution(joinpath(refroot, "solutions", r["key"] * ".csv"))
        spec = COPS_PROBLEMS[Symbol(problem)]
        return (; key=r["key"], obj=num(r, "objective"), p=parse_floats(r["p_hat"]), tf=num(r, "tf"),
                free_tf=spec.tf === nothing, X=sol.X)
    end
    fine = pack(rs[end])
    floor = if length(rs) >= 2
        coarse = pack(rs[end-1])
        errors(coarse.obj, coarse.p, coarse.tf, coarse.X, fine)
    else
        (err_obj=NaN, err_p=NaN, err_tf=NaN, err_x_mesh=NaN, err_x_unif=NaN)
    end
    return (; ref=fine, floor)
end

const ERR_COLS = ["err_obj", "err_p", "err_tf", "err_x_mesh", "err_x_unif"]

function main()
    o = cli(ARGS)
    grid, results = o["grid"], o["results"]
    sweeproot = joinpath(results, "sweep_$grid")
    refroot = joinpath(results, "reference_$grid")
    rows = rows_under(sweeproot)
    refrows = rows_under(refroot)
    @info "$(length(rows)) sweep rows, $(length(refrows)) reference rows"
    refs = Dict(p => load_reference(refrows, refroot, string(p)) for p in PROBLEM_ORDER)
    for (p, r) in refs
        r === nothing && @warn "no converged reference for $p: error columns stay empty"
    end

    out_cols = vcat(COLUMNS, ERR_COLS, ["below_ref_floor", "ref_key"])
    out = Dict{String,String}[]
    for key in sort(collect(keys(rows)))
        r = copy(rows[key])
        ref = refs[Symbol(r["problem"])]
        if ref !== nothing && converged(r)
            sol = read_solution(joinpath(sweeproot, "solutions", key * ".csv"))
            if sol !== nothing
                e = errors(num(r, "objective"), parse_floats(r["p_hat"]), num(r, "tf"), sol.X, ref.ref)
                for c in ERR_COLS
                    r[c] = repr(getfield(e, Symbol(c)))
                end
                f = ref.floor.err_x_mesh
                r["below_ref_floor"] = isfinite(f) && e.err_x_mesh < 10f ? "1" : "0"
                r["ref_key"] = ref.ref.key
            end
        end
        push!(out, r)
    end
    path = joinpath(results, "sweep_$grid.csv")
    open(path, "w") do io
        println(io, join(out_cols, ','))
        for r in out
            println(io, join((get(r, c, "") for c in out_cols), ','))
        end
    end
    @info "wrote $path ($(length(out)) rows)"

    # Observed orders between consecutive N (h halves): log2(e_N / e_2N), using converged runs
    # whose error is above 10x the reference floor. Reported per metric as the median of the
    # pairwise orders and the order of the finest valid pair.
    groups = Dict{NTuple{5,String},Vector{Dict{String,String}}}()
    for r in out
        haskey(r, "err_x_mesh") || continue
        push!(get!(groups, (r["problem"], r["family"], r["K"], r["config"], r["device"]), Dict{String,String}[]), r)
    end
    rpath = joinpath(results, "rates_$grid.csv")
    open(rpath, "w") do io
        println(io, "problem,family,K,config,device,metric,n_pairs,median_order,finest_order,N_range")
        for (gk, rs) in sort(collect(groups); by=first)
            sort!(rs; by=r -> parse(Int, r["N"]))
            floor = refs[Symbol(gk[1])].floor
            for metric in ("err_x_mesh", "err_x_unif", "err_p", "err_obj")
                fl = getfield(floor, Symbol(metric))
                orders = Float64[]; Ns = Int[]
                for (a, b) in zip(rs[1:end-1], rs[2:end])
                    ea, eb = num(a, metric), num(b, metric)
                    parse(Int, b["N"]) == 2 * parse(Int, a["N"]) || continue
                    isfinite(ea) && isfinite(eb) && ea > 0 && eb > 0 || continue
                    isfinite(fl) && eb < 10fl && continue
                    push!(orders, log2(ea / eb)); push!(Ns, parse(Int, b["N"]))
                end
                isempty(orders) && continue
                @printf(io, "%s,%s,%s,%s,%s,%s,%d,%.3f,%.3f,%d-%d\n", gk..., metric, length(orders),
                        median(orders), orders[end], minimum(Ns) ÷ 2, maximum(Ns))
            end
        end
    end
    @info "wrote $rpath"
end

main()
