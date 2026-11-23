# Merge sweep shards, compute errors against the reference solutions, fit convergence orders.
#
#   julia --project=. scripts/collect.jl [--grid full] [--results dir]   (defaults to results/ in this folder)
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
#
# If the reference stage has no converged run for a problem (marine ran out of memory, glider hit
# the wall-time cap and robot hung in the full screen), the reference comes from the sweep itself:
# among consecutive pairs (N, 2N) of one exact-config CPU ladder that both converged, the pair that
# agrees best. The finer run is the reference and the pair's difference is the floor, which
# overestimates the finer run's own error when the ladder converges. ref_key names the run used.

include(joinpath(@__DIR__, "..", "src", "Screen.jl"))
using .Screen
using Printf, Statistics, LinearAlgebra

function cli(args)
    o = Dict("grid" => "full", "results" => normpath(joinpath(@__DIR__, "..", "results")))
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

"Reference data of a finished run (states sampled from its solution file), or nothing."
function pack(r, root, spec)
    sol = read_solution(joinpath(root, "solutions", r["key"] * ".csv"))
    sol === nothing && return nothing
    return (; key=r["key"], obj=num(r, "objective"), p=parse_floats(r["p_hat"]), tf=num(r, "tf"),
            free_tf=spec.tf === nothing, X=sol.X)
end

const NO_FLOOR = (err_obj=NaN, err_p=NaN, err_tf=NaN, err_x_mesh=NaN, err_x_unif=NaN)

function load_reference(refrows, refroot, problem)
    spec = COPS_PROBLEMS[Symbol(problem)]
    rs = sort([r for r in values(refrows) if r["problem"] == problem && converged(r)]; by=r -> parse(Int, r["N"]))
    isempty(rs) && return nothing
    fine = pack(rs[end], refroot, spec)
    fine === nothing && return nothing
    coarse = length(rs) >= 2 ? pack(rs[end-1], refroot, spec) : nothing
    floor = coarse === nothing ? NO_FLOOR : errors(coarse.obj, coarse.p, coarse.tf, coarse.X, fine)
    return (; ref=fine, floor)
end

"Fallback reference from the sweep's exact-config CPU ladders (see the header)."
function sweep_reference(rows, sweeproot, problem)
    spec = COPS_PROBLEMS[Symbol(problem)]
    ladders = Dict{Tuple{String,String},Vector{Dict{String,String}}}()
    for r in values(rows)
        r["problem"] == problem && r["config"] == "exact" && r["device"] == "cpu" && converged(r) || continue
        push!(get!(ladders, (r["family"], r["K"]), Dict{String,String}[]), r)
    end
    packed = Dict{String,Any}()
    getpack(r) = get!(() -> pack(r, sweeproot, spec), packed, r["key"])
    best = nothing
    for rs in values(ladders)
        sort!(rs; by=r -> parse(Int, r["N"]))
        for (a, b) in zip(rs[1:end-1], rs[2:end])
            parse(Int, b["N"]) == 2 * parse(Int, a["N"]) || continue
            pa, pb = getpack(a), getpack(b)
            (pa === nothing || pb === nothing) && continue
            floor = errors(pa.obj, pa.p, pa.tf, pa.X, pb)
            diffs = filter(isfinite, collect(values(floor)))
            isempty(diffs) && continue
            score = (maximum(diffs), -parse(Int, b["N"]))   # best agreement, then the finer pair
            if best === nothing || score < best.score
                best = (; score, ref=pb, floor)
            end
        end
    end
    return best === nothing ? nothing : (; ref=best.ref, floor=best.floor)
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
    refs = Dict{Symbol,Any}(p => load_reference(refrows, refroot, string(p)) for p in PROBLEM_ORDER)
    for p in PROBLEM_ORDER
        refs[p] === nothing || continue
        refs[p] = sweep_reference(rows, sweeproot, string(p))
        if refs[p] === nothing
            @warn "no converged reference for $p: error columns stay empty"
        else
            @warn @sprintf("no converged reference run for %s: using sweep run %s (floor err_x_mesh %.1e, err_obj %.1e)",
                           p, refs[p].ref.key, refs[p].floor.err_x_mesh, refs[p].floor.err_obj)
        end
    end

    out_cols = vcat(COLUMNS, ERR_COLS, ["below_ref_floor", "ref_key", "ref_floor_x_mesh", "ref_floor_obj"])
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
                r["ref_floor_x_mesh"] = repr(f)
                r["ref_floor_obj"] = repr(ref.floor.err_obj)
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
