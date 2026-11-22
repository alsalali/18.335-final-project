# The K x N screen: tiers, job profiles, run keys, and deterministic sharding.

const PROBLEM_ORDER = (:gasoil, :rocket, :methanol, :pinene, :marine, :glider, :robot, :steering, :chain, :catmix)
const N_LADDER = [5 * 2^j for j in 0:11]          # 5, 10, ..., 10240: shared mesh points at s = 0, 0.2, ..., 1
const N_LADDER_TIER3 = filter(<=(640), N_LADDER)
const MAX_NVAR = 2_000_000                        # guard only; the full grid peaks near 1.07M
const CORE_K = 1:8
const HIGH_K = (10, 12)
const HEADLINE_K = (2, 4, 8)
const MAIN_FAMILIES = ("radau", "legendre")
const OTHER_FAMILIES = ("lobatto_nodes", "uniform")

# Problems with bang-bang or singular arcs. Their references use low K on a fine mesh.
# Revise after the first look at the solutions.
const NONSMOOTH = (:rocket, :robot, :catmix)

scheme_family(label::AbstractString) = label == "lobatto_nodes" ? :lobatto : Symbol(label)

"One (problem, family, K, config, device) ladder; its N values run in ascending order in one process."
struct Group
    stage::String
    problem::Symbol
    family::String
    K::Int
    config::String
    device::String
    tier::String
    Ns::Vector{Int}
end

group_id(g::Group) = "$(g.problem)_$(g.family)_K$(g.K)_$(g.config)_$(g.device)"
run_key(g::Group, N) = "$(g.problem)_$(g.family)_K$(g.K)_N$(N)_$(g.config)_$(g.device)"

function _ladder(problem, K, Ns)
    spec = COPS_PROBLEMS[problem]
    keep = [N for N in Ns if estimate_nvar(spec, K, N) <= MAX_NVAR]
    length(keep) < length(Ns) && @warn "size guard drops N > $(maximum(keep; init=0)) for $problem K=$K"
    return keep
end

function _add!(out, stage, problems, families, Ks, configs, device, tier, Ns)
    for problem in problems, family in families, K in Ks, config in configs
        push!(out, Group(stage, problem, family, K, config, device, tier, _ladder(problem, K, Ns)))
    end
    return out
end

function reference_plan(problem::Symbol, grid::String)
    if problem in NONSMOOTH
        return ("radau", 4, grid == "full" ? [10240, 20480] : [320, 640])
    else
        return ("legendre", 12, grid == "full" ? [640, 1280] : [40, 80])
    end
end

"""
    groups_for(profile, stage, grid; problems=PROBLEM_ORDER)

Every group a job profile runs. `profile` is "cpu" or "gpu", `stage` is "sweep" or
"reference", `grid` is "full", "smoke" or "pilot".
"""
function groups_for(profile::String, stage::String, grid::String; problems=PROBLEM_ORDER)
    out = Group[]
    if stage == "reference"
        profile == "cpu" || throw(ArgumentError("references run in the cpu profile"))
        for problem in problems
            family, K, Ns = reference_plan(problem, grid)
            push!(out, Group(stage, problem, family, K, "reference", "cpu", "ref", Ns))
        end
        return out
    end
    stage == "sweep" || throw(ArgumentError("unknown stage $stage"))

    if grid == "smoke"
        ps = [p for p in (:gasoil, :rocket) if p in problems]
        Ns = [5, 10, 20, 40]
        if profile == "cpu"
            _add!(out, stage, ps, MAIN_FAMILIES, (1, 2, 4), ("exact", "lifted"), "cpu", "smoke", Ns)
        else
            _add!(out, stage, ps, MAIN_FAMILIES, (1, 2, 4), ("lifted",), "gpu", "smoke", Ns)
            _add!(out, stage, ps, MAIN_FAMILIES, (2, 4), ("lifted",), "cpu_gpunode", "smoke", Ns)
        end
    elseif grid == "pilot"
        # Largest points, to calibrate time limits and memory.
        ps = [p for p in (:marine,) if p in problems]
        dev = profile == "cpu" ? ("cpu",) : ("gpu", "cpu_gpunode")
        for d in dev
            _add!(out, stage, ps, ("radau",), (8, 12), ("lifted",), d, "pilot", [10240])
        end
    elseif grid == "full"
        if profile == "cpu"
            _add!(out, stage, problems, MAIN_FAMILIES, CORE_K, ("exact", "lifted"), "cpu", "1", N_LADDER)
            _add!(out, stage, problems, MAIN_FAMILIES, HIGH_K, ("exact", "lifted"), "cpu", "2", N_LADDER)
            _add!(out, stage, problems, OTHER_FAMILIES, CORE_K, ("exact",), "cpu", "3", N_LADDER_TIER3)
        else
            _add!(out, stage, problems, MAIN_FAMILIES, CORE_K, ("lifted",), "gpu", "1", N_LADDER)
            _add!(out, stage, problems, MAIN_FAMILIES, HEADLINE_K, ("lifted",), "cpu_gpunode", "headline", N_LADDER)
            _add!(out, stage, problems, MAIN_FAMILIES, HIGH_K, ("lifted",), "gpu", "2", N_LADDER)
        end
    else
        throw(ArgumentError("unknown grid $grid"))
    end
    return out
end

# Rough relative cost of a group, only used to balance shards.
function group_cost(g::Group)
    spec = COPS_PROBLEMS[g.problem]
    factor = g.config == "lifted" ? 3.0 : g.config == "reference" ? 2.0 : 1.5
    g.device == "gpu" && (factor *= 0.3)
    return factor * sum(Float64(estimate_nvar(spec, g.K, N))^1.2 for N in g.Ns; init=0.0)
end

const TIER_ORDER = Dict("1" => 1, "smoke" => 1, "pilot" => 1, "ref" => 1, "headline" => 2, "2" => 3, "3" => 4)

"""
    shard(groups, task, ntasks)

Deterministic greedy longest-processing-time split of groups over tasks, so a requeued task
gets the same groups. Each task then runs tier 1 first, gasoil and rocket first.
"""
function shard(groups::Vector{Group}, task::Int, ntasks::Int)
    0 <= task < ntasks || throw(ArgumentError("task $task not in 0:$(ntasks-1)"))
    order = sort(collect(eachindex(groups)); by=i -> (-group_cost(groups[i]), group_id(groups[i])))
    load = zeros(ntasks)
    owner = zeros(Int, length(groups))
    for i in order
        t = argmin(load) - 1
        owner[i] = t
        load[t+1] += group_cost(groups[i])
    end
    mine = [groups[i] for i in eachindex(groups) if owner[i] == task]
    problem_rank(p) = something(findfirst(==(p), PROBLEM_ORDER), 99)
    sort!(mine; by=g -> (TIER_ORDER[g.tier], problem_rank(g.problem), g.family, g.K, g.config, g.device))
    return mine
end
