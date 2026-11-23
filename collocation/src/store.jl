# Resumable result storage: one CSV shard per (device, config, task), in-flight sentinels,
# and one solution file per run.
#
# Layout under results/<stage>_<grid>/:
#   <device>_<config>/shard_<task>.csv   one row per finished (or skipped) run
#   solutions/<key>.csv                  sampled states, written to a tmp file and renamed
#   inflight/<key>.txt                   "restart_count launch interruptions" while a run is live

const COLUMNS = [
    "key", "stage", "grid", "problem", "family", "K", "N", "device", "config", "tier",
    "linear_solver", "kkt_system", "tol", "status", "iters", "objective", "tf", "p_hat", "x_mesh",
    "primal_inf", "dual_inf", "nvar", "ncon", "nnzj", "nnzh", "nnz_kkt", "kkt_dim", "block_size",
    "t_compile", "t_build", "t_first", "t_sgm", "n_reruns",
    "hostname", "cpu_model", "blas_threads", "gpu_name", "driver", "versions",
    "slurm_job_id", "array_task", "restart_count", "launch", "timestamp", "note",
]

struct Store
    root::String
end

shard_dir(st::Store, device, config) = joinpath(st.root, "$(device)_$(config)")
shard_path(st::Store, device, config, task) = joinpath(shard_dir(st, device, config), @sprintf("shard_%03d.csv", task))
solution_path(st::Store, key) = joinpath(st.root, "solutions", key * ".csv")
inflight_path(st::Store, key) = joinpath(st.root, "inflight", key * ".txt")

clean_field(v) = replace(string(v), ',' => ' ', '\n' => ' ', '\r' => ' ')
join_floats(v) = join((repr(Float64(x)) for x in v), ';')
parse_floats(s::AbstractString) = isempty(s) ? Float64[] : parse.(Float64, split(s, ';'))

"Rows of one shard file; lines with the wrong field count (a write cut off by a kill) are dropped."
function read_shard(path)
    rows = Dict{String,String}[]
    isfile(path) || return rows
    lines = readlines(path)
    isempty(lines) && return rows
    header = split(lines[1], ',')
    for line in lines[2:end]
        fields = split(line, ',')
        length(fields) == length(header) || continue
        push!(rows, Dict(String(h) => String(f) for (h, f) in zip(header, fields)))
    end
    return rows
end

"Every finished key under the store, from all shard files, so re-sharding is safe."
function load_done(st::Store)
    done = Dict{String,Dict{String,String}}()
    isdir(st.root) || return done
    for d in readdir(st.root; join=true)
        isdir(d) || continue
        basename(d) in ("solutions", "inflight") && continue
        for f in readdir(d; join=true)
            startswith(basename(f), "shard_") && endswith(f, ".csv") || continue
            for row in read_shard(f)
                done[row["key"]] = row
            end
        end
    end
    return done
end

function fsync(io::IO)
    flush(io)
    ccall(:fsync, Cint, (Cint,), Base.Libc.fd(io))
    return nothing
end

function append_row(path, row::Dict{String,String})
    mkpath(dirname(path))
    new = !isfile(path) || filesize(path) == 0
    # A kill mid-write can leave a line without its newline; start a fresh line so the
    # next row is not glued to the broken one.
    needs_newline = !new && open(io -> (seekend(io); skip(io, -1); read(io, UInt8) != UInt8('\n')), path)
    open(path, "a") do io
        new && println(io, join(COLUMNS, ','))
        needs_newline && println(io)
        println(io, join((clean_field(get(row, c, "")) for c in COLUMNS), ','))
        fsync(io)
    end
end

function write_solution(st::Store, key, meta::Vector{Pair{String,String}}, svals, S::AbstractMatrix)
    path = solution_path(st, key)
    mkpath(dirname(path))
    tmp = path * ".tmp"
    open(tmp, "w") do io
        for (k, v) in meta
            println(io, "# ", k, "=", v)
        end
        println(io, join(["s"; ["x$(j)" for j in axes(S, 2)]], ','))
        for (r, s) in enumerate(svals)
            println(io, join([repr(s); [repr(S[r, j]) for j in axes(S, 2)]], ','))
        end
        fsync(io)
    end
    mv(tmp, path; force=true)
end

"""
Run context: who is running, so a leftover sentinel can be classified.
"""
struct RunContext
    stage::String
    grid::String
    task::Int
    ntasks::Int
    launch::Int
    restart_count::Int
    job_id::String
    hostname::String
    cpu_model::String
    versions::String
end

"""
    claim!(store, key, ctx) -> :run | :crashed | :killed

Write the in-flight sentinel for `key`. A leftover sentinel means the previous attempt never
wrote its row:
- same allocation, earlier Julia launch: Julia died on this run (segfault, OOM), so :crashed;
- earlier allocation (preemption or time limit): retry, unless it was interrupted 3 times (:killed).
"""
function claim!(st::Store, key, ctx::RunContext)
    path = inflight_path(st, key)
    interruptions = 0
    if isfile(path)
        f = split(strip(read(path, String)))
        rc, launch, prev = length(f) == 3 ? parse.(Int, f) : (-1, -1, 0)
        interruptions = prev + 1
        rc == ctx.restart_count && launch < ctx.launch && return :crashed
        interruptions >= 3 && return :killed
    end
    mkpath(dirname(path))
    write(path, "$(ctx.restart_count) $(ctx.launch) $(interruptions)\n")
    return :run
end

release!(st::Store, key) = rm(inflight_path(st, key); force=true)
