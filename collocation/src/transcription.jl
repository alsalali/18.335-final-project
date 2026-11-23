# Collocation transcription of an ODEProblemSpec into an ExaModel (HANDOFF.md items 1-2).
#
# Layout, with n = 1..K+1 indexing scheme.nodes (n = 1 is the interval's left endpoint):
#   x[s, i, n]   states at every node of every interval
#   u[c, i, j]   controls at the K collocation nodes only (j = 1..K, node n = j + 1)
#   p[q]         parameters
#   tf[1]        free final time (only when spec.tf === nothing)
#   xf[s]        explicit final states, xf = sum_n endpoint[n] * x[s, N, n]
#   yhat[m, s]   predicted states at measurement times (least-squares problems only)
#
# Scheme coefficients enter as data in the generator iterators, never as arrays captured in
# closures, so the same kernels run on the GPU. Multi-term sums are built with add_con!
# augmentation: one base block per constraint family, then one augmentation that carries
# (row, indices..., coefficient) for every term.

"""
Where each variable block lives in the solution vector, plus what is needed to sample the
solution. All fields are host data, so it works the same for CPU and GPU models.
"""
struct ModelLayout
    problem::Symbol
    nx::Int
    nu::Int
    np::Int
    N::Int
    K::Int
    free_tf::Bool
    interval::Tuple{Float64,Float64}   # solver time coordinate (0, 1) when tf is free
    x_offset::Int
    u_offset::Int
    p_offset::Int
    tf_offset::Int
    xf_offset::Int
    nvar::Int
    ncon::Int
end

# Offset of a variable block in the solution vector.
block_offset(v) = v.offset

estimate_nvar(spec, K, N) = N * (K + 1) * spec.nx + N * K * spec.nu + spec.np +
                            (spec.tf === nothing ? 1 : 0) + spec.nx + length(spec.tmeas) * spec.nx

# Physical time of node n in interval i, given an interval [a, b] in the solver coordinate
# and a map from solver coordinate to physical time.
node_coordinate(a, h, scheme, i, n) = a + h * ((i - 1) + scheme.nodes[n])

"""
    build_collocation_model(spec, scheme, N; backend=nothing, problem=Symbol(spec.name))

Return `(model, layout)`: an ExaModel for the degree-K collocation transcription of `spec`
on N uniform mesh intervals, and the layout needed to read its solution.
"""
function build_collocation_model(spec::ODEProblemSpec, scheme::CollocationScheme, N::Integer;
                                 backend=nothing, problem::Symbol=Symbol(spec.name),
                                 min_duration=nothing)
    N >= 1 || throw(ArgumentError("N must be at least 1"))
    K = scheme.degree
    nx, nu, np = spec.nx, spec.nu, spec.np
    nn = K + 1
    free_tf = spec.tf === nothing
    c = ExaCore(; backend)

    # The free final time is created first, inside build_ode_problem, so the time-scaled RHS
    # closure can capture its node.
    core_ref = Ref{Any}(c)
    tf_offset = -1
    function add_final_time(lower, upper, start)
        core, v = add_var(core_ref[], 1; lvar=lower, uvar=upper, start=start)
        core_ref[] = core
        tf_offset = block_offset(v)
        return v[1]
    end
    prob = build_ode_problem(spec; add_final_time, min_duration)
    c = core_ref[]
    a, b = prob.interval
    h = (b - a) / N
    rhs = prob.rhs
    guess_tf = free_tf ? spec.guess.tf : spec.tf
    # Physical time used for the initial guess: free-time problems use the guessed tf.
    guess_time(sv) = free_tf ? spec.t0 + (guess_tf - spec.t0) * sv : sv

    # ---- variables -------------------------------------------------------------------
    xstart = zeros(nx, N, nn)
    xl = similar(xstart); xu = similar(xstart)
    for n in 1:nn, i in 1:N
        g = spec.guess.x(guess_time(node_coordinate(a, h, scheme, i, n)))
        for s in 1:nx
            xstart[s, i, n] = g[s]
            xl[s, i, n], xu[s, i, n] = spec.xbounds[s]
        end
    end
    c, x = add_var(c, nx, N, nn; start=xstart, lvar=xl, uvar=xu)
    x_offset = block_offset(x)

    u = nothing
    u_offset = -1
    if nu > 0
        ustart = zeros(nu, N, K)
        ul = similar(ustart); uu = similar(ustart)
        for j in 1:K, i in 1:N
            g = spec.guess.u(guess_time(node_coordinate(a, h, scheme, i, j + 1)))
            for q in 1:nu
                ustart[q, i, j] = g[q]
                ul[q, i, j], uu[q, i, j] = spec.ubounds[q]
            end
        end
        c, u = add_var(c, nu, N, K; start=ustart, lvar=ul, uvar=uu)
        u_offset = block_offset(u)
    end

    p = nothing
    p_offset = -1
    if np > 0
        c, p = add_var(c, np; start=collect(Float64, spec.guess.p),
                       lvar=[bd[1] for bd in spec.pbounds], uvar=[bd[2] for bd in spec.pbounds])
        p_offset = block_offset(p)
    end

    xf_guess = spec.guess.x(guess_time(b))
    c, xf = add_var(c, nx; start=[xf_guess[s] for s in 1:nx],
                    lvar=[spec.xbounds[s][1] for s in 1:nx], uvar=[spec.xbounds[s][2] for s in 1:nx])
    xf_offset = block_offset(xf)

    tf_node = free_tf ? prob.tf : spec.tf
    ptuple = np > 0 ? ntuple(q -> p[q], np) : ()

    # ---- collocation defects: sum_n D[j, n] x[s, i, n] - h f_s(x_{i, j+1}, u_{i, j}, p, t) = 0 ----
    defect_rows = vec([((i - 1) * K + j, i, j, node_coordinate(a, h, scheme, i, j + 1)) for j in 1:K, i in 1:N])
    defect_terms = vec([((i - 1) * K + j, i, n, scheme.D[j, n]) for n in 1:nn, j in 1:K, i in 1:N])
    for s in 1:nx
        # s is a Julia constant here, so `[s]` selects the component while tracing.
        c, defect = add_con(c, -h * rhs(ntuple(r -> x[r, i, j + 1], nx),
                                        nu > 0 ? ntuple(q -> u[q, i, j], nu) : (),
                                        ptuple, t)[s]
                               for (row, i, j, t) in defect_rows)
        c, _ = add_con!(c, defect, row => d * x[s, i, n] for (row, i, n, d) in defect_terms)
    end

    # ---- continuity: x[s, i+1, 1] - sum_n endpoint[n] x[s, i, n] = 0 ----------------------
    if N > 1
        cont_rows = vec([(s, i) for s in 1:nx, i in 1:N-1])
        cont_terms = vec([((i - 1) * nx + s, s, i, n, -scheme.endpoint[n]) for n in 1:nn, s in 1:nx, i in 1:N-1])
        c, cont = add_con(c, x[s, i + 1, 1] for (s, i) in cont_rows)
        c, _ = add_con!(c, cont, row => w * x[s, i, n] for (row, s, i, n, w) in cont_terms)
    end

    # ---- final states: xf[s] - sum_n endpoint[n] x[s, N, n] = 0 -------------------------
    c, xfcon = add_con(c, xf[s] for s in 1:nx)
    final_terms = vec([(s, n, -scheme.endpoint[n]) for n in 1:nn, s in 1:nx])
    c, _ = add_con!(c, xfcon, s => w * x[s, N, n] for (s, n, w) in final_terms)

    # ---- endpoint equalities --------------------------------------------------------------
    fixed0 = [(s, Float64(v)) for (s, v) in enumerate(spec.x0) if v !== nothing]
    if !isempty(fixed0)
        c, _ = add_con(c, x[s, 1, 1] for (s, v) in fixed0;
                       lcon=[v for (_, v) in fixed0], ucon=[v for (_, v) in fixed0])
    end
    fixedf = [(s, Float64(v)) for (s, v) in enumerate(spec.xf) if v !== nothing]
    if !isempty(fixedf)
        c, _ = add_con(c, xf[s] for (s, v) in fixedf;
                       lcon=[v for (_, v) in fixedf], ucon=[v for (_, v) in fixedf])
    end

    # ---- objective --------------------------------------------------------------------------
    if spec.objective === :least_squares
        nm = length(spec.tmeas)
        ystart = zeros(nm, nx)
        meas_terms = Tuple{Int,Int,Int,Int,Float64}[]
        for m in 1:nm
            i, tau = locate(spec.tmeas[m], a, h, N)
            w = interpolation_weights(scheme, tau)
            g = spec.guess.x(spec.tmeas[m])
            for s in 1:nx
                ystart[m, s] = g[s]
                for n in 1:nn
                    push!(meas_terms, ((s - 1) * nm + m, s, i, n, -w[n]))
                end
            end
        end
        c, yhat = add_var(c, nm, nx; start=ystart)
        pred_rows = vec([(m, s) for m in 1:nm, s in 1:nx])   # row (s - 1) * nm + m
        c, pred = add_con(c, yhat[m, s] for (m, s) in pred_rows)
        c, _ = add_con!(c, pred, row => w * x[s, i, n] for (row, s, i, n, w) in meas_terms)
        residuals = vec([(m, s, spec.ymeas[m, s]) for m in 1:nm, s in 1:nx])
        c, _ = add_obj(c, (yhat[m, s] - y)^2 for (m, s, y) in residuals)
    elseif spec.objective === :mayer
        xft = ntuple(s -> xf[s], nx)
        c, _ = add_obj(c, spec.terminal_cost(xft, ptuple, tf_node) for _ in 1:1)
    else
        throw(ArgumentError("unknown objective $(spec.objective)"))
    end

    model = ExaModel(c)
    layout = ModelLayout(problem, nx, nu, np, N, K, free_tf, (a, b), x_offset, u_offset, p_offset,
                         tf_offset, xf_offset, model.meta.nvar, model.meta.ncon)
    return model, layout
end

# Interval index and reference coordinate of solver time t on a uniform mesh.
function locate(t, a, h, N)
    pos = (t - a) / h
    i = clamp(floor(Int, pos) + 1, 1, N)
    tau = clamp(pos - (i - 1), 0.0, 1.0)
    return i, tau
end

"""
Solution pulled back to the host: states X[s, i, n], parameters (solver units), tf.
"""
struct CollocationSolution
    X::Array{Float64,3}
    p::Vector{Float64}
    tf::Float64
end

function extract_solution(layout::ModelLayout, solution_vector)
    v = Array(solution_vector)   # copies from the device when the model is on the GPU
    nx, N, nn = layout.nx, layout.N, layout.K + 1
    X = reshape(v[layout.x_offset+1:layout.x_offset+nx*N*nn], nx, N, nn)
    pvals = layout.np > 0 ? v[layout.p_offset+1:layout.p_offset+layout.np] : Float64[]
    tf = layout.free_tf ? v[layout.tf_offset+1] : layout.interval[2]
    return CollocationSolution(copy(X), pvals, tf)
end

"""
    sample_states(layout, scheme, sol, M)

States at normalized times s = m / M, m = 0..M, as an (M+1) × nx matrix. The interval
position m*N/M is computed from integers, so shared mesh points are hit exactly.
"""
function sample_states(layout::ModelLayout, scheme::CollocationScheme, sol::CollocationSolution, M::Int)
    N, nx = layout.N, layout.nx
    out = zeros(M + 1, nx)
    for m in 0:M
        pos = (m * N) / M
        i = clamp(floor(Int, pos) + 1, 1, N)
        tau = clamp(pos - (i - 1), 0.0, 1.0)
        w = interpolation_weights(scheme, tau)
        for s in 1:nx
            out[m+1, s] = sum(w[n] * sol.X[s, i, n] for n in eachindex(w))
        end
    end
    return out
end
