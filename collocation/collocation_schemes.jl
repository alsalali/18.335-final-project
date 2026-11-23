module CollocationSchemes

using LinearAlgebra: SymTridiagonal, eigvals

export CollocationScheme, collocation_scheme, interpolation_weights, interpolate

"""
Lagrange collocation coefficients on the reference interval [0, 1].

For degree K, nodes has K + 1 entries: the left endpoint followed by K
collocation points. Julia column k of D corresponds to nodes[k], and
row j differentiates the polynomial at nodes[j + 1]. The derivative
is with respect to the reference coordinate. endpoint gives weights
for evaluating the polynomial at 1. Each set of coefficients can be
reused on every mesh interval.
"""
struct CollocationScheme
    family::Symbol
    degree::Int
    nodes::Vector{Float64}
    D::Matrix{Float64}
    endpoint::Vector{Float64}
    barycentric::Vector{Float64}
end

"""
    collocation_scheme(family, degree)

Construct a degree-K state interpolant, with K >= 1. Use `:radau` for
right Radau, `:legendre` for Gauss–Legendre, `:lobatto` for
Legendre–Gauss–Lobatto state nodes, or `:uniform` for equally spaced
state nodes. The ODE is enforced at the K non-left nodes in every family.

`:lobatto` is therefore not Lobatto IIIA, which also collocates at the left
endpoint with a degree-(K+1) interpolant. Do not expect Lobatto
superconvergence from it; report its order as measured.

Multiply the RHS by the mesh interval width when forming a defect
with D. Uniform nodes can be ill-conditioned at high degree.
"""
function collocation_scheme(family::Symbol, degree::Integer)
    degree >= 1 || throw(ArgumentError("degree must be at least 1"))
    K = Int(degree)
    nodes = if family === :radau
        vcat(0.0, gauss_jacobi_nodes(K - 1, 1.0, 0.0), 1.0)
    elseif family === :legendre
        vcat(0.0, gauss_jacobi_nodes(K, 0.0, 0.0))
    elseif family === :lobatto
        vcat(0.0, gauss_jacobi_nodes(K - 1, 1.0, 1.0), 1.0)
    elseif family === :uniform
        collect(range(0.0, 1.0; length=K + 1))
    else
        throw(ArgumentError("unknown family $family. Use :radau, :legendre, :lobatto, or :uniform."))
    end

    barycentric = barycentric_weights(nodes)
    D = differentiation_matrix(nodes, barycentric)
    endpoint = weights_at(nodes, barycentric, 1.0)
    return CollocationScheme(family, K, nodes, D, endpoint, barycentric)
end

# The eigenvalues of the Jacobi recurrence matrix give the roots on [-1, 1].
# Right Radau uses interior roots for the weight (1 - x), and Lobatto uses (1 - x^2).
# https://dlmf.nist.gov/18.9 and https://mathworld.wolfram.com/Golub-WelschAlgorithm.html
function gauss_jacobi_nodes(count::Int, alpha::Float64, beta::Float64)
    count == 0 && return Float64[]

    total = alpha + beta
    diagonal = zeros(count)
    if alpha != beta
        for n in 0:count-1
            s = 2.0 * n + total
            diagonal[n + 1] = (beta^2 - alpha^2) / (s * (s + 2.0))
        end
    end

    off_diagonal = Vector{Float64}(undef, count - 1)
    for n in eachindex(off_diagonal)
        s = 2.0 * n + total
        numerator = n * (n + alpha) * (n + beta) * (n + total)
        off_diagonal[n] = (2.0 / s) * sqrt(numerator / ((s - 1.0) * (s + 1.0)))
    end

    roots = eigvals(SymTridiagonal(diagonal, off_diagonal))
    return (roots .+ 1.0) ./ 2.0
end

# A common scale factor cancels out of both interpolation and differentiation.
# Compute inverse node products in log space to keep intermediate values in range.
function barycentric_weights(nodes)
    count = length(nodes)
    log_weights = zeros(count)
    for j in eachindex(nodes), k in eachindex(nodes)
        if j != k
            log_weights[j] -= log(abs(nodes[j] - nodes[k]))
        end
    end

    log_weights .-= maximum(log_weights)
    # With sorted nodes, the inverse product has count - j negative factors.
    weights = [(isodd(count - j) ? -1.0 : 1.0) * exp(log_weights[j]) for j in eachindex(nodes)]
    all(w -> isfinite(w) && !iszero(w), weights) ||
        throw(ArgumentError("barycentric weights exceed Float64 range for this node set"))
    return weights
end

function differentiation_matrix(nodes, weights)
    count = length(nodes)
    D = zeros(count - 1, count)
    for j in 2:count
        row = j - 1
        for k in 1:count
            if k != j
                D[row, k] = (weights[k] / weights[j]) / (nodes[j] - nodes[k])
            end
        end
        D[row, j] = -sum(@view D[row, :])
    end
    all(isfinite, D) || throw(ArgumentError("differentiation coefficients exceed Float64 range"))
    return D
end

# Barycentric formula, with exact handling of evaluations at nodes.
# https://doi.org/10.1137/S0036144502417715
function weights_at(nodes, barycentric, tau)
    offsets = tau .- nodes
    _, nearest = findmin(abs, offsets)
    delta = offsets[nearest]
    if iszero(delta)
        weights = zeros(eltype(offsets), length(nodes))
        weights[nearest] = one(delta)
        return weights
    end

    # Scale by the nearest offset so every offset ratio has magnitude at most one.
    ratios = barycentric .* (delta ./ offsets)
    scale = sum(ratios)
    iszero(scale) && throw(ArgumentError("barycentric denominator vanished in floating-point arithmetic"))
    return ratios ./ scale
end

"""
    interpolation_weights(scheme, tau)

Return weights for evaluating the state polynomial at the reference coordinate
tau in [0, 1]. Use the same weights for each state component.
"""
function interpolation_weights(scheme::CollocationScheme, tau::Real)
    0 <= tau <= 1 || throw(DomainError(tau, "reference coordinate must be in [0, 1]"))
    return weights_at(scheme.nodes, scheme.barycentric, tau)
end

"""
    interpolate(scheme, values, tau)

Evaluate one state component from its K + 1 values in scheme.nodes order.
"""
function interpolate(scheme::CollocationScheme, values::AbstractVector, tau::Real)
    length(values) == length(scheme.nodes) ||
        throw(DimensionMismatch("expected $(length(scheme.nodes)) nodal values"))
    weights = interpolation_weights(scheme, tau)
    return sum(weight * value for (weight, value) in zip(weights, values))
end

end # module CollocationSchemes
