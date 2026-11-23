module COPSProblems

export ODEProblemSpec, COPS_PROBLEMS, build_ode_problem, least_squares_loss, time_scale

# Continuous problems from COPS 3.0, §§4–6 and 8–14.
# https://www.mcs.anl.gov/~more/cops/cops3.pdf
# Observation tables cross-checked against https://github.com/madsuite-org/COPSBenchmark.jl.

const EndpointValue = Union{Float64,Nothing}
const Bound = Tuple{Float64,Float64}
const UNBOUNDED = (-Inf, Inf)
const NONNEGATIVE = (0.0, Inf)

"""
Continuous ODE benchmark in physical time.

- rhs(x, u, p, t) returns an nx-tuple. np counts physical parameters.
- xbounds, ubounds, and pbounds hold one (lower, upper) pair per component.
- x0 and xf fix endpoint components wherever their entries are numbers.
- guess.x(t) and guess.u(t) take physical time on the guessed horizon.
  guess.p holds the parameter starts and guess.tf holds the free-time start.
- A Mayer problem minimizes terminal_cost(xf, p, tf).
  Least-squares problems compare all states against ymeas at tmeas.
"""
struct ODEProblemSpec
    name::String
    nx::Int
    nu::Int
    np::Int
    rhs::Function
    t0::Float64
    tf::Union{Float64,Nothing}  # nothing means free final time
    x0::Vector{Union{Float64,Nothing}}
    xbounds::Vector{Bound}
    ubounds::Vector{Bound}
    pbounds::Vector{Bound}
    objective::Symbol
    tmeas::Vector{Float64}
    ymeas::Matrix{Float64}
    guess::NamedTuple
    xf::Vector{Union{Float64,Nothing}}
    terminal_cost::Union{Function,Nothing}
end

"""
    build_ode_problem(spec; add_final_time, min_duration)

Prepare the time coordinate for a shooting or collocation builder. For a
free-time spec, the caller supplies add_final_time(lower, upper, start) to
create a scalar decision variable in its model. Its return value becomes tf,
and the returned RHS already includes the time scaling. For a fixed-time
spec the builder returns the original interval and RHS.

The lower bound on tf is t0 + min_duration, strictly above t0: at tf = t0 every
defect vanishes trivially. The default is 1e-3 times the guessed duration.

physical_time(s) maps a solver coordinate back to the original time. Pass
the returned tf to terminal costs and impose the numerical entries in
spec.x0 and spec.xf as endpoint equalities in the transcription.
"""
function build_ode_problem(spec::ODEProblemSpec; add_final_time=nothing, min_duration=nothing)
    if spec.tf === nothing
        add_final_time === nothing &&
            throw(ArgumentError("free final time requires add_final_time(lower, upper, start)"))
        duration_floor = something(min_duration, 1e-3 * (spec.guess.tf - spec.t0))
        duration_floor > 0 || throw(ArgumentError("min_duration must be positive"))
        tf = add_final_time(spec.t0 + duration_floor, Inf, spec.guess.tf)
        duration = tf - spec.t0
        physical_time = s -> spec.t0 + duration * s
        rhs = (x, u, p, s) -> map(dx -> duration * dx, spec.rhs(x, u, p, physical_time(s)))
        return (; spec, interval=(0.0, 1.0), tf, rhs, physical_time)
    end
    return (; spec, interval=(spec.t0, spec.tf), tf=spec.tf, rhs=spec.rhs, physical_time=identity)
end

"""Sum of squared residuals with one predicted state row per observation time."""
function least_squares_loss(spec::ODEProblemSpec, predicted::AbstractMatrix)
    spec.objective === :least_squares ||
        throw(ArgumentError("$(spec.name) does not use a least-squares objective"))
    size(predicted) == size(spec.ymeas) ||
        throw(DimensionMismatch("expected predicted states of size $(size(spec.ymeas))"))
    return sum((predicted[i, j] - spec.ymeas[i, j])^2
               for i in axes(spec.ymeas, 1), j in axes(spec.ymeas, 2))
end

function mayer_spec(name, rhs, tf, x0, xf, xbounds, ubounds, terminal_cost, guess)
    return ODEProblemSpec(
        name, length(x0), length(ubounds), 0, rhs, 0.0, tf, EndpointValue[x0...],
        Bound[xbounds...], Bound[ubounds...], Bound[], :mayer, Float64[],
        zeros(0, length(x0)), guess, EndpointValue[xf...], terminal_cost,
    )
end

function estimation_spec(name, rhs, x0, times, observations, pguess)
    nx = length(x0)
    np = length(pguess)
    initial_guess = ntuple(i -> something(x0[i], observations[1, i]), nx)
    # Interpolate observations for a grid-independent start, using fixed x0 at t = 0.
    has_initial_observation = first(times) == 0.0
    guess_times = has_initial_observation ? times : vcat(0.0, times)
    first_later_row = has_initial_observation ? 2 : 1
    guess_states = vcat(reshape(collect(initial_guess), 1, nx), observations[first_later_row:end, :])
    function xguess(t)
        t <= first(guess_times) && return initial_guess
        i = searchsortedlast(guess_times, t)
        i == length(guess_times) && return Tuple(guess_states[i, :])
        weight = (t - guess_times[i]) / (guess_times[i + 1] - guess_times[i])
        return ntuple(j -> (1 - weight) * guess_states[i, j] + weight * guess_states[i + 1, j], nx)
    end
    guess = (x=xguess, u=t -> (), p=pguess, tf=nothing)
    return ODEProblemSpec(
        name, nx, 0, np, rhs, 0.0, last(times), EndpointValue[x0...],
        fill(UNBOUNDED, nx), Bound[], fill(NONNEGATIVE, np),
        :least_squares, times, observations, guess, EndpointValue[nothing for _ in 1:nx], nothing,
    )
end

# Goddard rocket. States (height, vertical velocity, mass), control (thrust,).
function rocket_rhs(x, u, p, t)
    height, velocity, mass = x
    thrust = u[1]
    drag = 310.0 * velocity^2 * exp(-500.0 * (height - 1.0))
    gravity = 1.0 / height^2
    return (velocity, (thrust - drag) / mass - gravity, -2.0 * thrust)
end

const ROCKET = mayer_spec(
    "rocket", rocket_rhs, nothing, (1.0, 0.0, 1.0), (nothing, nothing, 0.6),
    ((1.0, Inf), NONNEGATIVE, (0.6, 1.0)), ((0.0, 3.5),), (xf, p, tf) -> -xf[1],
    (x=t -> (1.0, t * (1.0 - t), 1.0 - 0.4 * t), u=t -> (1.75,), p=Float64[], tf=1.0),
)

# Hang glider. States (horizontal position, altitude, vx, vy), control (lift coefficient,).
function glider_rhs(x, u, p, t)
    position, _, vx, vy = x
    lift_coefficient = u[1]
    r = (position / 100.0 - 2.5)^2
    relative_vy = vy - 2.5 * (1.0 - r) * exp(-r)
    airspeed = sqrt(vx^2 + relative_vy^2)
    drag_coefficient = 0.034 + 0.069662 * lift_coefficient^2
    force_scale = (0.5 * 1.13 * 14.0 / 100.0) * airspeed
    ax = -force_scale * (lift_coefficient * relative_vy + drag_coefficient * vx)
    ay = force_scale * (lift_coefficient * vx - drag_coefficient * relative_vy) - 9.81
    return (vx, vy, ax, ay)
end

const GLIDER = mayer_spec(
    "glider", glider_rhs, nothing, (0.0, 1000.0, 13.23, -1.288), (nothing, 900.0, 13.23, -1.288),
    (NONNEGATIVE, UNBOUNDED, NONNEGATIVE, UNBOUNDED), ((0.0, 1.4),), (xf, p, tf) -> -xf[1],
    (x=t -> (13.23 * t, 1000.0 - 100.0 * t, 13.23, -1.288), u=t -> (0.7,), p=Float64[], tf=1.0),
)

# Robot arm. States (rho, theta, phi, rho_dot, theta_dot, phi_dot), controls in the same order.
function robot_rhs(x, u, p, t)
    rho, _, phi, rho_dot, theta_dot, phi_dot = x
    arm_length = 5.0
    inertia_phi = ((arm_length - rho)^3 + rho^3) / 3.0
    # The spherical-coordinate model is singular at the poles.
    inertia_theta = inertia_phi * sin(phi)^2
    return (rho_dot, theta_dot, phi_dot, u[1] / arm_length, u[2] / inertia_theta, u[3] / inertia_phi)
end

function robot_state_guess(t)
    # Smooth step from zero to 2*pi/3, with zero angular velocity at both ends.
    angle = (2pi / 3) * t^2 * (3.0 - 2.0 * t)
    angular_velocity = 4pi * t * (1.0 - t)
    return (4.5, angle, pi / 4, 0.0, angular_velocity, 0.0)
end

const ROBOT = mayer_spec(
    "robot", robot_rhs, nothing, (4.5, 0.0, pi / 4, 0.0, 0.0, 0.0),
    (4.5, 2pi / 3, pi / 4, 0.0, 0.0, 0.0),
    ((0.0, 5.0), (-pi, pi), (0.0, pi), UNBOUNDED, UNBOUNDED, UNBOUNDED),
    ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)), (xf, p, tf) -> tf,
    (x=robot_state_guess, u=t -> (0.0, 0.0, 0.0), p=Float64[], tf=1.0),
)

# Particle steering. States (horizontal position, altitude, vx, vy), control (angle,).
steering_rhs(x, u, p, t) = (x[3], x[4], 100.0 * cos(u[1]), 100.0 * sin(u[1]))

const STEERING = mayer_spec(
    "steering", steering_rhs, nothing, (0.0, 0.0, 0.0, 0.0), (nothing, 5.0, 45.0, 0.0),
    (UNBOUNDED, UNBOUNDED, UNBOUNDED, UNBOUNDED), ((-pi / 2, pi / 2),), (xf, p, tf) -> tf,
    (x=t -> (0.0, 5.0 * t, 45.0 * t, 0.0), u=t -> (0.0,), p=Float64[], tf=1.0),
)

# Hanging chain. States (height, accumulated potential energy, accumulated length).
# Enforce total length through xf[3].
function chain_rhs(x, u, p, t)
    slope = u[1]
    length_density = sqrt(1.0 + slope^2)
    return (slope, x[1] * length_density, length_density)
end

const CHAIN = mayer_spec(
    "chain", chain_rhs, 1.0, (1.0, 0.0, 0.0), (3.0, nothing, 4.0),
    (UNBOUNDED, UNBOUNDED, UNBOUNDED), (UNBOUNDED,), (xf, p, tf) -> xf[2],
    (x=t -> (1.0 + 4.0 * t * (t - 0.5), 0.0, 4.0 * t),
     u=t -> (8.0 * t - 2.0,), p=Float64[], tf=nothing),
)

# Catalyst mixing. States are the two concentrations, control is the mixing fraction.
function catmix_rhs(x, u, p, t)
    fraction = u[1]
    transfer = fraction * (10.0 * x[2] - x[1])
    return (transfer, -transfer - (1.0 - fraction) * x[2])
end

const CATMIX = mayer_spec(
    "catmix", catmix_rhs, 1.0, (1.0, 0.0), (nothing, nothing),
    (UNBOUNDED, UNBOUNDED), ((0.0, 1.0),), (xf, p, tf) -> xf[1] + xf[2] - 1.0,
    (x=t -> (1.0, 0.0), u=t -> (0.0,), p=Float64[], tf=nothing),
)

function gasoil_rhs(x, u, p, t)
    k1, k2, k3 = p
    reactant_squared = x[1]^2
    return (-(k1 + k3) * reactant_squared, k1 * reactant_squared - k2 * x[2])
end

function methanol_rhs(x, u, p, t)
    k1, k2, k3, k4, k5 = p
    y1, y2, _ = x
    # The benchmark's rational kinetics require a nonzero denominator.
    denominator = (k2 + k5) * y1 + y2
    rate = k1 * y1 / denominator
    return (-(2.0 * k2 + k3 + k4) * y1 + rate * y2,
            rate * (k2 * y1 - y2) + k3 * y1,
            rate * (y2 + k5 * y1) + k4 * y1)
end

function pinene_rhs(x, u, p, t)
    k1, k2, k3, k4, k5 = p
    return (-(k1 + k2) * x[1],
            k1 * x[1],
            k2 * x[1] - (k3 + k4) * x[3] + k5 * x[5],
            k3 * x[3],
            k4 * x[3] - k5 * x[5])
end

# Marine parameters (mortality rates m1:m8, growth rates g1:g7).
function marine_rhs(x, u, p, t)
    middle = ntuple(Val(6)) do i
        stage = i + 1
        p[8 + i] * x[i] - (p[stage] + p[8 + stage]) * x[stage]
    end
    return (-(p[1] + p[9]) * x[1], middle..., p[15] * x[7] - p[8] * x[8])
end

const GASOIL_OBSERVATIONS = Float64[
    1.0000  0;
    0.8105  0.2000;
    0.6208  0.2886;
    0.5258  0.3010;
    0.4345  0.3215;
    0.3903  0.3123;
    0.3342  0.2716;
    0.3034  0.2551;
    0.2735  0.2258;
    0.2405  0.1959;
    0.2283  0.1789;
    0.2071  0.1457;
    0.1669  0.1198;
    0.1530  0.0909;
    0.1339  0.0719;
    0.1265  0.0561;
    0.1200  0.0460;
    0.0990  0.0280;
    0.0870  0.0190;
    0.0770  0.0140;
    0.0690  0.0100;
]

const METHANOL_OBSERVATIONS = Float64[
    1.0000  0  0;
    0.7085  0.1621  0.0811;
    0.5971  0.1855  0.0965;
    0.5537  0.1989  0.1198;
    0.3684  0.2845  0.1535;
    0.1712  0.3491  0.2097;
    0.1198  0.3098  0.2628;
    0.0747  0.3576  0.2467;
    0.0529  0.3347  0.2884;
    0.0415  0.3388  0.2757;
    0.0261  0.3557  0.3167;
    0.0208  0.3483  0.2954;
    0.0085  0.3836  0.2950;
    0.0053  0.3611  0.2937;
    0.0019  0.3609  0.2831;
    0.0018  0.3485  0.2846;
    0.0006  0.3698  0.2899;
]

const PINENE_OBSERVATIONS = Float64[
    88.35  7.3  2.3  0.4  1.75;
    76.4  15.6  4.5  0.7  2.8;
    65.1  23.1  5.3  1.1  5.8;
    50.4  32.9  6.0  1.5  9.3;
    37.5  42.7  6.0  1.9  12.0;
    25.9  49.1  5.9  2.2  17.0;
    14.0  57.4  5.1  2.6  21.0;
    4.5  63.1  3.8  2.9  25.7;
]

const MARINE_OBSERVATIONS = Float64[
    20000.0  17000.0  10000.0  15000.0  12000.0  9000.0  7000.0  3000.0;
    12445.0  15411.0  13040.0  13338.0  13484.0  8426.0  6615.0  4022.0;
    7705.0  13074.0  14623.0  11976.0  12453.0  9272.0  6891.0  5020.0;
    4664.0  8579.0  12434.0  12603.0  11738.0  9710.0  6821.0  5722.0;
    2977.0  7053.0  11219.0  11340.0  13665.0  8534.0  6242.0  5695.0;
    1769.0  5054.0  10065.0  11232.0  12112.0  9600.0  6647.0  7034.0;
    943.0  3907.0  9473.0  10334.0  11115.0  8826.0  6842.0  7348.0;
    581.0  2624.0  7421.0  10297.0  12427.0  8747.0  7199.0  7684.0;
    355.0  1744.0  5369.0  7748.0  10057.0  8698.0  6542.0  7410.0;
    223.0  1272.0  4713.0  6869.0  9564.0  8766.0  6810.0  6961.0;
    137.0  821.0  3451.0  6050.0  8671.0  8291.0  6827.0  7525.0;
    87.0  577.0  2649.0  5454.0  8430.0  7411.0  6423.0  8388.0;
    49.0  337.0  2058.0  4115.0  7435.0  7627.0  6268.0  7189.0;
    32.0  228.0  1440.0  3790.0  6474.0  6658.0  5859.0  7467.0;
    17.0  168.0  1178.0  3087.0  6524.0  5880.0  5562.0  7144.0;
    11.0  99.0  919.0  2596.0  5360.0  5762.0  4480.0  7256.0;
    7.0  65.0  647.0  1873.0  4556.0  5058.0  4944.0  7538.0;
    4.0  44.0  509.0  1571.0  4009.0  4527.0  4233.0  6649.0;
    2.0  27.0  345.0  1227.0  3677.0  4229.0  3805.0  6378.0;
    1.0  20.0  231.0  934.0  3197.0  3695.0  3159.0  6454.0;
    1.0  12.0  198.0  707.0  2562.0  3163.0  3232.0  5566.0;
]

const GASOIL = estimation_spec(
    "gasoil", gasoil_rhs, (1.0, 0.0),
    [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.150, 0.175, 0.20, 0.225, 0.250,
     0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.75, 0.85, 0.95],
    GASOIL_OBSERVATIONS, zeros(3),
)
const METHANOL = estimation_spec(
    "methanol", methanol_rhs, (1.0, 0.0, 0.0),
    [0.0, 0.050, 0.065, 0.080, 0.123, 0.233, 0.273, 0.354, 0.397, 0.418,
     0.502, 0.553, 0.681, 0.750, 0.916, 0.937, 1.122],
    METHANOL_OBSERVATIONS, ones(5),
)
# Pinene runs to t = 36420 with rate constants near 1e-4 to 1e-5, which makes the KKT
# system badly scaled. Solve it in tau = t / PINENE_TIME_SCALE on [0, 1]. The kinetics are
# linear in k, so with scaled parameters k_tilde = PINENE_TIME_SCALE * k the right-hand side
# keeps the same form. Only the measurement times scale. Report k = k_tilde / scale.
const PINENE_TIME_SCALE = 36420.0
const PINENE = estimation_spec(
    "pinene", pinene_rhs, (100.0, 0.0, 0.0, 0.0, 0.0),
    [1230.0, 3060.0, 4920.0, 7800.0, 10680.0, 15030.0, 22620.0, 36420.0] ./ PINENE_TIME_SCALE,
    PINENE_OBSERVATIONS, zeros(5),
)
const MARINE = estimation_spec(
    "marine", marine_rhs, ntuple(_ -> nothing, 8),
    collect(0.0:0.5:10.0), MARINE_OBSERVATIONS, zeros(15),
)

const COPS_PROBLEMS = (
    rocket=ROCKET, glider=GLIDER, robot=ROBOT, steering=STEERING, chain=CHAIN, catmix=CATMIX,
    gasoil=GASOIL, methanol=METHANOL, pinene=PINENE, marine=MARINE,
)

"""
Factor between the solver's time coordinate and physical time for problems solved in
rescaled time. Physical parameters are solver parameters divided by this factor
(see PINENE_TIME_SCALE). Problems not listed use a factor of 1.
"""
const TIME_SCALES = (pinene=PINENE_TIME_SCALE,)
time_scale(name::Symbol) = get(TIME_SCALES, name, 1.0)

end # module COPSProblems
