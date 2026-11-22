# 18.335 project: collocation screen on COPS ODE problems

Handoff brief for a coding agent. Written 21 Nov 2026.

## Project in one paragraph

18.335 (numerical linear algebra) final project, 5-person team. We compare CPU and GPU methods for ODE-constrained nonlinear optimization, extending Choi & Shin, "Revisiting Simultaneous Methods for Dynamic Optimization in the GPU Era" (arXiv 2607.12201; benchmark repo github.com/mit-shin-group/exa-petab-focapo-2027). Software stack: ExaModels.jl (modeling, runs on GPU) + MadNLP.jl (interior point), CPU vs GPU (MadNLPGPU / cuDSS). The theory side analyzes the condensed KKT system in terms of collocation parameters $(N, K, n_x)$. The experimental side, which this brief covers, re-discretizes the COPS ODE problems ourselves so we can vary the number of mesh intervals $N$ and the collocation degree $K$ and measure accuracy and solve cost.

## What exists (two files in this folder)

**`collocation_schemes.jl`** (module `CollocationSchemes`). Builds Lagrange collocation data on the reference interval $[0,1]$ for degree $K$: `nodes` (left endpoint + $K$ collocation points), `D` ($K \times (K+1)$ differentiation matrix, row $j$ = derivative at `nodes[j+1]`, w.r.t. the reference coordinate), `endpoint` (weights to evaluate at 1), and barycentric weights. `interpolation_weights(scheme, tau)` gives weights at any $\tau \in [0,1]$. Families: `:radau`, `:legendre`, `:lobatto`, `:uniform`. Verified: nodes are correct, $D$ is exact on degree-$K$ polynomials (to about 1e-11 at $K=15$), and the interpolation weights are exact.

**`cops_problems.jl`** (module `COPSProblems`). `ODEProblemSpec` struct plus 10 problems in `COPS_PROBLEMS`: rocket, glider, robot, steering, chain, catmix (optimal control, Mayer objectives) and gasoil, methanol, pinene, marine (parameter estimation, least squares on all states at `tmeas`). `rhs(x, u, p, t)` returns an `nx`-tuple. `x0`/`xf` entries that are numbers are endpoint equalities, and `nothing` means free. `tf === nothing` means free final time. `build_ode_problem(spec; add_final_time)` rescales free-time problems to $s \in [0,1]$ and returns a time-scaled RHS. Integral objectives (chain, catmix) were already rewritten as terminal costs on extra states, so no quadrature weights are needed. The RHS functions were checked against the COPS 3.0 formulations. COPS flowchan (a 4th-order BVP) is intentionally not included.

## What needs to be built

1. **Transcription builder** `build_collocation_model(spec, scheme, N; backend)` returning an ExaModel.
   - Variables: states $x_{s,i,j}$ for state $s$, interval $i = 1..N$, node $j = 0..K$; controls $u_{c,i,j}$ at the $K$ collocation nodes only; parameters $p$; $t_f$ if free.
   - Defects: $\sum_{k} D_{jk}\, x_{s,i,k} = h\, f_s(x_{\cdot,i,j}, u_{\cdot,i,j}, p, t_{i,j})$ for $j = 1..K$, where $h$ is the interval width in the solver time coordinate.
   - Continuity: $x_{s,i+1,0} = \sum_k \text{endpoint}_k\, x_{s,i,k}$.
   - Endpoint equalities from numeric entries of `x0` (at $x_{\cdot,1,0}$) and `xf` (at the end of interval $N$, via `endpoint` weights; introduce explicit $x_f$ variables).
   - Bounds on states at all nodes, controls, parameters.
   - Least squares: for each measurement time find interval $i$ and reference coordinate $\tau$, get weights from `interpolation_weights`. Introduce prediction variables $\hat y_{m,s} = \sum_k w_{m,k}\, x_{s,i_m,k}$ as equality constraints and use objective $\sum (\hat y_{m,s} - y_{m,s})^2$. This keeps each generator expression simple.
   - Mayer: `terminal_cost(xf, p, tf)` on the explicit $x_f$ variables.
   - **ExaModels/GPU rules:** pass scheme coefficients ($D_{jk}$, endpoint and interpolation weights) through the generator iterators as data, not as arrays captured in closures. Build multi-term sums with `constraint!` (augmenting existing constraints) rather than giant single expressions. Confirm that capturing the free-time `duration` expression inside the RHS closure works on the GPU backend.
2. **Initial guess**: evaluate `spec.guess.x`, `guess.u` at the physical time of every node (`physical_time`), plus `guess.p`, `guess.tf`.
3. **Solver driver**: solve with MadNLP on CPU and on GPU. Record status, objective, iteration count, wall time (separate model build, first solve and steady-state if feasible), and NLP size (variables, constraints, KKT nonzeros).
4. **Reference solutions**: for each problem solve once at large $N$, high $K$ (e.g. Radau), tight tolerance, and save. Error metrics: estimation problems, relative error in $\hat p$; control problems, objective error and state error at a common set of times. Do not use controls as an error metric for bang-bang problems (rocket, robot likely), since they won't converge pointwise.
5. **Sweep**: loop over problems × family (`:radau`, `:legendre`, others optional) × $K$ (e.g. 1..8) × $N$ (geometric, e.g. 10..1600) × device. Write one CSV row per run. Make it resumable and runnable as a SLURM array job.

## Fixes needed in the existing files

- **`:lobatto` is not Lobatto IIIA.** It enforces the ODE only at the $K$ non-left nodes, so don't expect Lobatto superconvergence. Report its order as measured, or drop it from the main comparison.
- **Free final time lower bound**: `add_final_time(spec.t0, Inf, ...)` allows $t_f = t_0$, which makes the defects trivial. Use a strictly positive lower bound.
- **Pinene scaling**: horizon 36420 with rate constants around 1e-4 to 1e-5. Consider rescaling time to $[0,1]$ (and parameters accordingly) for KKT conditioning, and note it in results.

## Validation order

1. gasoil (smooth): check that error vs $h$ shows the expected rates. With $K$ collocation points, Gauss–Legendre should give about $O(h^{2K})$ and Radau about $O(h^{2K-1})$ at mesh points, for small enough $h$.
2. rocket (bang-bang control): expect low order regardless of $K$. This is the contrast case.
3. Then the rest.

Sanity check at $K=4$ Legendre: results should be close to the original COPS discretization (COPS uses 4 Gauss–Legendre points in a monomial basis) for the same $N$.

## What the results should show

- Error vs number of NLP variables ($\approx N(K+1)n_x$ + controls) and vs wall time, per family, not just error vs $K$ at fixed $N$.
- CPU vs GPU time as $N$ and $K$ grow. Higher $K$ means fewer but denser diagonal KKT blocks (size about $(K+1)n_x$), which affects fill-in and how cuDSS compares to the CPU solver.

## Environment notes

Runs on MIT Engaging (SLURM). Check available GPU partitions with `sinfo` before writing job scripts, and don't assume a GPU type. Julia packages: ExaModels, MadNLP, MadNLPGPU, CUDA (check current APIs, since these packages move quickly). Keep CPU and GPU runs on comparable nodes and record the hardware in the CSV.
