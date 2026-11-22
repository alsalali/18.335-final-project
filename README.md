# Collocation K × N screen on the COPS ODE problems

Background and goals are in `HANDOFF.md`. This file covers how to run the screen.

## Layout
| Path | What |
|---|---|
| `collocation_schemes.jl`, `cops_problems.jl` | Collocation coefficients and the 10 COPS problems |
| `src/Screen.jl` | Module that loads everything below |
| `src/transcription.jl` | `build_collocation_model(spec, scheme, N; backend)` → `(ExaModel, layout)` |
| `src/solvers.jl` | Solver configs `exact` / `lifted` / `reference`, devices, the timed solve |
| `src/grid.jl` | Grid tiers, job profiles, sharding |
| `src/store.jl`, `src/runner.jl` | Resumable CSV shards, in-flight sentinels, the per-ladder loop |
| `scripts/run_sweep.jl` | Driver for both jobs |
| `scripts/validate.jl` | Convergence and COPS checks; `--gpu` adds the GPU vs CPU check |
| `scripts/collect.jl` | Merges shards and computes errors vs the references → `results/sweep_<grid>.csv`, `results/rates_<grid>.csv` |
| `cluster/` | `env.sh`, `setup.sh`, `cpu.sbatch`, `gpu.sbatch`, `job_body.sh`, `submit.sh` |

## Setup (once)
```bash
bash cluster/setup.sh     # login node: instantiate the pinned Manifest, precompile (~35 min on 1 CPU)
```
`cluster/env.sh` loads `julia/1.12.6`, puts the depot on `~/orcd/pool/julia_depot` and sets
`JULIA_CPU_TARGET` so one set of package caches works on every node type.
The stack is ExaModels 0.15.0, MadNLP 0.11.0, MadNLPGPU 0.11.1, CUDA.jl 6.5.1 and CUDSS 1.0.0.
The CPU linear solver is MUMPS. MA57 needs libHSL: develop `HSL_jll` into the project, add `MadNLPHSL`,
then pass `--cpu-solver ma57`.

## Running
Submit from the repo root. If you submit from inside another Slurm job, unset the inherited `SLURM_*` variables first.
```bash
sbatch -p mit_quicktest --time=00:15:00 --array=0-1 --no-requeue cluster/cpu.sbatch sweep smoke   # CPU smoke
sbatch --time=00:30:00 --array=0 cluster/gpu.sbatch sweep smoke                                   # GPU smoke
sbatch --time=02:00:00 --array=0 cluster/gpu.sbatch sweep pilot                                   # largest points
bash cluster/submit.sh                       # references (10 tasks) + CPU sweep (32) + GPU sweep (4 GPUs)
source cluster/env.sh && julia --project=. scripts/collect.jl --grid full
```
Locally: `julia --project=. scripts/run_sweep.jl --profile cpu --grid smoke --problems gasoil`.
Add `--dry-run` to list a task's groups.

## The screen
- N = 5·2^j, j = 0…11 (5 to 10240). Every N in the ladder has mesh points at s = 0, 0.2, …, 1.
- Tier 1: radau and legendre, K = 1–8. Tier 2: K = 10, 12. Tier 3: lobatto_nodes and uniform, K = 1–8, N ≤ 640, exact config only.
- The CPU job runs `exact` (augmented KKT, tol 1e-10) for accuracy and `lifted` (condensed KKT + relaxed equalities, tol 1e-8, as in Choi & Shin) for timing.
- The GPU job runs `lifted` on the GPU with cuDSS, plus K ∈ {2, 4, 8} on the same node's CPU (`cpu_gpunode`).
- Timing: an untimed warmup per (problem, config, device) gives `t_compile`. `t_first` is the first solve. `t_sgm` is the shifted geometric mean of up to 5 reruns (δ = 0.01 s).
- References: `--stage reference` solves each problem at two fine resolutions. If neither converges
  (marine, glider and robot in the full screen), `collect.jl` uses the best-agreeing consecutive pair of
  the sweep's own `exact` CPU runs instead; `ref_key` in `sweep_<grid>.csv` names the run used.
- Pruning is always recorded in the CSV:
  - `SKIPPED_COST`: larger N skipped after a wall-time hit, OOM or crash.
  - `SATURATED`: `exact` only, larger N skipped after two consecutive changes below 1e-10.

## Requeue and resume
- Preemption: Slurm requeues the job itself.
- Time limit: `--signal=B:USR1@300` makes `job_body.sh` call `scontrol requeue`.
- Every attempt skips keys already present in any shard file and retries the run that was in flight.
- If Julia dies inside an allocation, the job relaunches it and records the run it died on as `CRASHED`.
- Hang watchdog: MadNLP checks `--max-wall` only between iterations, and one MUMPS factorization can run
  for hours (seen in the full screen). If the job log has not changed for `--max-wall` + 15 min, the job
  kills Julia, which records that run as `CRASHED` and skips the larger N of its ladder.
- A run interrupted 3 times is recorded as `KILLED`.

## Validation results (21 Nov 2026)
- gasoil with parameters fixed at the reference (pure IVP), mesh-point orders at N = 80:
  - Legendre K = 1–4: 2.0, 4.0, 5.9, 7.8 (expected 2K).
  - Radau K = 1–4: 0.96, 2.9, 4.9, 6.8 (expected 2K−1).
- gasoil as an estimation problem gives irregular orders between roughly K+1 and 2K. The measurement times fall inside mesh intervals, so the least-squares fit sees interior (stage-order) error, and τ changes with N.
- rocket: order about 1 for every K, as expected for bang-bang control.
- K = 4 Legendre objectives against COPSBenchmark.jl's expected values: within 1e-4 relative for 9 of 10 problems. Glider is 1247.99 vs 1255.05 (0.56%). Check its formulation against COPS 3.0.
