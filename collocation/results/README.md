# Results index

Git tracks this index and the four full-sweep CSV summaries listed below. The local raw stores hold the
original runs. Findings are in the [MA57 results notes](../docs/RESULTS.md).

## Raw stores (written by the jobs, never edited by hand)
| Folder | What | Made by |
|---|---|---|
| `sweep_full/` | The first full screen: CPU exact and lifted with **MUMPS**, GPU lifted with cuDSS (H200), and the MUMPS `cpu_gpunode` rows on the GPU job's node | `cluster/submit.sh` |
| `ma57/sweep_full/` | The CPU sweep again with **MA57** (default options, libHSL 2025.7.21). Same run keys as above | `cpu.sbatch ... --cpu-solver ma57 --results $PWD/results/ma57` |
| `reference_full/` | Fine-mesh references, 2 resolutions per problem. No converged run for marine, glider or robot: `collect.jl` falls back to the sweep's own exact runs for those | `cpu.sbatch reference full` |
| `sweep_pilot/`, `sweep_smoke/`, `reference_smoke/`, `ma57/sweep_smoke/` | Small test runs (the pilot times the largest points). Not results. `ma57/sweep_smoke/` was run before the HSL versions were recorded in the `versions` column | see the top-level README |

Inside a store: `<device>_<config>/shard_*.csv` are the runs, `solutions/` the sampled states (one file per run that produced
a solution), `inflight/` markers of live runs (empty when the jobs are done).

## Derived tables (rebuild any time)
| File | From | Command |
|---|---|---|
| `sweep_full.csv`, `rates_full.csv` | MUMPS + GPU, errors vs references | `julia --project=. scripts/collect.jl --grid full` |
| `ma57/sweep_full.csv`, `ma57/rates_full.csv` | **MA57 CPU** + the same GPU and `cpu_gpunode` rows | see "CPU sweep with MA57" in the README |
| `analysis/tables/*.csv` | Both tables: coverage, speed, crashes, CPU vs GPU, accuracy | `python3 analysis/compare_solvers.py` |
| `analysis/out/runs.csv` | MUMPS table, the dataset of the analysis page | `python3 analysis/make_runs.py` |
| `analysis/out/runs_ma57.csv` | MA57 table, same columns | `python3 analysis/make_runs.py results/ma57/sweep_full.csv analysis/out/runs_ma57.csv` |

## Caveats when reading the tables
- A row's `linear_solver` says MUMPS, MA57 or cuDSS. The `cpu_gpunode` rows are MUMPS in both tables.
- Error columns for marine, robot and glider use references from the sweep itself. The two tables' references give the same errors
  (to 3 digits) for marine and robot. For glider they differ (`glider_radau_K10_N160` vs `glider_radau_K7_N320`), so do not compare glider
  errors between the two tables.
- `SKIPPED_COST` rows were pruned after an earlier failure on the same ladder; `SATURATED` rows were pruned because the ladder
  had already converged to roundoff, so count those as done.
