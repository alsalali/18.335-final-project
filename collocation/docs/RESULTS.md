# Results of the CPU sweep rerun with MA57

Status of 24 Nov 2026. Numbers are from `analysis/tables/`, written by `analysis/compare_solvers.py` from
`results/sweep_full.csv` (MUMPS) and `results/ma57/sweep_full.csv` (MA57). Same 6,080 CPU keys in both (3,680 exact,
2,400 lifted). Runs are complete: every key is present, with no duplicates. MA57 uses MadNLPHSL's default options; the Choi & Shin repo sets none in `benchmark/options.jl`, the only file of theirs we
read, so treat that match as likely rather than confirmed.

## Does MA57 solve more?
Done means converged, or cut as `SATURATED` (already at roundoff). `lost_pruned` keys were skipped because an earlier run on
the same ladder failed.

| config | MUMPS done | MA57 done | gained (MA57 only) | lost directly | lost by pruning |
|---|---|---|---|---|---|
| exact, 3,680 | 3,329 | 3,369 | 61 | 17 | 4 |
| lifted, 2,400 | 2,176 | 2,181 | 68 | 33 | 30 |

So MA57 ends up about even overall, with large swings per problem (`tables/coverage.csv`): gains in steering lifted (+54),
marine exact (+33) and robot exact (+16), losses in chain lifted (29), robot lifted (19) and glider exact (11).
Which keys differ is in `tables/only_one_done.csv`.

## MA57 crashes and hangs
23 keys ended in a crash or error (22 `CRASHED`, 1 `ERROR`), and MUMPS converged on 17 of them. From the job logs: 16 runs
were killed after more than 45 min without output (a hang inside one solve), and 6 were segfaults inside libHSL (`ma57od_`).
The `ERROR` is direct evidence of an integer overflow: MadNLPHSL's workspace-growth step asked for 2.25e9 entries, above the
Int32 limit of 2.15e9. Details in `tables/ma57_crashes.csv`.
- Cause of the hangs, diagnosed on one point (`chain_legendre_K2_N5120_lifted`): as the solve proceeds the factorization
  grows to about 30× its predicted size, and MadNLPHSL answers MA57's out-of-space codes by growing the workspace 5% and
  starting the factorization over, so one step can take many restarts and the wall-time check never runs.
- On that one point, `ma57_automatic_scaling=true` solved it (12 iterations, 6.6 s, no restarts); a 2× growth factor did
  not. It is untested on the other failing points.
- MA57 exceeded its 1,800 s wall-time limit before it was checked in some runs, up to 2,721 s.

## Speed
Pairs where both solvers converged. Exact uses `t_first`, lifted uses `t_sgm`. Ratio is MUMPS time / MA57 time.

| config | filter | n | geomean | median | MA57 faster on |
|---|---|---|---|---|---|
| exact | both over 1 s | 471 | 1.29× | 1.03× | 52% |
| lifted | both over 1 s | 568 | 1.55× | 1.13× | 60% |
| exact | nvar ≥ 20,000 | 374 | 1.43× | 1.23× | 62% |
| lifted | nvar ≥ 20,000 | 598 | 1.45× | 1.15× | 62% |

(`tables/speed.csv` also lists a `BIASED_mumps_over_1s` row: filtering on MUMPS time alone inflates the ratio, to 3.1× exact and
2.2× lifted. Do not quote that.)

Caveats: the two sweeps ran on different nodes, and no exact pair shares a CPU model. The time to build the model is the
same work for both solvers; it is 1.25 to 1.4× faster on the MUMPS sweep's nodes for the same keys (`build_ratio` 0.73 to
0.80), so the nodes MA57 ran on were probably slower and these ratios probably understate MA57. Build time is mostly
single-threaded, so it is a rough control only.

Same-node check (`tables/same_node_check.csv`, one EPYC 9474F node): MA57 beat MUMPS on 11 of 12 points. Nine were 1.2 to
3.5× faster and two were far apart (`steering_legendre_K10_N160_lifted` 49 s vs 0.5 s, `marine_legendre_K7_N40_exact` 25 s
vs 0.15 s); MUMPS was 1.3× faster on `catmix_legendre_K8_N10240_exact`. The points were picked from MUMPS runs of 5 to 120 s,
which favours MA57.

MA27 was not swept. The partial check B (same log file, Xeon Platinum 8580 node) ran it on a few points MUMPS had trouble with:
on `robot_legendre_K4_N1280_exact` MA27 converged in 6.4 s where MA57 ran into its wall-time limit (2,074 s) and MUMPS had too;
on `methanol_legendre_K1_N5120_exact`, one of the 23 crashed MA57 keys, MA27 converged in 8.4 s. Check B was killed by the GPU
shutdown after those, so nothing else in it is known.

## CPU vs GPU (lifted, `t_sgm`)
Keys where the GPU, MUMPS and MA57 all converged (2,107). Ratio is CPU time / GPU time (H200).

| nvar | n | MUMPS | MA57 | best CPU of the two |
|---|---|---|---|---|
| under 1,000 | 677 | 0.14× | 0.10× | 0.09× |
| 1,000 to 10,000 | 655 | 1.24× | 0.78× | 0.69× |
| 10,000 to 100,000 | 554 | 5.85× | 2.96× | 2.61× |
| over 100,000 | 221 | 6.68× | 7.11× | 5.44× |

The choice of CPU solver changes the GPU's advantage by about 2× between 10⁴ and 10⁵ variables, and little above 10⁵.
CPU nodes were mixed models here too. On the 2,400 lifted keys the GPU converged on 2,262, MUMPS CPU on 2,176 and MA57 CPU
on 2,181. The CPU/GPU ratio uses whichever CPU node ran each key, so it is not a same-node comparison. The `cpu_gpunode`
runs (same node as the GPU) were MUMPS only and were not rerun.

## Accuracy
Where MA57 and MUMPS both converged (exact config, radau and legendre, problems with their own reference: gasoil, rocket,
methanol, pinene, steering, chain, catmix), MA57's mesh-point state error matches MUMPS's within 10% on 1,238 of 1,256
keys (`tables/accuracy.csv`).
99 pairs (both converged) have objectives that differ by more than 1e-3 relatively: glider 57, chain 35, steering 6, rocket
1, mostly lobatto_nodes (43) and uniform (22) families. A likely reason is that the problems have several local solutions,
but this was not checked case by case. List in `tables/objective_disagreements.csv`. Never compare errors across them.
For glider, the two tables use different fallback references (see `results/README.md`).

## Open decisions
1. MA57 automatic scaling: departs from the defaults used so far, and has only been tested on one point. Test it on the 17
   keys MUMPS solved and MA57 crashed or errored on, plus a sample of others, before any full rerun.
2. MA27 as a fallback for points MA57 fails (it fixed two of the points tried above).
3. A common glider reference across the two tables.
4. Whether to rerun `cpu_gpunode` with MA57, so that it matches the MA57 CPU sweep.
5. Whether timing should come from one CPU model. All CPU timings here are from a mix of node types.
