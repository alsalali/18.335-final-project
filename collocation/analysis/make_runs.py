"""Write analysis/out/runs.csv, the one dataset behind the analysis page, from results/sweep_full.csv
(scripts/collect.jl's output), and print the headline numbers as a cross-check of the page's calculations.

    python3 analysis/make_runs.py [path/to/sweep_full.csv]     (defaults to results/sweep_full.csv in this folder)

Standard library only (the cluster's python3 has no numpy or pandas).
"""
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "sweep_full.csv")
OUT = os.path.join(HERE, "out", "runs.csv")

TEXT = ["problem", "family", "K", "N", "device", "config", "status", "iters", "nvar", "nnz_kkt", "block_size",
        "n_reruns", "below_ref_floor", "ref_key", "cpu_model", "gpu_name"]
NUM = ["t_first", "t_sgm", "err_obj", "err_p", "err_tf", "err_x_mesh", "err_x_unif", "ref_floor_x_mesh", "ref_floor_obj"]
CONVERGED = ("SOLVE_SUCCEEDED", "SOLVED_TO_ACCEPTABLE_LEVEL")
FLOOR_FACTOR = 10   # a series at its accuracy floor counts pairs only while the error is 10x above its minimum


def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def fmt(v):
    x = num(v)
    return "" if x is None else "%.4g" % x


def main():
    with open(SRC) as f:
        rows = list(csv.DictReader(f))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TEXT + NUM)
        for r in rows:
            w.writerow([r.get(c, "") for c in TEXT] + [fmt(r.get(c)) for c in NUM])
    print("wrote %s (%d rows, %.0f KB)" % (OUT, len(rows), os.path.getsize(OUT) / 1024))

    # Cross-check 1: observed mesh-point order per series (exact config, CPU), with the page's rule. A series has
    # reached its accuracy floor if its smallest error comes before its largest converged N, or if the ladder
    # was cut as SATURATED. Then only pairs up to that N count, with the finer error above 10x the minimum.
    # Pairs whose finer error is below 10x the reference's own uncertainty never count.
    series = defaultdict(list)
    saturated = set()
    for r in rows:
        if r["config"] != "exact" or r["device"] != "cpu":
            continue
        key = (r["problem"], r["family"], int(r["K"]))
        if r["status"] == "SATURATED":
            saturated.add(key)
        if r["status"] in CONVERGED and (num(r["err_x_mesh"]) or 0) > 0:
            series[key].append((int(r["N"]), num(r["err_x_mesh"]), r["below_ref_floor"]))
    orders = {}
    for key, pts in series.items():
        pts.sort()
        nmin, emin = min(pts, key=lambda p: p[1])[:2]
        at_floor = nmin < pts[-1][0] or key in saturated
        ords = [math.log2(a[1] / b[1]) for a, b in zip(pts, pts[1:])
                if b[0] == 2 * a[0] and b[2] != "1"
                and (not at_floor or (b[0] <= nmin and b[1] > FLOOR_FACTOR * emin))]
        if ords:
            orders[key] = statistics.median(ords)
    for fam, expect in (("legendre", lambda K: 2 * K), ("radau", lambda K: 2 * K - 1)):
        print("\n%s observed order (expected %s)" % (fam, "2K" if fam == "legendre" else "2K-1"))
        for p in ("marine", "gasoil", "methanol", "pinene", "chain", "steering", "glider", "catmix", "rocket", "robot"):
            print("  %-9s" % p + "".join("%6s" % ("%.1f" % orders[(p, fam, K)] if (p, fam, K) in orders else "-")
                                        for K in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12)))

    # Cross-check 2: same-node CPU vs GPU, lifted config.
    lifted = {(r["problem"], r["family"], r["K"], r["N"], r["device"]): r for r in rows if r["config"] == "lifted"}
    by_n = defaultdict(list)
    per_iter = defaultdict(list)
    for (p, fam, K, N, dev), c in lifted.items():
        g = lifted.get((p, fam, K, N, "gpu"))
        if dev != "cpu_gpunode" or g is None or c["status"] not in CONVERGED or g["status"] not in CONVERGED:
            continue
        tc, tg = num(c["t_sgm"]), num(g["t_sgm"])
        by_n[int(N)].append(tc / tg)
        per_iter[int(N)].append((tc / int(c["iters"])) / (tg / int(g["iters"])))
    print("\nN       median speedup  median per-iteration speedup  (same node, K = 2, 4, 8, both families)")
    for N in sorted(by_n):
        print("%-7d %8.2f %14.2f   n=%d" % (N, statistics.median(by_n[N]), statistics.median(per_iter[N]), len(by_n[N])))


if __name__ == "__main__":
    main()
