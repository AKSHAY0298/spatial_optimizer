"""Uniform epsilon-constraint sweep at fixed radii.

Unlike core.pareto.compute_pareto_front, which binary-searches for a cost
transition, this traces a uniform grid of epsilon values between the minimum
achievable interference and the interference of the minimum-cost solution.
Radii are pinned so the front corresponds to the design point reported in
Chapter 5 rather than to a fresh (time-limit dependent) radius search.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.candidates import generate_candidate_locations, generate_candidates_from_locations
from core.data import load_cities
from core.matrices import build_coverage_matrix, build_interference_pairs
from core.optimizer import solve_with_lazy_interference
from core.spatial import cluster_cities


def build_problem(cities_file: Path, eps_km: float, min_samples: int, radii: tuple[float, float]):
    cities = load_cities(cities_file)
    labels = cluster_cities(cities, eps_km=eps_km, min_samples=min_samples)
    locations = generate_candidate_locations(
        cities, labels,
        include_cluster_centroids=True,
        include_midpoints=True,
        midpoint_max_distance_km=80.0,
    )
    candidates = generate_candidates_from_locations(locations, radii)
    coverage = build_coverage_matrix(cities, candidates)
    keep = coverage.sum(axis=1) > 0
    if not np.all(keep):
        candidates = [c for c, k in zip(candidates, keep) if k]
        coverage = coverage[keep]
    pairs, penalties = build_interference_pairs(candidates)
    print(f"cities={len(cities)} candidates={len(candidates)} pairs={len(pairs)}", flush=True)
    return cities, candidates, coverage, pairs, penalties


def solve(candidates, coverage, pairs, penalties, *, alpha=0.5, epsilon=None,
          time_limit=45.0, max_rounds=400):
    """Solve one point. A high *max_rounds* is essential: the epsilon constraint
    is only imposed over the lazily activated pair subset, so if the cutting-plane
    loop hits its round cap it returns a solution that violates the epsilon bound
    while still reporting the sub-solver's "Optimal" status."""
    started = time.time()
    result, _, _ = solve_with_lazy_interference(
        candidates, coverage, pairs, penalties,
        alpha=alpha, epsilon=epsilon,
        hard_coverage=True, require_all_tower_types=True,
        mutually_exclusive_locations=True, time_limit=time_limit,
        max_rounds=max_rounds,
    )
    return result, time.time() - started


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", required=True)
    ap.add_argument("--r1", type=float, required=True)
    ap.add_argument("--r2", type=float, required=True)
    ap.add_argument("--eps-km", type=float, default=40.0)
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=45.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    radii = (args.r1, args.r2)
    cities, candidates, coverage, pairs, penalties = build_problem(
        Path(args.cities), args.eps_km, args.min_samples, radii
    )

    # Anchor 1: minimum cost, interference unconstrained.
    res_cost, secs = solve(candidates, coverage, pairs, penalties,
                           epsilon=1e9, time_limit=args.time_limit * 2)
    eps_max = res_cost.total_interference
    print(f"[min-cost]  cost={res_cost.total_cost:.4f} interf={eps_max:.4f} "
          f"towers={len(res_cost.selected_candidates)} status={res_cost.message} {secs:.0f}s", flush=True)

    # The minimum-cost solution already carries very little interference, so the
    # informative part of the front lies BELOW eps_max: tightening epsilon there
    # forces cost upward. An alpha=0 anchor is deliberately not used -- with lazy
    # pair generation the objective starts identically zero, so the solver returns
    # an arbitrary feasible point rather than the true minimum-interference one.
    rows = []
    for eps in np.linspace(0.0, eps_max, args.steps + 1)[:-1]:
        try:
            res, secs = solve(candidates, coverage, pairs, penalties,
                              epsilon=float(eps), time_limit=args.time_limit)
        except RuntimeError as exc:
            print(f"[eps={eps:.4f}] infeasible/failed: {exc}", flush=True)
            rows.append(dict(label=f"eps={eps:.4f}", epsilon=float(eps), cost="",
                             interference="", towers="", coverage="", status="Infeasible",
                             respects_epsilon=""))
            continue
        feasible = res.total_interference <= eps + 1e-6
        flag = "" if feasible else "  <-- VIOLATES eps (round cap hit)"
        print(f"[eps={eps:.4f}] cost={res.total_cost:.4f} interf={res.total_interference:.4f} "
              f"towers={len(res.selected_candidates)} status={res.message} {secs:.0f}s{flag}", flush=True)
        rows.append(dict(label=f"eps={eps:.4f}", epsilon=float(eps), cost=res.total_cost,
                         interference=res.total_interference,
                         towers=len(res.selected_candidates),
                         coverage=res.coverage_ratio, status=res.message,
                         respects_epsilon=feasible))

    rows.append(dict(label="min-cost", epsilon=eps_max, cost=res_cost.total_cost,
                     interference=eps_max, towers=len(res_cost.selected_candidates),
                     coverage=res_cost.coverage_ratio, status=res_cost.message,
                     respects_epsilon=True))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
