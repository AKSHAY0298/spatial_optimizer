"""Check whether the lazy interference loop converges under an epsilon constraint.

Reports, for a given epsilon, the interference the solver was told to respect
versus the interference actually realized by the returned solution, at the
default round cap and at a much higher one.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.candidates import generate_candidate_locations, generate_candidates_from_locations
from core.data import load_cities
from core.matrices import build_coverage_matrix, build_interference_pairs
from core.optimizer import solve_with_lazy_interference
from core.spatial import cluster_cities

CITIES = Path("data/cities_fr_30k.txt")
RADII = (18.5, 34.0)
EPS = 0.0447

cities = load_cities(CITIES)
labels = cluster_cities(cities, eps_km=40.0, min_samples=2)
locations = generate_candidate_locations(
    cities, labels, include_cluster_centroids=True,
    include_midpoints=True, midpoint_max_distance_km=80.0,
)
candidates = generate_candidates_from_locations(locations, RADII)
coverage = build_coverage_matrix(cities, candidates)
keep = coverage.sum(axis=1) > 0
candidates = [c for c, k in zip(candidates, keep) if k]
coverage = coverage[keep]
pairs, penalties = build_interference_pairs(candidates)
pair_array = np.asarray(pairs, dtype=int).reshape(-1, 2)
print(f"candidates={len(candidates)} pairs={len(pairs)} epsilon={EPS}", flush=True)


def realized_interference(result) -> float:
    """Total penalty over EVERY pair whose two towers are both selected."""
    mask = np.zeros(len(candidates), dtype=bool)
    mask[result.selected_indices] = True
    both = mask[pair_array[:, 0]] & mask[pair_array[:, 1]]
    return float(penalties[both].sum())


for max_rounds in (30, 400):
    result, active_pairs, _ = solve_with_lazy_interference(
        candidates, coverage, pairs, penalties,
        epsilon=EPS, hard_coverage=True, require_all_tower_types=True,
        mutually_exclusive_locations=True, time_limit=60.0,
        max_rounds=max_rounds,
    )
    true_interf = realized_interference(result)
    print(
        f"max_rounds={max_rounds:>4d} | cost={result.total_cost:.4f} "
        f"| reported_interf={result.total_interference:.4f} "
        f"| TRUE_interf={true_interf:.4f} "
        f"| active_pairs={len(active_pairs)} "
        f"| satisfies_eps={true_interf <= EPS + 1e-6} "
        f"| status={result.message}",
        flush=True,
    )
