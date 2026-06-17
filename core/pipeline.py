from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import TowerCandidate, generate_candidates
from .data import load_cities
from .matrices import build_coverage_matrix, build_interference_pairs
from .optimizer import OptimizationResult, solve
from .radius_search import RadiusPlan, plan_tower_radii
from .spatial import ClusterProfile, cluster_cities, describe_clusters


@dataclass(frozen=True)
class Result:
    cities: pd.DataFrame
    labels: np.ndarray
    cluster_profile: ClusterProfile
    radius_plan: RadiusPlan
    candidates: list[TowerCandidate]
    coverage_matrix: np.ndarray
    interference_pairs: list[tuple[int, int]]
    interference_penalties: np.ndarray
    optimization: OptimizationResult


def run(
    cities_file: str | Path | None = None,
    *,
    eps_km: float = 35.0,
    min_samples: int = 2,
    alpha: float = 0.5,
    beta: float = 0.5,
    time_limit: float | None = 60.0,
) -> Result:
    """Execute the full two-phase pipeline from data loading to optimization."""

    cities = load_cities(cities_file)
    labels = cluster_cities(cities, eps_km=eps_km, min_samples=min_samples) # Clustering Alogorithm
    cluster_profile = describe_clusters(labels)
    radius_plan = plan_tower_radii(cities, labels) # Finding the optimal radius for both tower types
    candidates = generate_candidates(cities, labels, radius_plan) # Create J; It gives dict of towers placeed at each location with centroind

    coverage_matrix = build_coverage_matrix(cities, candidates) # If two cities have inference it will be identified via this function
    keep_mask = coverage_matrix.sum(axis=1) > 0
    if not np.all(keep_mask):
        candidates = [candidate for candidate, keep in zip(candidates, keep_mask) if keep]
        coverage_matrix = build_coverage_matrix(cities, candidates)

    interference_pairs, interference_penalties = build_interference_pairs(candidates) # Penaties
    optimization = solve(  # Actual optimiser.
        candidates,
        coverage_matrix,
        interference_pairs,
        interference_penalties,
        alpha=alpha,
        beta=beta,
        time_limit=time_limit,
    )

    return Result(
        cities=cities,
        labels=labels,
        cluster_profile=cluster_profile,
        radius_plan=radius_plan,
        candidates=candidates,
        coverage_matrix=coverage_matrix,
        interference_pairs=interference_pairs,
        interference_penalties=interference_penalties,
        optimization=optimization,
    )