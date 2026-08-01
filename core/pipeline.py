from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import TowerCandidate, generate_candidates
from .data import load_cities
from .matrices import build_coverage_matrix, build_interference_pairs
from .optimizer import OptimizationResult, solve
from .radius_pair_search import RadiusPairEvaluation, optimize_radius_pair_with_milp
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
    radius_pair_evaluations: list[RadiusPairEvaluation] | None = None


def run(
    cities_file: str | Path | None = None,
    *,
    eps_km: float = 35.0,
    min_samples: int = 2,
    alpha: float = 0.5,
    beta: float = 0.5,
    radius_mode: str = "milp_pair_search",
    radius_milp_top_k: int = 8,
    radius_step: float = 5.0,
    radius_time_limit: float | None = 20.0,
    time_limit: float | None = 60.0,
    include_midpoints: bool = True,
    midpoint_max_distance_km: float = 80.0,
    include_voronoi: bool = False,
    include_meanshift: bool = False,
    include_grid: bool = False,
    grid_spacing_km: float = 50.0,
    refine_radii: bool = False,
    refine_radii_grid: bool = True,
    solver: str | None = None,
) -> Result:
    """Execute the full two-phase pipeline from data loading to optimization."""

    cities = load_cities(cities_file)
    labels = cluster_cities(cities, eps_km=eps_km, min_samples=min_samples)
    cluster_profile = describe_clusters(labels)
    radius_pair_evaluations = None

    if radius_mode == "milp_pair_search":
        radius_search = optimize_radius_pair_with_milp(
            cities,
            labels,
            alpha=alpha,
            milp_top_k=radius_milp_top_k,
            radius_step=radius_step,
            time_limit_per_pair=radius_time_limit,
            include_midpoints=include_midpoints,
            midpoint_max_distance_km=midpoint_max_distance_km,
            include_voronoi=include_voronoi,
            include_meanshift=include_meanshift,
            include_grid=include_grid,
            grid_spacing_km=grid_spacing_km,
            refine_radii=refine_radii,
            refine_radii_grid=refine_radii_grid,
            solver=solver,
        )
        radius_plan = radius_search.radius_plan
        candidates = radius_search.candidates
        coverage_matrix = radius_search.coverage_matrix
        interference_pairs = radius_search.interference_pairs
        interference_penalties = radius_search.interference_penalties
        optimization = radius_search.optimization
        radius_pair_evaluations = radius_search.evaluations
    elif radius_mode == "dbscan_representative":
        radius_plan = plan_tower_radii(cities, labels)
        candidates = generate_candidates(cities, labels, radius_plan)

        coverage_matrix = build_coverage_matrix(cities, candidates)
        keep_mask = coverage_matrix.sum(axis=1) > 0
        if not np.all(keep_mask):
            candidates = [candidate for candidate, keep in zip(candidates, keep_mask) if keep]
            coverage_matrix = build_coverage_matrix(cities, candidates)

        interference_pairs, interference_penalties = build_interference_pairs(candidates)
        optimization = solve(
            candidates,
            coverage_matrix,
            interference_pairs,
            interference_penalties,
            alpha=alpha,
            beta=beta,
            hard_coverage=True,
            require_all_tower_types=True,
            mutually_exclusive_locations=True,
            time_limit=time_limit,
            solver=solver,
        )
    else:
        raise ValueError("radius_mode must be 'milp_pair_search' or 'dbscan_representative'.")

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
        radius_pair_evaluations=radius_pair_evaluations,
    )
