from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .candidates import (
    TowerCandidate,
    TowerLocation,
    generate_candidate_locations,
    generate_candidates_from_locations,
)
from .costs import cost_critical_radii, tower_cost
from .matrices import (
    build_coverage_matrix,
    build_interference_pairs,
    normalized_overlap_penalty,
)
from .optimizer import OptimizationResult, solve, solve_with_lazy_interference
from .radius_search import RadiusPlan
from .spatial import haversine_distance_km, pairwise_haversine_km


@dataclass(frozen=True)
class RadiusPairEvaluation:
    dense_radius_km: float
    sparse_radius_km: float
    proxy_objective: float
    proxy_cost: float
    proxy_interference: float
    proxy_tower_count: int
    lp_objective: float | None = None
    milp_objective: float | None = None
    total_cost: float | None = None
    total_interference: float | None = None
    selected_tower_count: int | None = None
    coverage_ratio: float | None = None
    solver_status: str | None = None


@dataclass(frozen=True)
class RadiusPairSearchResult:
    radius_plan: RadiusPlan
    evaluations: list[RadiusPairEvaluation]
    candidates: list[TowerCandidate]
    coverage_matrix: np.ndarray
    interference_pairs: list[tuple[int, int]]
    interference_penalties: np.ndarray
    optimization: OptimizationResult


def _radius_pairs(candidate_radii: list[float]) -> list[tuple[float, float]]:
    clean_radii = sorted({float(r) for r in candidate_radii if 5.0 <= float(r) <= 100.0})
    return [
        (left, right)
        for left_index, left in enumerate(clean_radii)
        for right in clean_radii[left_index + 1 :]
    ]


def _coverage_from_distances(distances_km: np.ndarray, radius_km: float) -> np.ndarray:
    return distances_km <= float(radius_km)


def _greedy_proxy_evaluation(
    locations: list[TowerLocation],
    distances_km: np.ndarray,
    radii_km: tuple[float, float],
    *,
    alpha: float,
) -> tuple[float, float, float, int]:
    """Fast vectorized set-cover proxy used to rank radius pairs before MILP solves."""

    radius_a, radius_b = radii_km
    # coverages[t] is shape (num_locations, num_cities) boolean
    coverages = [
        _coverage_from_distances(distances_km, radius_a),
        _coverage_from_distances(distances_km, radius_b),
    ]
    costs = [tower_cost(radius_a), tower_cost(radius_b)]

    num_locations = distances_km.shape[0]
    city_count = distances_km.shape[1]
    uncovered = np.ones(city_count, dtype=bool)
    selected: list[tuple[int, int]] = []
    selected_locations: set[int] = set()
    selected_types: set[int] = set()

    # Mask for available locations (not yet selected)
    available = np.ones(num_locations, dtype=bool)

    while bool(uncovered.any()):
        best_ratio = math.inf
        best_choice: tuple[int, int, np.ndarray] | None = None

        for type_index, coverage in enumerate(coverages):
            # Vectorized: compute gains for all locations at once
            gains = np.logical_and(coverage, uncovered).sum(axis=1)
            # Zero out unavailable locations
            gains[~available] = 0

            # Find locations with positive gain
            positive_mask = gains > 0
            if not positive_mask.any():
                continue

            # Vectorized cost/gain ratio
            ratios = np.full(num_locations, math.inf)
            ratios[positive_mask] = costs[type_index] / gains[positive_mask].astype(float)

            best_loc = int(np.argmin(ratios))
            if ratios[best_loc] < best_ratio:
                best_ratio = ratios[best_loc]
                best_choice = (type_index, best_loc, coverage[best_loc])

        if best_choice is None:
            raise ValueError("No candidate can cover the remaining cities.")

        type_index, location_index, coverage_row = best_choice
        selected.append((type_index, location_index))
        selected_locations.add(location_index)
        selected_types.add(type_index)
        available[location_index] = False
        uncovered &= ~coverage_row

    # Ensure both tower types are used
    for missing_type in ({0, 1} - selected_types):
        # Pick the available location with least added interference
        best_location: int | None = None
        best_added_interference = math.inf

        selected_coords = np.array(
            [[locations[loc_idx].latitude, locations[loc_idx].longitude] for _, loc_idx in selected],
            dtype=float,
        )
        selected_type_list = [t for t, _ in selected]

        for location_index in range(num_locations):
            if not available[location_index]:
                continue
            loc = locations[location_index]
            loc_point = np.array([[loc.latitude, loc.longitude]], dtype=float)
            dists = pairwise_haversine_km(loc_point, selected_coords)[0]
            added = 0.0
            for i, d in enumerate(dists):
                r_sum = radii_km[missing_type] + radii_km[selected_type_list[i]]
                if d < r_sum:
                    added += (r_sum - d) / r_sum
            if added < best_added_interference:
                best_location = location_index
                best_added_interference = added

        if best_location is None:
            raise ValueError("Both tower types cannot be selected with exclusive locations.")
        selected.append((missing_type, best_location))
        selected_locations.add(best_location)
        selected_types.add(missing_type)
        available[best_location] = False

    total_cost = float(sum(costs[type_index] for type_index, _ in selected))

    # Quick interference estimate using haversine distances
    total_interference = 0.0
    sel_coords = np.array(
        [[locations[loc_idx].latitude, locations[loc_idx].longitude] for _, loc_idx in selected],
        dtype=float,
    )
    sel_types = [t for t, _ in selected]
    n_sel = len(selected)
    dist_matrix = pairwise_haversine_km(sel_coords)
    for i in range(n_sel):
        for j in range(i + 1, n_sel):
            d = dist_matrix[i, j]
            r_sum = radii_km[sel_types[i]] + radii_km[sel_types[j]]
            if d < r_sum:
                total_interference += (r_sum - d) / r_sum

    proxy_objective = float(alpha * total_cost + (1.0 - alpha) * total_interference)
    return proxy_objective, total_cost, float(total_interference), len(selected)


def _filter_uncovering_candidates(
    candidates: list[TowerCandidate],
    coverage_matrix: np.ndarray,
) -> tuple[list[TowerCandidate], np.ndarray]:
    keep_mask = coverage_matrix.sum(axis=1) > 0
    if np.all(keep_mask):
        return candidates, coverage_matrix
    filtered_candidates = [candidate for candidate, keep in zip(candidates, keep_mask) if keep]
    return filtered_candidates, coverage_matrix[keep_mask]


def optimize_radius_pair_with_milp(
    cities: pd.DataFrame,
    labels: np.ndarray,
    *,
    alpha: float = 0.5,
    candidate_radii: list[float] | None = None,
    radius_step: float = 5.0,
    milp_top_k: int = 8,
    time_limit_per_pair: float | None = 20.0,
    include_cluster_centroids: bool = True,
    include_midpoints: bool = True,
    midpoint_max_distance_km: float = 80.0,
    include_voronoi: bool = False,
    include_meanshift: bool = False,
    include_grid: bool = False,
    grid_spacing_km: float = 50.0,
    refine_radii: bool = True,
    refine_radii_grid: bool = True,
) -> RadiusPairSearchResult:
    """Choose the two radii by ranking all candidate pairs and solving top MILPs.

    DBSCAN labels are used only to add centroid candidate locations. They do not
    decide the radii. Midpoint candidates between neighboring cities provide
    geometrically superior placement options independent of clustering.

    Parameters
    ----------
    radius_step : float
        Step size for candidate radii in [5, 100]. Use 1.0 for integer search,
        0.5 for half-km resolution (more pairs but potentially better solution).
    include_midpoints : bool
        Add midpoints between neighboring cities as candidate tower locations.
    refine_radii : bool
        After MILP solve, shrink each tower's radius to the minimum needed to
        maintain coverage, reducing cost without losing coverage.
    refine_radii_grid : bool
        After the coarse radius-step pass, spawn a fine 0.5 km micro-grid around
        the winning radii (2 km window) and re-evaluate for higher precision.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within the closed interval [0, 1].")
    if milp_top_k < 1:
        raise ValueError("milp_top_k must be at least 1.")

    if candidate_radii is not None:
        radii = candidate_radii
    else:
        # Generate candidate radii at given step size, plus cost-critical points
        radii_set: set[float] = set()
        r = 5.0
        while r <= 100.0:
            radii_set.add(round(r, 1))
            r += radius_step
        # Inject cost-critical radii (segment boundaries, inflection points)
        for crit_r in cost_critical_radii():
            if 5.0 <= crit_r <= 100.0:
                radii_set.add(round(crit_r, 1))
        radii = sorted(radii_set)

    pairs = _radius_pairs(radii)
    if not pairs:
        raise ValueError("At least two distinct candidate radii are required.")

    locations = generate_candidate_locations(
        cities,
        labels,
        include_cluster_centroids=include_cluster_centroids,
        include_midpoints=include_midpoints,
        midpoint_max_distance_km=midpoint_max_distance_km,
        include_voronoi=include_voronoi,
        include_meanshift=include_meanshift,
        include_grid=include_grid,
        grid_spacing_km=grid_spacing_km,
    )
    location_points = np.array(
        [[location.latitude, location.longitude] for location in locations],
        dtype=float,
    )
    city_points = cities[["latitude", "longitude"]].to_numpy(dtype=float)
    distances_km = pairwise_haversine_km(location_points, city_points)

    proxy_evaluations: list[RadiusPairEvaluation] = []
    for radius_a, radius_b in pairs:
        proxy_objective, proxy_cost, proxy_interference, proxy_tower_count = _greedy_proxy_evaluation(
            locations,
            distances_km,
            (radius_a, radius_b),
            alpha=alpha,
        )
        proxy_evaluations.append(
            RadiusPairEvaluation(
                dense_radius_km=radius_a,
                sparse_radius_km=radius_b,
                proxy_objective=proxy_objective,
                proxy_cost=proxy_cost,
                proxy_interference=proxy_interference,
                proxy_tower_count=proxy_tower_count,
            )
        )

    ranked = sorted(proxy_evaluations, key=lambda item: item.proxy_objective)
    pairs_to_solve = ranked[: min(milp_top_k, len(ranked))]

    solved_evaluations: list[RadiusPairEvaluation] = []
    best_payload = None
    best_evaluation: RadiusPairEvaluation | None = None

    for proxy_eval in pairs_to_solve:
        radii_km = (proxy_eval.dense_radius_km, proxy_eval.sparse_radius_km)

        # Build shared problem data once for both LP and MILP
        candidates = generate_candidates_from_locations(locations, radii_km)
        coverage_matrix = build_coverage_matrix(cities, candidates)
        candidates, coverage_matrix = _filter_uncovering_candidates(candidates, coverage_matrix)
        interference_pairs, interference_penalties = build_interference_pairs(candidates)

        # MILP with lazily generated interference pairs — only pairs that a
        # candidate solution actually realizes ever enter the model.
        optimization, active_pairs, active_penalties = solve_with_lazy_interference(
            candidates,
            coverage_matrix,
            interference_pairs,
            interference_penalties,
            alpha=alpha,
            beta=0.0,
            hard_coverage=True,
            require_all_tower_types=True,
            mutually_exclusive_locations=True,
            time_limit=time_limit_per_pair,
        )

        # LP relaxation over the active pair subset — a valid lower bound on
        # the full MILP objective (the subset model is itself a relaxation).
        lp_result = solve(
            candidates,
            coverage_matrix,
            active_pairs,
            active_penalties,
            alpha=alpha,
            beta=0.0,
            hard_coverage=True,
            require_all_tower_types=True,
            mutually_exclusive_locations=True,
            time_limit=time_limit_per_pair,
            relaxed=True,
        )

        solved_eval = RadiusPairEvaluation(
            dense_radius_km=proxy_eval.dense_radius_km,
            sparse_radius_km=proxy_eval.sparse_radius_km,
            proxy_objective=proxy_eval.proxy_objective,
            proxy_cost=proxy_eval.proxy_cost,
            proxy_interference=proxy_eval.proxy_interference,
            proxy_tower_count=proxy_eval.proxy_tower_count,
            lp_objective=lp_result.objective_value,
            milp_objective=optimization.objective_value,
            total_cost=optimization.total_cost,
            total_interference=optimization.total_interference,
            selected_tower_count=len(optimization.selected_candidates),
            coverage_ratio=optimization.coverage_ratio,
            solver_status=optimization.message,
        )
        solved_evaluations.append(solved_eval)

        # Accept any solution that produced usable results (CBC may return
        # "Not Solved" when it finds a feasible solution but hits time limit)
        if optimization.selected_indices.size == 0:
            continue
        if best_evaluation is None or optimization.objective_value < float(best_evaluation.milp_objective):
            best_evaluation = solved_eval
            best_payload = (
                candidates,
                coverage_matrix,
                active_pairs,
                active_penalties,
                optimization,
            )

    if best_evaluation is None or best_payload is None:
        raise RuntimeError("No optimal MILP solution was found for the tested radius pairs.")

    candidates, coverage_matrix, interference_pairs, interference_penalties, optimization = best_payload

    # --- Adaptive grid refinement: fine grid around coarse winners ---
    if refine_radii_grid and best_evaluation is not None:
        fine_radius_set: set[float] = set()

        def _fine_grid_around(center: float, window: float = 2.0, step: float = 0.5) -> list[float]:
            """Generate fine-grained radii around a center value."""
            fine: list[float] = []
            r = max(5.0, center - window)
            while r <= min(100.0, center + window):
                fine.append(round(r, 1))
                r += step
            return fine

        # Build fine candidate pool around the winning radii
        fine_dense = _fine_grid_around(best_evaluation.dense_radius_km)
        fine_sparse = _fine_grid_around(best_evaluation.sparse_radius_km)
        fine_radius_set.update(fine_dense)
        fine_radius_set.update(fine_sparse)

        # Also inject cost-critical radii in the neighborhood
        for crit_r in cost_critical_radii():
            if 5.0 <= crit_r <= 100.0:
                if abs(crit_r - best_evaluation.dense_radius_km) <= 3.0 or abs(crit_r - best_evaluation.sparse_radius_km) <= 3.0:
                    fine_radius_set.add(round(crit_r, 1))

        fine_radii = sorted(fine_radius_set)
        fine_pairs = _radius_pairs(fine_radii)

        if fine_pairs:
            # Proxy-rank all fine pairs
            fine_proxy: list[RadiusPairEvaluation] = []
            for radius_a, radius_b in fine_pairs:
                obj, cost, interf, cnt = _greedy_proxy_evaluation(
                    locations, distances_km, (radius_a, radius_b), alpha=alpha,
                )
                fine_proxy.append(RadiusPairEvaluation(
                    dense_radius_km=radius_a, sparse_radius_km=radius_b,
                    proxy_objective=obj, proxy_cost=cost,
                    proxy_interference=interf, proxy_tower_count=cnt,
                ))

            fine_ranked = sorted(fine_proxy, key=lambda item: item.proxy_objective)
            fine_top_k = min(milp_top_k, len(fine_ranked))
            fine_best_eval: RadiusPairEvaluation | None = None
            fine_best_payload = None

            for fine_eval in fine_ranked[:fine_top_k]:
                r_pair = (fine_eval.dense_radius_km, fine_eval.sparse_radius_km)
                fc = generate_candidates_from_locations(locations, r_pair)
                fcov = build_coverage_matrix(cities, fc)
                fc, fcov = _filter_uncovering_candidates(fc, fcov)
                fint_pairs, fint_penalties = build_interference_pairs(fc)

                # MILP with lazily generated interference pairs
                milp_r, factive_pairs, factive_penalties = solve_with_lazy_interference(
                    fc, fcov, fint_pairs, fint_penalties, alpha=alpha,
                    beta=0.0, hard_coverage=True, require_all_tower_types=True,
                    mutually_exclusive_locations=True, time_limit=time_limit_per_pair)

                # LP relaxation over the active pair subset (valid lower bound)
                lp_r = solve(fc, fcov, factive_pairs, factive_penalties, alpha=alpha,
                             beta=0.0, hard_coverage=True, require_all_tower_types=True,
                             mutually_exclusive_locations=True, time_limit=time_limit_per_pair,
                             relaxed=True)

                fe = RadiusPairEvaluation(
                    dense_radius_km=fine_eval.dense_radius_km,
                    sparse_radius_km=fine_eval.sparse_radius_km,
                    proxy_objective=fine_eval.proxy_objective,
                    proxy_cost=fine_eval.proxy_cost,
                    proxy_interference=fine_eval.proxy_interference,
                    proxy_tower_count=fine_eval.proxy_tower_count,
                    lp_objective=lp_r.objective_value,
                    milp_objective=milp_r.objective_value,
                    total_cost=milp_r.total_cost,
                    total_interference=milp_r.total_interference,
                    selected_tower_count=len(milp_r.selected_candidates),
                    coverage_ratio=milp_r.coverage_ratio,
                    solver_status=milp_r.message,
                )
                solved_evaluations.append(fe)

                if milp_r.selected_indices.size == 0:
                    continue
                if fine_best_eval is None or milp_r.objective_value < float(fine_best_eval.milp_objective):
                    fine_best_eval = fe
                    fine_best_payload = (fc, fcov, factive_pairs, factive_penalties, milp_r)

            # If fine-grid found a better solution, use it
            if fine_best_eval is not None and fine_best_payload is not None:
                if float(fine_best_eval.milp_objective) < float(best_evaluation.milp_objective):
                    best_evaluation = fine_best_eval
                    best_payload = fine_best_payload
                    candidates, coverage_matrix, interference_pairs, interference_penalties, optimization = best_payload

    # --- Post-MILP radius refinement: shrink each selected tower to minimum needed ---
    if refine_radii and optimization.selected_candidates:
        candidates, coverage_matrix, interference_pairs, interference_penalties, optimization = (
            _refine_tower_radii(
                locations,
                cities,
                (best_evaluation.dense_radius_km, best_evaluation.sparse_radius_km),
                optimization,
                candidates,
                alpha=alpha,
                time_limit=time_limit_per_pair,
            )
        )

    radius_plan = RadiusPlan(
        dense_radius_km=best_evaluation.dense_radius_km,
        sparse_radius_km=best_evaluation.sparse_radius_km,
        dense_score=best_evaluation.proxy_objective,
        sparse_score=float(best_evaluation.milp_objective),
        dense_bounds_km=(5, 100),
        sparse_bounds_km=(5, 100),
        search_method="milp_pair_search",
        pair_objective=best_evaluation.milp_objective,
        proxy_objective=best_evaluation.proxy_objective,
        evaluated_radius_pairs=len(proxy_evaluations),
        solved_radius_pairs=len(solved_evaluations),
    )

    return RadiusPairSearchResult(
        radius_plan=radius_plan,
        evaluations=[*ranked, *solved_evaluations],
        candidates=candidates,
        coverage_matrix=coverage_matrix,
        interference_pairs=interference_pairs,
        interference_penalties=interference_penalties,
        optimization=optimization,
    )


def _refine_tower_radii(
    locations: list[TowerLocation],
    cities: pd.DataFrame,
    original_radii_km: tuple[float, float],
    optimization: OptimizationResult,
    candidates: list[TowerCandidate],
    *,
    alpha: float,
    time_limit: float | None,
) -> tuple[
    list[TowerCandidate],
    np.ndarray,
    list[tuple[int, int]],
    np.ndarray,
    OptimizationResult,
]:
    """Shrink each selected tower's radius to the minimum that still covers its cities.

    This post-processing step keeps the same tower locations but reduces radii,
    which lowers cost (since c(r) is monotonically non-decreasing) and reduces
    interference without losing coverage.
    """

    city_coords = cities[["latitude", "longitude"]].to_numpy(dtype=float)
    selected = optimization.selected_candidates

    # For each selected tower, find the max distance to any city it must cover
    # (cities that are ONLY covered by this tower — essential coverage)
    coverage_matrix_selected = np.zeros((len(selected), len(cities)), dtype=bool)
    for idx, cand in enumerate(selected):
        cand_point = np.array([cand.latitude, cand.longitude], dtype=float)
        for city_idx in range(len(cities)):
            dist = haversine_distance_km(cand_point, city_coords[city_idx])
            if dist <= cand.radius_km:
                coverage_matrix_selected[idx, city_idx] = True

    # For each city, count how many selected towers cover it
    city_coverage_count = coverage_matrix_selected.sum(axis=0)

    refined_candidates = []
    for idx, cand in enumerate(selected):
        # Find cities this tower covers
        covered_cities = np.flatnonzero(coverage_matrix_selected[idx])
        if covered_cities.size == 0:
            refined_candidates.append(cand)
            continue

        # Find the minimum radius needed: max distance to any city covered solely by this tower
        # or to any city it covers (to maintain redundancy we use essential-only cities)
        essential_cities = covered_cities[city_coverage_count[covered_cities] == 1]

        if essential_cities.size == 0:
            # This tower doesn't uniquely cover anyone; keep original radius to be safe
            refined_candidates.append(cand)
            continue

        cand_point = np.array([cand.latitude, cand.longitude], dtype=float)
        max_essential_dist = 0.0
        for city_idx in essential_cities:
            dist = haversine_distance_km(cand_point, city_coords[city_idx])
            max_essential_dist = max(max_essential_dist, dist)

        # Add a small buffer (0.5 km) to avoid floating-point coverage loss
        new_radius = min(cand.radius_km, max(5.0, math.ceil(max_essential_dist + 0.5)))

        if new_radius < cand.radius_km:
            refined_candidates.append(
                TowerCandidate(
                    candidate_id=cand.candidate_id,
                    location_id=cand.location_id,
                    name=cand.name,
                    source=cand.source,
                    tower_type=cand.tower_type,
                    latitude=cand.latitude,
                    longitude=cand.longitude,
                    radius_km=new_radius,
                    cost=tower_cost(new_radius),
                    cluster_label=cand.cluster_label,
                )
            )
        else:
            refined_candidates.append(cand)

    # Verify coverage is maintained after refinement
    from .matrices import build_coverage_matrix as _build_cov

    refined_cov = _build_cov(cities, refined_candidates)
    if not refined_cov.any(axis=0).all():
        # Refinement broke coverage — fall back to original
        fallback_pairs, fallback_penalties = build_interference_pairs(candidates)
        return candidates, build_coverage_matrix(cities, candidates), fallback_pairs, fallback_penalties, optimization

    # Rebuild optimization result with refined candidates
    refined_interference_pairs, refined_penalties = build_interference_pairs(refined_candidates)
    total_cost = float(sum(c.cost for c in refined_candidates))
    total_interference = 0.0
    # Interference is computed on the refined selected set
    for (li, ri), penalty in zip(refined_interference_pairs, refined_penalties):
        total_interference += penalty

    refined_optimization = OptimizationResult(
        status=optimization.status,
        message=optimization.message,
        objective_value=alpha * total_cost + (1.0 - alpha) * total_interference,
        total_cost=total_cost,
        total_interference=total_interference,
        coverage_ratio=float(refined_cov.any(axis=0).mean()),
        selected_indices=np.arange(len(refined_candidates)),
        selected_candidates=refined_candidates,
        pair_values=np.ones(len(refined_interference_pairs), dtype=float),
    )

    return refined_candidates, refined_cov, refined_interference_pairs, refined_penalties, refined_optimization
