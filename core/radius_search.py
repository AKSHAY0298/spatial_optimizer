from __future__ import annotations

from dataclasses import dataclass
import math

# pyrefly: ignore [missing-import]
import numpy as np 
import pandas as pd

from .costs import tower_cost
from .spatial import cluster_centroids, pairwise_haversine_km, EARTH_RADIUS_KM


@dataclass(frozen=True)
class RadiusPlan:
    dense_radius_km: int
    sparse_radius_km: int
    dense_score: float
    sparse_score: float
    dense_bounds_km: tuple[int, int]
    sparse_bounds_km: tuple[int, int]


def _integer_bounds(points: np.ndarray) -> tuple[int, int]:
    n = len(points)
    if n < 2:
        return 5, 100 # This is bound for type 1 tower (Edge case)

    coords = np.asarray(points, dtype=float)
    # For small inputs compute full pairwise distances (exact)
    if n <= 2000:
        return _compute_exact_bounds(coords)

    # For large inputs, use BallTree to get nearest neighbors in O(n log n)
    try:
        from sklearn.neighbors import BallTree
    except Exception:
        # Fall back to exact computation if BallTree isn't available
        return _compute_exact_bounds(coords)

    # Build BallTree using radians and haversine metric (returns radian distances)
    rad = np.radians(coords)
    tree = BallTree(rad, metric="haversine")

    # Query the 2 nearest neighbors (self + nearest) to get per-point NN
    dist_rad, _ = tree.query(rad, k=2)
    # dist_rad[:, 1] is nearest neighbor in radians; convert to km
    nearest_km = dist_rad[:, 1] * EARTH_RADIUS_KM
    finite_nearest = nearest_km[np.isfinite(nearest_km)]
    if finite_nearest.size == 0:
        return 5, 100

    d10 = float(np.percentile(finite_nearest, 10))
    lower_bound = max(5, int(math.floor(d10 / 2.0)))

    # Approximate upper bound via sampling to avoid full O(n^2) pairwise work
    m = min(2000, n)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, size=m, replace=False)
    sample_coords = coords[sample_idx]
    sample_dists = pairwise_haversine_km(sample_coords)
    sample_upper_tri = sample_dists[np.triu_indices(m, k=1)]
    sample_positive = sample_upper_tri[sample_upper_tri > 0]
    if sample_positive.size == 0:
        upper_bound = 100
    else:
        upper_pct = float(np.percentile(sample_positive, 95))
        upper_bound = min(100, int(math.ceil(upper_pct)))

    if lower_bound > upper_bound:
        lower_bound = upper_bound
    return lower_bound, upper_bound


def _compute_exact_bounds(coords: np.ndarray) -> tuple[int, int]:
    
    """Compute bounds exactly from full pairwise distances (used for small n
    or fallback when BallTree is unavailable)."""
    distances = pairwise_haversine_km(coords)
    n = len(coords)
    upper_triangle = distances[np.triu_indices(n, k=1)]
    positive_distances = upper_triangle[upper_triangle > 0]
    if positive_distances.size == 0:
        return 5, 100 # Edge 

    masked = np.where(distances > 0, distances, np.inf)
    nearest = masked.min(axis=1)
    finite_nearest = nearest[np.isfinite(nearest)]
    if finite_nearest.size == 0:
        return 5, 100

    d10 = float(np.percentile(finite_nearest, 10))
    lower_bound = max(5, int(math.floor(d10 / 2.0)))
    upper_pct = float(np.percentile(positive_distances, 95))
    upper_bound = min(100, int(math.ceil(upper_pct)))
    if lower_bound > upper_bound:
        lower_bound = upper_bound
    return lower_bound, upper_bound


def _representatives_for_dense_search(cities: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    labeled_cities = cities.copy()
    labeled_cities["cluster_label"] = labels
    representatives: list[np.ndarray] = []

    for cluster_label, group in labeled_cities[labeled_cities["cluster_label"] >= 0].groupby("cluster_label"):
        centroid = group[["latitude", "longitude"]].mean().to_numpy(dtype=float)
        representatives.append(centroid)
        representatives.extend(group[["latitude", "longitude"]].to_numpy(dtype=float))

    if not representatives:
        return cities[["latitude", "longitude"]].to_numpy(dtype=float)

    return np.asarray(representatives, dtype=float)


def _representatives_for_sparse_search(cities: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    labeled_cities = cities.copy()
    labeled_cities["cluster_label"] = labels
    noise = labeled_cities[labeled_cities["cluster_label"] < 0]

    if noise.empty:
        return cities[["latitude", "longitude"]].to_numpy(dtype=float)

    return noise[["latitude", "longitude"]].to_numpy(dtype=float)


def _search_radius(
    representative_points: np.ndarray,
    city_points: np.ndarray,
    bounds: tuple[int, int],
    *,
    coarse_step: int = 15,
    fine_window: int = 14,
    top_k: int = 3,
) -> tuple[int, float]:
    lower_bound, upper_bound = bounds
    if lower_bound == upper_bound:
        score = tower_cost(lower_bound) / max(1.0, float((pairwise_haversine_km(representative_points, city_points) <= lower_bound).sum(axis=1).mean()))
        return lower_bound, score

    distances = pairwise_haversine_km(representative_points, city_points)

    def score(radius_km: int) -> float:
        coverage_counts = (distances <= radius_km).sum(axis=1)
        average_coverage = max(1.0, float(coverage_counts.mean()))
        return tower_cost(radius_km) / average_coverage

    coarse_radii = list(range(lower_bound, upper_bound + 1, coarse_step))
    if coarse_radii[-1] != upper_bound:
        coarse_radii.append(upper_bound)

    coarse_rank = sorted(((radius, score(radius)) for radius in coarse_radii), key=lambda item: item[1])
    candidate_radii = set()
    for radius, _ in coarse_rank[:top_k]:
        start = max(lower_bound, radius - fine_window)
        stop = min(upper_bound, radius + fine_window)
        candidate_radii.update(range(start, stop + 1))

    best_radius = min(candidate_radii, key=score)
    return best_radius, score(best_radius)


def plan_tower_radii(cities: pd.DataFrame, labels: np.ndarray, *, coarse_step: int = 15, fine_window: int = 14, top_k: int = 3) -> RadiusPlan:
    """Select the two tower radii using a coarse-to-fine search."""

    city_points = cities[["latitude", "longitude"]].to_numpy(dtype=float)
    dense_representatives = _representatives_for_dense_search(cities, labels)
    sparse_representatives = _representatives_for_sparse_search(cities, labels)

    labeled_cities = cities.copy()
    labeled_cities["cluster_label"] = labels
    clustered_points = labeled_cities[labeled_cities["cluster_label"] >= 0][["latitude", "longitude"]].to_numpy(dtype=float)
    noise_points = labeled_cities[labeled_cities["cluster_label"] < 0][["latitude", "longitude"]].to_numpy(dtype=float)

    dense_bounds = _integer_bounds(clustered_points if len(clustered_points) else city_points)
    sparse_bounds = _integer_bounds(noise_points if len(noise_points) else city_points)

    dense_radius, dense_score = _search_radius(
        dense_representatives,
        city_points,
        dense_bounds,
        coarse_step=coarse_step,
        fine_window=fine_window,
        top_k=top_k,
    )
    sparse_radius, sparse_score = _search_radius(
        sparse_representatives,
        city_points,
        sparse_bounds,
        coarse_step=coarse_step,
        fine_window=fine_window,
        top_k=top_k,
    )

    return RadiusPlan(
        dense_radius_km=int(dense_radius),
        sparse_radius_km=int(sparse_radius),
        dense_score=float(dense_score),
        sparse_score=float(sparse_score),
        dense_bounds_km=dense_bounds,
        sparse_bounds_km=sparse_bounds,
    )
