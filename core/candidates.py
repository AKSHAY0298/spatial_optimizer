from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Voronoi, cKDTree

from .costs import tower_cost
from .radius_search import RadiusPlan


def _meanshift_candidate_locations(
    cities: pd.DataFrame,
    bandwidth_quantile: float = 0.08,
) -> list[tuple[float, float]]:
    """Generate candidate tower locations from Mean-Shift density modes.

    Mean-Shift is a mode-seeking algorithm that finds local maxima of the
    probability density function of city coordinates. Unlike DBSCAN (which
    finds connected dense regions), Mean-Shift identifies the exact points
    of highest city density — natural locations for coverage-maximizing towers.

    Parameters
    ----------
    cities : pd.DataFrame
        Must contain 'latitude' and 'longitude' columns.
    bandwidth_quantile : float
        Controls the kernel bandwidth as a quantile of pairwise distances.
        Lower values → smaller bandwidth → more modes (clusters).
        0.25 gives ~15-40 modes for 236 cities.
    """
    from sklearn.cluster import MeanShift, estimate_bandwidth

    from .spatial import project_coordinates_km

    coords = cities[["latitude", "longitude"]].to_numpy(dtype=float)
    if len(coords) < 3:
        return []

    projected = project_coordinates_km(coords)

    # Estimate bandwidth from the data
    try:
        bandwidth = estimate_bandwidth(projected, quantile=bandwidth_quantile, n_samples=min(200, len(projected)))
        bandwidth = max(bandwidth, 15.0)  # minimum 15 km bandwidth
    except Exception:
        bandwidth = 30.0  # fallback

    # Mean-Shift clustering
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, max_iter=300)
    ms.fit(projected)

    # Convert cluster centers back to lat/lon
    ref_lat_rad = float(np.radians(coords[:, 0]).mean())
    cos_ref = math.cos(ref_lat_rad)
    earth_r = 6371.0088

    candidates: list[tuple[float, float]] = []
    for center in ms.cluster_centers_:
        lat = float(np.degrees(center[1] / earth_r))
        lon = float(np.degrees(center[0] / (earth_r * cos_ref)))
        candidates.append((lat, lon))

    return candidates


@dataclass(frozen=True)
class TowerCandidate:
    candidate_id: str
    location_id: str
    name: str
    source: str
    tower_type: str
    latitude: float
    longitude: float
    radius_km: float
    cost: float
    cluster_label: int


@dataclass(frozen=True)
class TowerLocation:
    location_id: str
    name: str
    source: str
    latitude: float
    longitude: float
    cluster_label: int


def _midpoint_locations(
    cities: pd.DataFrame,
    max_distance_km: float,
) -> list[tuple[float, float]]:
    """Generate midpoint candidates between neighboring cities within *max_distance_km*."""

    from .spatial import project_coordinates_km

    coords = cities[["latitude", "longitude"]].to_numpy(dtype=float)
    projected = project_coordinates_km(coords)
    tree = cKDTree(projected)
    pairs = tree.query_pairs(r=max_distance_km)

    midpoints: list[tuple[float, float]] = []
    for i, j in pairs:
        mid_lat = (coords[i, 0] + coords[j, 0]) / 2.0
        mid_lon = (coords[i, 1] + coords[j, 1]) / 2.0
        midpoints.append((mid_lat, mid_lon))

    return midpoints


def _grid_locations(
    cities: pd.DataFrame,
    spacing_km: float,
) -> list[tuple[float, float]]:
    """Generate a hexagonal grid of candidate points covering the bounding box of cities."""

    coords = cities[["latitude", "longitude"]].to_numpy(dtype=float)
    lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
    lon_min, lon_max = coords[:, 1].min(), coords[:, 1].max()

    # Add padding of one spacing unit
    lat_step = spacing_km / 110.574
    mean_lat = (lat_min + lat_max) / 2.0
    lon_step = spacing_km / (111.320 * math.cos(math.radians(mean_lat)))

    lat_min -= lat_step
    lat_max += lat_step
    lon_min -= lon_step
    lon_max += lon_step

    points: list[tuple[float, float]] = []
    row_idx = 0
    lat = lat_min
    while lat <= lat_max:
        lon_offset = (lon_step / 2.0) if (row_idx % 2 == 1) else 0.0
        lon = lon_min + lon_offset
        while lon <= lon_max:
            points.append((lat, lon))
            lon += lon_step
        lat += lat_step
        row_idx += 1

    # Filter: keep only points within max_distance_km of at least one city
    if not points:
        return []

    from .spatial import project_coordinates_km

    grid_arr = np.array(points, dtype=float)
    projected_grid = project_coordinates_km(grid_arr)
    projected_cities = project_coordinates_km(coords)
    city_tree = cKDTree(projected_cities)

    # Keep grid points within 1.5x spacing of a city (they are useful candidates)
    keep_radius = spacing_km * 1.5
    close_mask = city_tree.query(projected_grid)[0] <= keep_radius
    return [points[i] for i in range(len(points)) if close_mask[i]]


def _weisafeld_geometric_median(
    points_km: np.ndarray,
    weights: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> np.ndarray:
    """Compute the weighted geometric median via Weiszfeld's algorithm.

    The geometric median minimizes Σ wᵢ · ||x - pᵢ||, the sum of weighted
    Euclidean distances to all input points. It is more robust to outliers
    than the arithmetic mean (centroid).

    Parameters
    ----------
    points_km : np.ndarray of shape (n, 2)
        Points in a flat km-projected coordinate system.
    weights : np.ndarray of shape (n,) or None
        Point weights (e.g., city population). Uniform if None.
    """
    n = len(points_km)
    if n == 0:
        raise ValueError("At least one point is required.")
    if n == 1:
        return points_km[0].copy()

    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()

    # Start from the weighted arithmetic mean
    current = np.average(points_km, axis=0, weights=weights)

    for _ in range(max_iter):
        diffs = points_km - current
        dists = np.linalg.norm(diffs, axis=1)
        # Guard: if current coincides with a data point, nudge it
        zero_mask = dists < 1e-12
        if zero_mask.any():
            dists[zero_mask] = 1e-12

        w_over_d = weights / dists
        denominator = w_over_d.sum()
        if denominator < 1e-15:
            break

        new_point = (points_km * w_over_d[:, None]).sum(axis=0) / denominator
        shift = np.linalg.norm(new_point - current)
        current = new_point
        if shift < tol:
            break

    return current


def _voronoi_candidate_locations(
    cities: pd.DataFrame,
    buffer_km: float = 20.0,
    max_nearest_city_km: float = 50.0,
) -> list[tuple[float, float]]:
    """Generate candidate tower locations from Voronoi vertices of city positions.

    Voronoi vertices are points equidistant from three or more cities — locally
    optimal for minimizing the maximum distance to nearby cities. We keep only
    finite vertices that:
      - Fall within the convex hull of the cities (plus *buffer_km*).
      - Have at least one city within *max_nearest_city_km*.

    Parameters
    ----------
    cities : pd.DataFrame
        Must contain 'latitude' and 'longitude' columns.
    buffer_km : float
        Extra margin (km) beyond the convex hull to accept Voronoi vertices.
    max_nearest_city_km : float
        Maximum distance to the nearest city for a Voronoi vertex to be kept.
        Prunes remote vertices that would never be useful as tower locations.
    """
    from .spatial import project_coordinates_km

    coords = cities[["latitude", "longitude"]].to_numpy(dtype=float)
    if len(coords) < 3:
        return []  # Voronoi requires at least 3 points

    projected = project_coordinates_km(coords)

    vor = Voronoi(projected)
    if vor.vertices is None or len(vor.vertices) == 0:
        return []

    # Build convex hull for filtering
    try:
        hull = ConvexHull(projected)
        hull_vertices = projected[hull.vertices]
    except Exception:
        hull_vertices = projected

    hull_center = hull_vertices.mean(axis=0)
    hull_radius = float(np.linalg.norm(hull_vertices - hull_center, axis=1).max())

    # Build KD-tree of city projected coords for nearest-neighbor queries
    city_tree = cKDTree(projected)

    # Query nearest city for each Voronoi vertex
    if len(vor.vertices) > 0:
        nearest_dists, _ = city_tree.query(vor.vertices, k=1)
    else:
        return []

    # Reference latitude for inverse projection
    ref_lat_rad = float(np.radians(coords[:, 0]).mean())
    cos_ref = math.cos(ref_lat_rad)
    earth_r = 6371.0088

    candidates: list[tuple[float, float]] = []
    for i, vertex in enumerate(vor.vertices):
        # Filter by hull containment
        dist_to_center = float(np.linalg.norm(vertex - hull_center))
        if dist_to_center > hull_radius + buffer_km:
            continue

        # Filter by nearest city distance (prunes remote vertices)
        if float(nearest_dists[i]) > max_nearest_city_km:
            continue

        lat = float(np.degrees(vertex[1] / earth_r))
        lon = float(np.degrees(vertex[0] / (earth_r * cos_ref)))
        candidates.append((lat, lon))

    return candidates


def generate_candidate_locations(
    cities: pd.DataFrame,
    labels: np.ndarray,
    *,
    include_cluster_centroids: bool = True,
    include_midpoints: bool = True,
    midpoint_max_distance_km: float = 80.0,
    include_voronoi: bool = False,
    include_meanshift: bool = False,
    include_grid: bool = False,
    grid_spacing_km: float = 50.0,
    dedup_tolerance_km: float = 1.5,
) -> list[TowerLocation]:
    """Generate possible physical tower locations independent of radius.

    Sources:
      1. Cluster centroids (arithmetic mean of DBSCAN cluster members)
      2. Every city location (always included)
      3. Midpoints between neighboring cities (geometric candidates)
      4. Voronoi vertices within the convex hull (experimental, opt-in)
      5. Mean-Shift density modes (experimental, opt-in)
      6. Hexagonal grid points near cities (optional, for very dense search)

    All sources are merged and near-duplicates (within *dedup_tolerance_km*)
    are removed, keeping the first occurrence.
    """

    from .spatial import project_coordinates_km

    labeled_cities = cities.copy()
    labeled_cities["cluster_label"] = labels

    # Temporary storage: (lat, lon, name, source, cluster_label)
    raw: list[tuple[float, float, str, str, int]] = []

    # --- Cluster centroids (arithmetic mean) ---
    if include_cluster_centroids:
        clustered_groups = labeled_cities[labeled_cities["cluster_label"] >= 0].groupby("cluster_label")
        for cluster_label, group in clustered_groups:
            centroid = group[["latitude", "longitude"]].mean()
            raw.append((float(centroid["latitude"]), float(centroid["longitude"]),
                        f"cluster_{int(cluster_label)}_centroid", "cluster_centroid", int(cluster_label)))

    # --- Every city is a candidate location ---
    for row in labeled_cities.itertuples(index=False):
        source = "noise_city" if int(row.cluster_label) < 0 else "cluster_city"
        raw.append((float(row.latitude), float(row.longitude), str(row.city), source, int(row.cluster_label)))

    # --- Midpoint candidates between neighboring cities ---
    if include_midpoints:
        for mid_lat, mid_lon in _midpoint_locations(cities, max_distance_km=midpoint_max_distance_km):
            raw.append((mid_lat, mid_lon, f"midpoint_{len(raw)}", "midpoint", -1))

    # --- Voronoi vertex candidates (keep only top-N most isolated) ---
    if include_voronoi:
        voronoi_raw = _voronoi_candidate_locations(cities, max_nearest_city_km=20.0)
        # Sort by distance from nearest city (descending) and keep top 50 —
        # these are the most isolated points in coverage gaps, offering the
        # best new coverage options that city locations alone can't provide
        if len(voronoi_raw) > 50:
            from .spatial import pairwise_haversine_km
            vor_coords = np.array(voronoi_raw, dtype=float)
            city_coords = cities[["latitude", "longitude"]].to_numpy(dtype=float)
            vor_dists = pairwise_haversine_km(vor_coords, city_coords).min(axis=1)
            top_idx = np.argsort(vor_dists)[-50:]  # farthest 50 from any city
            voronoi_raw = [voronoi_raw[i] for i in sorted(top_idx)]
        for vor_lat, vor_lon in voronoi_raw:
            raw.append((vor_lat, vor_lon, f"voronoi_{len(raw)}", "voronoi", -1))

    # --- Mean-Shift density mode candidates ---
    if include_meanshift:
        for ms_lat, ms_lon in _meanshift_candidate_locations(cities):
            raw.append((ms_lat, ms_lon, f"meanshift_{len(raw)}", "meanshift", -1))

    # --- Grid-based candidates ---
    if include_grid:
        for grid_lat, grid_lon in _grid_locations(cities, spacing_km=grid_spacing_km):
            raw.append((grid_lat, grid_lon, f"grid_{len(raw)}", "grid", -1))

    # --- Deduplicate (only among non-city sources; cities are always kept) ---
    if dedup_tolerance_km > 0 and len(raw) > 1:
        from .spatial import pairwise_haversine_km

        # Determine which entries are actual cities (always keep them)
        city_indices = {i for i, (_, _, _, source, _) in enumerate(raw) if source in ("cluster_city", "noise_city")}

        raw_coords_arr = np.array([(lat, lon) for lat, lon, _, _, _ in raw], dtype=float)
        keep_mask = np.ones(len(raw_coords_arr), dtype=bool)

        for i in range(len(raw_coords_arr)):
            if not keep_mask[i]:
                continue
            later = np.flatnonzero(keep_mask)[np.flatnonzero(keep_mask) > i]
            if len(later) == 0:
                continue
            a = raw_coords_arr[i:i + 1]
            b = raw_coords_arr[later]
            dists = pairwise_haversine_km(a, b)[0]
            close_mask = dists <= dedup_tolerance_km

            # Never remove a city location
            for j_idx, j in enumerate(later):
                if close_mask[j_idx] and j not in city_indices:
                    keep_mask[j] = False

        raw = [raw[i] for i in range(len(raw)) if keep_mask[i]]

    # --- Build TowerLocation list ---
    locations: list[TowerLocation] = []
    for idx, (lat, lon, name, source, cluster_label) in enumerate(raw):
        locations.append(
            TowerLocation(
                location_id=f"loc_{idx:04d}",
                name=name,
                source=source,
                latitude=lat,
                longitude=lon,
                cluster_label=cluster_label,
            )
        )

    return locations


def generate_candidates_from_locations(
    locations: list[TowerLocation],
    radii_km: tuple[float, float],
    *,
    tower_type_names: tuple[str, str] = ("dense", "sparse"),
) -> list[TowerCandidate]:
    """Generate one candidate per physical location and tower type."""

    radius_a, radius_b = (float(radii_km[0]), float(radii_km[1]))
    tower_specs = [
        (tower_type_names[0], radius_a, tower_cost(radius_a)),
        (tower_type_names[1], radius_b, tower_cost(radius_b)),
    ]

    candidates: list[TowerCandidate] = []
    candidate_index = 0
    for location in locations:
        for tower_type, radius_km, cost in tower_specs:
            candidates.append(
                TowerCandidate(
                    candidate_id=f"cand_{candidate_index:04d}",
                    location_id=location.location_id,
                    name=location.name,
                    source=location.source,
                    tower_type=tower_type,
                    latitude=location.latitude,
                    longitude=location.longitude,
                    radius_km=radius_km,
                    cost=cost,
                    cluster_label=location.cluster_label,
                )
            )
            candidate_index += 1

    return candidates


def generate_candidates(cities: pd.DataFrame, labels: np.ndarray, radius_plan: RadiusPlan) -> list[TowerCandidate]:
    """Generate cluster-centroid, cluster-city, and noise-city tower candidates."""
    locations = generate_candidate_locations(cities, labels)
    return generate_candidates_from_locations(
        locations,
        (radius_plan.dense_radius_km, radius_plan.sparse_radius_km),
    )
