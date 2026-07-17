from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .costs import tower_cost
from .radius_search import RadiusPlan


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


def generate_candidate_locations(
    cities: pd.DataFrame,
    labels: np.ndarray,
    *,
    include_cluster_centroids: bool = True,
    include_midpoints: bool = True,
    midpoint_max_distance_km: float = 80.0,
    include_grid: bool = False,
    grid_spacing_km: float = 50.0,
) -> list[TowerLocation]:
    """Generate possible physical tower locations independent of radius.

    Sources:
      1. Cluster centroids (from DBSCAN labels) — optional enrichment
      2. Every city location (always included)
      3. Midpoints between neighboring cities (geometric candidates)
      4. Hexagonal grid points near cities (optional, for very dense search)
    """

    labeled_cities = cities.copy()
    labeled_cities["cluster_label"] = labels

    locations: list[TowerLocation] = []
    location_index = 0

    # --- Cluster centroids (enrichment only) ---
    if include_cluster_centroids:
        clustered_groups = labeled_cities[labeled_cities["cluster_label"] >= 0].groupby("cluster_label")
        for cluster_label, group in clustered_groups:
            centroid = group[["latitude", "longitude"]].mean()
            locations.append(
                TowerLocation(
                    location_id=f"loc_{location_index:04d}",
                    name=f"cluster_{int(cluster_label)}_centroid",
                    source="cluster_centroid",
                    latitude=float(centroid["latitude"]),
                    longitude=float(centroid["longitude"]),
                    cluster_label=int(cluster_label),
                )
            )
            location_index += 1

    # --- Every city is a candidate location ---
    for row in labeled_cities.itertuples(index=False):
        source = "noise_city" if int(row.cluster_label) < 0 else "cluster_city"
        locations.append(
            TowerLocation(
                location_id=f"loc_{location_index:04d}",
                name=str(row.city),
                source=source,
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                cluster_label=int(row.cluster_label),
            )
        )
        location_index += 1

    # --- Midpoint candidates between neighboring cities ---
    if include_midpoints:
        midpoints = _midpoint_locations(cities, max_distance_km=midpoint_max_distance_km)
        for mid_lat, mid_lon in midpoints:
            locations.append(
                TowerLocation(
                    location_id=f"loc_{location_index:04d}",
                    name=f"midpoint_{location_index}",
                    source="midpoint",
                    latitude=mid_lat,
                    longitude=mid_lon,
                    cluster_label=-1,
                )
            )
            location_index += 1

    # --- Grid-based candidates ---
    if include_grid:
        grid_points = _grid_locations(cities, spacing_km=grid_spacing_km)
        for grid_lat, grid_lon in grid_points:
            locations.append(
                TowerLocation(
                    location_id=f"loc_{location_index:04d}",
                    name=f"grid_{location_index}",
                    source="grid",
                    latitude=grid_lat,
                    longitude=grid_lon,
                    cluster_label=-1,
                )
            )
            location_index += 1

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
