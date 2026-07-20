from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .candidates import TowerCandidate
from .spatial import haversine_distance_km, haversine_rows_km, pairwise_haversine_km, project_coordinates_km


@dataclass(frozen=True)
class InterferencePair:
    left_index: int
    right_index: int
    penalty: float


def build_coverage_matrix(cities: pd.DataFrame, candidates: list[TowerCandidate]) -> np.ndarray:
    """Return a binary matrix that marks whether a candidate covers each city."""

    if not candidates:
        return np.zeros((0, len(cities)), dtype=int)

    city_coordinates = cities[["latitude", "longitude"]].to_numpy(dtype=float)
    candidate_coordinates = np.array(
        [[candidate.latitude, candidate.longitude] for candidate in candidates],
        dtype=float,
    )
    candidate_radii = np.array([candidate.radius_km for candidate in candidates], dtype=float)
    distances = pairwise_haversine_km(candidate_coordinates, city_coordinates)
    return (distances <= candidate_radii[:, None]).astype(int)


def normalized_overlap_penalty(distance_km: float, radius_a_km: float, radius_b_km: float) -> float:
    """Compute the normalized overlap penalty."""

    radius_sum = radius_a_km + radius_b_km
    if distance_km >= radius_sum:
        return 0.0
    return float((radius_sum - distance_km) / radius_sum)


def build_interference_pairs(candidates: list[TowerCandidate]) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Build filtered candidate pairs and their overlap penalties using a KD-tree.

    Pairs of candidates sharing the same physical location are skipped: the
    mutual-exclusion constraint prevents them from ever being selected together,
    so they can never contribute interference.
    """

    if len(candidates) < 2:
        return [], np.zeros(0, dtype=float)

    coordinates = np.array(
        [[candidate.latitude, candidate.longitude] for candidate in candidates],
        dtype=float,
    )
    radii = np.array([candidate.radius_km for candidate in candidates], dtype=float)
    projected_coordinates = project_coordinates_km(coordinates)
    tree = cKDTree(projected_coordinates)
    max_radius = float(radii.max())
    # 5% margin absorbs projection distortion relative to haversine distances
    pair_array = tree.query_pairs(r=2.1 * max_radius, output_type="ndarray")
    if pair_array.size == 0:
        return [], np.zeros(0, dtype=float)

    left = pair_array[:, 0]
    right = pair_array[:, 1]

    location_ids = np.array([candidate.location_id or "" for candidate in candidates])
    distinct_location_mask = location_ids[left] != location_ids[right]
    left = left[distinct_location_mask]
    right = right[distinct_location_mask]
    if left.size == 0:
        return [], np.zeros(0, dtype=float)

    distances_km = haversine_rows_km(coordinates[left], coordinates[right])
    radius_sums = radii[left] + radii[right]
    penalties = (radius_sums - distances_km) / radius_sums
    overlap_mask = penalties > 0.0

    left = left[overlap_mask]
    right = right[overlap_mask]
    penalties = penalties[overlap_mask]

    order = np.lexsort((right, left))
    left = left[order]
    right = right[order]
    penalties = penalties[order]

    filtered_pairs = list(zip(left.tolist(), right.tolist()))
    return filtered_pairs, np.asarray(penalties, dtype=float)