from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .candidates import TowerCandidate
from .spatial import haversine_distance_km, pairwise_haversine_km, project_coordinates_km


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
    """Build filtered candidate pairs and their overlap penalties using a KD-tree."""

    if len(candidates) < 2:
        return [], np.zeros(0, dtype=float)

    coordinates = np.array(
        [[candidate.latitude, candidate.longitude] for candidate in candidates],
        dtype=float,
    )
    projected_coordinates = project_coordinates_km(coordinates)
    tree = cKDTree(projected_coordinates)
    max_radius = max(candidate.radius_km for candidate in candidates)
    candidate_pairs = sorted(tree.query_pairs(r=2.0 * max_radius))

    filtered_pairs: list[tuple[int, int]] = []
    penalties: list[float] = []

    for left_index, right_index in candidate_pairs:
        distance_km = haversine_distance_km(coordinates[left_index], coordinates[right_index])
        penalty = normalized_overlap_penalty(
            distance_km,
            candidates[left_index].radius_km,
            candidates[right_index].radius_km,
        )
        if penalty > 0.0:
            filtered_pairs.append((left_index, right_index))
            penalties.append(penalty)

    return filtered_pairs, np.asarray(penalties, dtype=float)