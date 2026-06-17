from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import tower_cost
from .radius_search import RadiusPlan


@dataclass(frozen=True)
class TowerCandidate:
    candidate_id: str
    name: str
    source: str
    tower_type: str
    latitude: float
    longitude: float
    radius_km: int
    cost: float
    cluster_label: int


def generate_candidates(cities: pd.DataFrame, labels: np.ndarray, radius_plan: RadiusPlan) -> list[TowerCandidate]:
    """Generate cluster-centroid, cluster-city, and noise-city tower candidates."""

    labeled_cities = cities.copy()
    labeled_cities["cluster_label"] = labels

    candidates: list[TowerCandidate] = []
    candidate_index = 0

    # Define the tower specifications available to be placed anywhere
    tower_specs = [
        ("dense", radius_plan.dense_radius_km, tower_cost(radius_plan.dense_radius_km)),
    ]
    # Only add the sparse tower type if its radius is actually different
    if radius_plan.dense_radius_km != radius_plan.sparse_radius_km:
        tower_specs.append(
            ("sparse", radius_plan.sparse_radius_km, tower_cost(radius_plan.sparse_radius_km))
        )

    clustered_groups = labeled_cities[labeled_cities["cluster_label"] >= 0].groupby("cluster_label")
    for cluster_label, group in clustered_groups:
        centroid = group[["latitude", "longitude"]].mean()
        
        # Generate ALL tower types for the centroid
        for t_type, t_radius, t_cost in tower_specs:
            candidates.append(
                TowerCandidate(
                    candidate_id=f"cand_{candidate_index:04d}",
                    name=f"cluster_{int(cluster_label)}_centroid",
                    source="cluster_centroid",
                    tower_type=t_type,
                    latitude=float(centroid["latitude"]),
                    longitude=float(centroid["longitude"]),
                    radius_km=t_radius,
                    cost=t_cost,
                    cluster_label=int(cluster_label),
                )
            )
            candidate_index += 1

        # Generate ALL tower types for every city in the cluster
        for row in group.itertuples(index=False):
            for t_type, t_radius, t_cost in tower_specs:
                candidates.append(
                    TowerCandidate(
                        candidate_id=f"cand_{candidate_index:04d}",
                        name=str(row.city),
                        source="cluster_city",
                        tower_type=t_type,
                        latitude=float(row.latitude),
                        longitude=float(row.longitude),
                        radius_km=t_radius,
                        cost=t_cost,
                        cluster_label=int(cluster_label),
                    )
                )
                candidate_index += 1

    noise_rows = labeled_cities[labeled_cities["cluster_label"] < 0]
    # Generate ALL tower types for every isolated city
    for row in noise_rows.itertuples(index=False):
        for t_type, t_radius, t_cost in tower_specs:
            candidates.append(
                TowerCandidate(
                    candidate_id=f"cand_{candidate_index:04d}",
                    name=str(row.city),
                    source="noise_city",
                    tower_type=t_type,
                    latitude=float(row.latitude),
                    longitude=float(row.longitude),
                    radius_km=t_radius,
                    cost=t_cost,
                    cluster_label=-1,
                )
            )
            candidate_index += 1

    return candidates