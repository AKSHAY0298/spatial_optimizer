from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class ClusterProfile:
    cluster_count: int
    noise_count: int
    cluster_sizes: dict[int, int]


def haversine_distance_km(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Return the great-circle distance in kilometers between two latitude/longitude points."""

    latitude_1, longitude_1 = np.radians(np.asarray(point_a, dtype=float))
    latitude_2, longitude_2 = np.radians(np.asarray(point_b, dtype=float))
    delta_latitude = latitude_2 - latitude_1
    delta_longitude = longitude_2 - longitude_1
    a = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(latitude_1) * np.cos(latitude_2) * np.sin(delta_longitude / 2.0) ** 2
    )
    return float(2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def pairwise_haversine_km(points_a: np.ndarray, points_b: np.ndarray | None = None) -> np.ndarray:
    """Compute a matrix of great-circle distances in kilometers."""

    coordinates_a = np.asarray(points_a, dtype=float)
    coordinates_b = coordinates_a if points_b is None else np.asarray(points_b, dtype=float)

    latitude_a = np.radians(coordinates_a[:, 0])[:, None]
    longitude_a = np.radians(coordinates_a[:, 1])[:, None]
    latitude_b = np.radians(coordinates_b[:, 0])[None, :]
    longitude_b = np.radians(coordinates_b[:, 1])[None, :]

    delta_latitude = latitude_b - latitude_a
    delta_longitude = longitude_b - longitude_a
    a = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(latitude_a) * np.cos(latitude_b) * np.sin(delta_longitude / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def cluster_cities(cities: pd.DataFrame, eps_km: float = 35.0, min_samples: int = 2) -> np.ndarray:
    """Cluster cities with DBSCAN using a haversine distance metric."""

    coordinates = np.radians(cities[["latitude", "longitude"]].to_numpy(dtype=float))
    clusterer = DBSCAN(
        eps=eps_km / EARTH_RADIUS_KM,
        min_samples=min_samples,
        metric="haversine",
    )
    return clusterer.fit_predict(coordinates)


def describe_clusters(labels: np.ndarray) -> ClusterProfile:
    """Summarize cluster labels for reporting and debugging."""

    cluster_ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    cluster_sizes = {int(cluster_id): int(count) for cluster_id, count in zip(cluster_ids, counts)}
    return ClusterProfile(
        cluster_count=len(cluster_ids),
        noise_count=int(np.count_nonzero(labels < 0)),
        cluster_sizes=cluster_sizes,
    )


def cluster_centroids(cities: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Return centroid coordinates for all non-noise clusters."""

    labeled_cities = cities.copy()
    labeled_cities["cluster_label"] = labels
    centroids = []
    for cluster_label, group in labeled_cities[labeled_cities["cluster_label"] >= 0].groupby("cluster_label"):
        centroid = group[["latitude", "longitude"]].mean()
        centroids.append(
            {
                "cluster_label": int(cluster_label),
                "latitude": float(centroid["latitude"]),
                "longitude": float(centroid["longitude"]),
                "city_count": int(len(group)),
            }
        )
    return pd.DataFrame(centroids)


def split_clustered_cities(cities: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the city table into clustered rows and noise rows."""

    labeled_cities = cities.copy()
    labeled_cities["cluster_label"] = labels
    clustered = labeled_cities[labeled_cities["cluster_label"] >= 0].copy()
    noise = labeled_cities[labeled_cities["cluster_label"] < 0].copy()
    return clustered, noise


def project_coordinates_km(coordinates: np.ndarray) -> np.ndarray:
    """Project latitude/longitude coordinates into a local kilometer-based plane."""

    coordinates = np.asarray(coordinates, dtype=float)
    latitude_radians = np.radians(coordinates[:, 0])
    longitude_radians = np.radians(coordinates[:, 1])
    reference_latitude = float(latitude_radians.mean()) if len(latitude_radians) else 0.0
    x = EARTH_RADIUS_KM * longitude_radians * np.cos(reference_latitude)
    y = EARTH_RADIUS_KM * latitude_radians
    return np.column_stack((x, y))