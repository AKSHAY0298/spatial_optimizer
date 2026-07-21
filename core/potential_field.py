"""Continuous potential-field relaxation for candidate tower locations.

This is an exploratory method (Phase 7): candidate points are treated as
particles that slide toward equilibrium under attractive forces from cities
and repulsive forces from other candidates, producing spatially-balanced
positions that can then be fed into the discrete MILP.

The method is related to Lloyd's algorithm (k-means) and force-directed
graph layout — it does not replace the MILP, only generates a better
candidate menu for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _attractive_force(
    candidate_km: np.ndarray,
    city_km: np.ndarray,
    city_weights: np.ndarray | None = None,
    falloff_km: float = 50.0,
) -> np.ndarray:
    """Compute attractive force from cities toward each candidate.

    Each city pulls candidates toward it with a force proportional to
    weight and inversely proportional to distance (up to falloff_km).
    """
    n_candidates = len(candidate_km)
    n_cities = len(city_km)

    if city_weights is None:
        city_weights = np.ones(n_cities, dtype=float)

    forces = np.zeros_like(candidate_km)

    for i in range(n_candidates):
        diffs = city_km - candidate_km[i]           # (n_cities, 2)
        dists = np.linalg.norm(diffs, axis=1)       # (n_cities,)
        dists = np.maximum(dists, 0.5)               # avoid division by zero

        # Force magnitude: w / d, clipped to falloff
        mask = dists <= falloff_km
        if mask.any():
            mag = city_weights[mask] / dists[mask]
            direction = diffs[mask] / dists[mask, None]
            forces[i] = (direction * mag[:, None]).sum(axis=0)

    return forces


def _repulsive_force(
    candidate_km: np.ndarray,
    strength: float = 1.0,
    min_distance_km: float = 5.0,
) -> np.ndarray:
    """Compute repulsive force between candidates (1/r² law).

    Candidates push each other apart to avoid clustering. Self-force is zero.
    """
    n = len(candidate_km)
    forces = np.zeros_like(candidate_km)

    for i in range(n):
        diffs = candidate_km - candidate_km[i]        # (n, 2)
        dists = np.linalg.norm(diffs, axis=1)         # (n,)
        dists = np.maximum(dists, 0.1)

        # 1/r² repulsion, zero for self and beyond min_distance
        mask = (dists > 0.1) & (dists < min_distance_km)
        if mask.any():
            mag = strength / (dists[mask] ** 2)
            direction = diffs[mask] / dists[mask, None]
            forces[i] -= (direction * mag[:, None]).sum(axis=0)

    return forces


def relax_candidates(
    candidate_coords_km: np.ndarray,
    city_coords_km: np.ndarray,
    *,
    city_weights: np.ndarray | None = None,
    n_iterations: int = 200,
    learning_rate: float = 0.5,
    attractive_scale: float = 1.0,
    repulsive_scale: float = 0.5,
    falloff_km: float = 80.0,
    min_distance_km: float = 20.0,
    bounds_km: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Iteratively relax candidate positions via gradient descent.

    The total force on each candidate is:
        F = attractive_scale · F_attr  +  repulsive_scale · F_rep

    Candidates slide along the force gradient until convergence or
    max iterations.

    Parameters
    ----------
    candidate_coords_km : np.ndarray of shape (n, 2)
        Initial candidate positions in km-projected coordinates.
    city_coords_km : np.ndarray of shape (m, 2)
        City positions in the same coordinate system.
    city_weights : np.ndarray of shape (m,) or None
        Per-city weight (e.g., population). Uniform if None.
    n_iterations : int
        Maximum gradient descent steps.
    learning_rate : float
        Step size scaling.
    attractive_scale : float
        Weight of attractive (city-pull) forces.
    repulsive_scale : float
        Weight of repulsive (inter-candidate) forces.
    falloff_km : float
        Maximum distance at which a city pulls a candidate.
    min_distance_km : float
        Candidates closer than this repel each other.
    bounds_km : tuple (x_min, x_max, y_min, y_max) or None
        Hard boundary constraint — candidates are clipped to this box.

    Returns
    -------
    np.ndarray of shape (n, 2)
        Relaxed candidate positions.
    """
    positions = candidate_coords_km.copy().astype(float)
    n = len(positions)

    for iteration in range(n_iterations):
        f_attr = _attractive_force(positions, city_coords_km, city_weights, falloff_km)
        f_rep = _repulsive_force(positions, strength=1.0, min_distance_km=min_distance_km)

        total_force = attractive_scale * f_attr + repulsive_scale * f_rep

        # Adaptive learning rate decay
        lr = learning_rate / (1.0 + iteration / 100.0)

        # Update positions
        step = lr * total_force
        max_step = np.max(np.linalg.norm(step, axis=1))
        if max_step > 10.0:
            step *= 10.0 / max_step  # clamp max step per iteration

        positions += step

        # Hard boundary clipping
        if bounds_km is not None:
            x_min, x_max, y_min, y_max = bounds_km
            positions[:, 0] = np.clip(positions[:, 0], x_min, x_max)
            positions[:, 1] = np.clip(positions[:, 1], y_min, y_max)

        # Convergence check
        if max_step < 0.01:
            break

    return positions


def generate_relaxed_locations(
    cities: pd.DataFrame,
    initial_coords_km: np.ndarray,
    *,
    n_relax_iterations: int = 200,
) -> np.ndarray:
    """Generate relaxed candidate positions from initial seed points.

    Wraps relax_candidates() with sensible defaults for the tower placement
    problem and returns the relaxed km-coordinates.
    """
    from .spatial import project_coordinates_km

    city_km = project_coordinates_km(
        cities[["latitude", "longitude"]].to_numpy(dtype=float)
    )

    # Compute bounding box from city positions for clipping
    x_min, x_max = city_km[:, 0].min() - 30, city_km[:, 0].max() + 30
    y_min, y_max = city_km[:, 1].min() - 30, city_km[:, 1].max() + 30
    bounds = (x_min, x_max, y_min, y_max)

    relaxed = relax_candidates(
        initial_coords_km,
        city_km,
        n_iterations=n_relax_iterations,
        learning_rate=0.5,
        attractive_scale=1.0,
        repulsive_scale=0.3,
        falloff_km=60.0,
        min_distance_km=15.0,
        bounds_km=bounds,
    )

    return relaxed
