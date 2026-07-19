from __future__ import annotations

import numpy as np


# Piecewise cubic cost coefficients: c(r) = a·r³ + b·r² + c·r + d
_COST_SEGMENTS: list[tuple[float, float, float, float, float, float]] = [
    # (r_min, r_max, a,          b,            c,           d)
    (5.0,  20.0,  0.00003898883009994121, -0.0005848324514991181,  0.000818342151675485,  0.9056554967666078),
    (20.0, 35.0, -0.00004679600235155791,  0.004562257495590829, -0.10212345679012345,   1.591934156378601),
    (35.0, 50.0,  0.00005930629041740153, -0.006578483245149912,  0.2878024691358025,   -2.957201646090535),
    (50.0, 100.0,-0.00001544973544973545,  0.004634920634920635, -0.27286772486772487,   6.387301587301588),
]


def tower_cost(radius_km: float) -> float:
    """Return the piecewise installation cost for a tower of radius in km."""

    if radius_km < 5 or radius_km > 100:
        raise ValueError("Radius must be within the inclusive range [5, 100] km.")

    for r_min, r_max, a, b, c, d in _COST_SEGMENTS:
        if r_min <= radius_km <= r_max:
            return float(a * (radius_km**3) + b * (radius_km**2) + c * radius_km + d)

    raise ValueError(f"Radius {radius_km} does not fall into any cost segment.")


def cost_derivative_1(radius_km: float) -> float:
    """First derivative of the piecewise cubic cost: dc/dr."""
    for r_min, r_max, a, b, c, d in _COST_SEGMENTS:
        if r_min <= radius_km <= r_max:
            return float(3.0 * a * radius_km**2 + 2.0 * b * radius_km + c)
    raise ValueError(f"Radius {radius_km} out of range [5, 100].")


def cost_derivative_2(radius_km: float) -> float:
    """Second derivative of the piecewise cubic cost: d²c/dr²."""
    for r_min, r_max, a, b, c, d in _COST_SEGMENTS:
        if r_min <= radius_km <= r_max:
            return float(6.0 * a * radius_km + 2.0 * b)
    raise ValueError(f"Radius {radius_km} out of range [5, 100].")


def cost_critical_radii() -> list[float]:
    """Return radii at which the cost function's curvature changes significantly.

    Includes:
      - Segment boundaries (where the piecewise definition changes)
      - Inflection points within each segment (where f''(r) = 0)

    These are the radii where the cost-per-km trade-off shifts — injecting them
    into the candidate pool avoids missing structurally important radius values.
    """
    critical: set[float] = set()

    for r_min, r_max, a, b, c, d in _COST_SEGMENTS:
        # Segment boundaries
        critical.add(r_min)
        critical.add(r_max)

        # Inflection point: f''(r) = 6a·r + 2b = 0  →  r = -b / (3a)  (a ≠ 0)
        if abs(a) > 1e-18:
            r_inflection = -b / (3.0 * a)
            if r_min <= r_inflection <= r_max:
                critical.add(round(r_inflection, 2))

    return sorted(critical)