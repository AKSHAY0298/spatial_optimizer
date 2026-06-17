from __future__ import annotations


def tower_cost(radius_km: float) -> float:
    """Return the piecewise installation cost for a tower of radius in km."""

    if radius_km < 5 or radius_km > 100:
        raise ValueError("Radius must be within the inclusive range [5, 100] km.")

    if 5 <= radius_km <= 20:
        return (
            0.00003898883009994121 * (radius_km**3)
            - 0.0005848324514991181 * (radius_km**2)
            + 0.000818342151675485 * radius_km
            + 0.9056554967666078
        )
    if 20 < radius_km <= 35:
        return (
            -0.00004679600235155791 * (radius_km**3)
            + 0.004562257495590829 * (radius_km**2)
            - 0.10212345679012345 * radius_km
            + 1.591934156378601
        )
    if 35 < radius_km <= 50:
        return (
            0.00005930629041740153 * (radius_km**3)
            - 0.006578483245149912 * (radius_km**2)
            + 0.2878024691358025 * radius_km
            - 2.957201646090535
        )
    return (
        -0.00001544973544973545 * (radius_km**3)
        + 0.004634920634920635 * (radius_km**2)
        - 0.27286772486772487 * radius_km
        + 6.387301587301588
    )