from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .candidates import generate_candidates_from_locations, TowerLocation
from .matrices import build_coverage_matrix, build_interference_pairs
from .optimizer import solve_with_lazy_interference


@dataclass(frozen=True)
class ParetoPoint:
    """One point on the cost-vs-interference Pareto front."""

    epsilon: float
    cost: float
    interference: float
    towers: int
    coverage_ratio: float
    solver_status: str
    lp_bound: float | None = None
    solve_time_s: float = 0.0


def compute_pareto_front(
    locations: list[TowerLocation],
    cities: pd.DataFrame,
    radii_km: tuple[float, float],
    *,
    num_steps: int = 20,
    time_limit: float | None = 30.0,
    use_lp_pruning: bool = True,
    coverage_requirement: float = 1.0,
) -> list[ParetoPoint]:
    """Trace the cost-vs-interference Pareto front via ε-constraint MILP solves.

    For each ε from min to max interference:
        minimize   Σ cost(i)·xᵢ
        s.t.       Σ Pₚ·yₚ  ≤  ε
                   (all standard constraints)

    A binary search first locates the ε threshold where cost transitions
    from the high-cost (strict interference) regime to the low-cost
    (relaxed interference) regime. The front typically has 1-3 distinct
    cost levels forming a step function.
    """
    # ── Build shared problem data ─────────────────────────────────────
    candidates = generate_candidates_from_locations(locations, radii_km)
    coverage_matrix = build_coverage_matrix(cities, candidates)
    keep_mask = coverage_matrix.sum(axis=1) > 0
    if not np.all(keep_mask):
        candidates = [c for c, k in zip(candidates, keep_mask) if k]
        coverage_matrix = coverage_matrix[keep_mask]
    int_pairs, int_penalties = build_interference_pairs(candidates)

    # ── Extreme: minimum cost (ε=∞) ────────────────────────────────────
    result_min_cost, _, _ = solve_with_lazy_interference(
        candidates, coverage_matrix, int_pairs, int_penalties,
        epsilon=1e9, hard_coverage=True, require_all_tower_types=True,
        mutually_exclusive_locations=True, time_limit=time_limit,
    )
    eps_max = result_min_cost.total_interference
    cost_at_min_cost = result_min_cost.total_cost

    # ── Heuristic: find a lower bound on ε via α=0 (interference-only) ──
    long_limit = max((time_limit or 30.0) * 3.0, 90.0)
    try:
        result_min_int, _, _ = solve_with_lazy_interference(
            candidates, coverage_matrix, int_pairs, int_penalties,
            alpha=0.0, hard_coverage=True, require_all_tower_types=True,
            mutually_exclusive_locations=True, time_limit=long_limit,
        )
        eps_min = result_min_int.total_interference
    except RuntimeError:
        eps_min = 0.05  # fallback lower bound

    # ── Binary search: find the ε threshold where cost jumps ───────────
    # Find the lowest ε that still achieves within 5% of min cost
    lo, hi = eps_min, eps_max
    threshold_eps: float = eps_min  # ε below which cost increases
    low_interf_point: ParetoPoint | None = None
    high_cost_points: list[ParetoPoint] = []

    for iteration in range(12):
        mid = (lo + hi) / 2.0
        try:
            result, _, _ = solve_with_lazy_interference(
                candidates, coverage_matrix, int_pairs, int_penalties,
                epsilon=float(mid), hard_coverage=True,
                require_all_tower_types=True, mutually_exclusive_locations=True,
                time_limit=time_limit,
            )
        except RuntimeError:
            lo = mid
            continue

        if result.coverage_ratio < coverage_requirement or result.selected_indices.size == 0:
            lo = mid
            continue

        pt = ParetoPoint(
            epsilon=float(mid), cost=result.total_cost,
            interference=result.total_interference,
            towers=len(result.selected_candidates),
            coverage_ratio=result.coverage_ratio,
            solver_status=result.message,
        )

        if result.total_cost <= cost_at_min_cost * 1.05:
            # Still near min-cost — tighten ε
            low_interf_point = pt
            threshold_eps = mid
            hi = mid
        else:
            # Cost increased — loosen ε, record this point
            lo = mid
            high_cost_points.append(pt)

    # ── Build the front ────────────────────────────────────────────────
    front: list[ParetoPoint] = []

    # Point(s) above threshold: higher cost, lower interference
    for pt in high_cost_points:
        if pt.interference < threshold_eps:
            front.append(pt)

    # Best min-cost + low-interference point near the threshold
    if low_interf_point is not None:
        front.append(low_interf_point)

    # Sweep a few well-spaced ε values between threshold and max
    # to capture the min-cost-at-various-interference region
    if low_interf_point is not None:
        mid_eps_vals = np.linspace(
            low_interf_point.interference + 0.05,
            eps_max - 0.02,
            min(num_steps, 8),
        )
        for eps in mid_eps_vals:
            if eps >= eps_max:
                continue
            try:
                result, _, _ = solve_with_lazy_interference(
                    candidates, coverage_matrix, int_pairs, int_penalties,
                    epsilon=float(eps), hard_coverage=True,
                    require_all_tower_types=True, mutually_exclusive_locations=True,
                    time_limit=time_limit,
                )
            except RuntimeError:
                continue
            if result.coverage_ratio < coverage_requirement:
                continue
            pt = ParetoPoint(
                epsilon=float(eps), cost=result.total_cost,
                interference=result.total_interference,
                towers=len(result.selected_candidates),
                coverage_ratio=result.coverage_ratio,
                solver_status=result.message,
            )
            # Only keep if it's different from existing points
            if not any(abs(p.cost - pt.cost) < 0.5 and abs(p.interference - pt.interference) < 0.02 for p in front):
                front.append(pt)

    # Include the pure min-cost extreme
    front.append(ParetoPoint(
        epsilon=eps_max, cost=cost_at_min_cost, interference=eps_max,
        towers=len(result_min_cost.selected_candidates),
        coverage_ratio=result_min_cost.coverage_ratio,
        solver_status=result_min_cost.message,
    ))

    return _deduplicate_front(front)


def _deduplicate_front(front: list[ParetoPoint]) -> list[ParetoPoint]:
    """Remove dominated and near-duplicate points.

    A point A dominates B if A has lower cost AND lower (or equal) interference.
    The Pareto front should only contain mutually non-dominating points.
    """
    if len(front) <= 1:
        return front

    # Sort by interference ascending
    sorted_front = sorted(front, key=lambda p: p.interference)

    # Filter: each point must have strictly lower cost than all points
    # with lower interference (otherwise it's dominated)
    result: list[ParetoPoint] = []
    best_cost_so_far = float("inf")

    for p in sorted_front:
        # A point is non-dominated iff its cost is strictly lower than
        # any point with less interference
        if p.cost < best_cost_so_far - 0.1:
            result.append(p)
            best_cost_so_far = min(best_cost_so_far, p.cost)

    return result


def pareto_front_summary(front: list[ParetoPoint]) -> str:
    """Return a human-readable summary of the Pareto front."""
    if not front:
        return "No Pareto-optimal points found."

    lines = [
        f"Pareto front: {len(front)} non-dominated points",
        f"Interference range: {front[0].interference:.4f} → {front[-1].interference:.4f}",
        f"Cost range:         {front[-1].cost:.2f} → {front[0].cost:.2f}",
        f"Tower range:        {min(p.towers for p in front)} → {max(p.towers for p in front)}",
        "",
        f"{'ε':>8s}  {'Cost':>8s}  {'Interf.':>8s}  {'Towers':>6s}  {'Status':<12s}",
        "-" * 60,
    ]

    for p in front:
        lines.append(
            f"{p.epsilon:>8.4f}  {p.cost:>8.4f}  {p.interference:>8.4f}  "
            f"{p.towers:>6d}  {p.solver_status:<12s}"
        )

    return "\n".join(lines)
