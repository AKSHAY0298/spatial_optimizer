from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pulp

from .candidates import TowerCandidate


@dataclass(frozen=True)
class OptimizationResult:
    status: int
    message: str
    objective_value: float
    total_cost: float
    total_interference: float
    coverage_ratio: float
    selected_indices: np.ndarray
    selected_candidates: list[TowerCandidate]
    pair_values: np.ndarray


def solve(
    candidates: list[TowerCandidate],
    coverage_matrix: np.ndarray,
    interference_pairs: list[tuple[int, int]],
    interference_penalties: np.ndarray,
    *,
    alpha: float = 0.5,
    beta: float = 0.2,
    hard_coverage: bool = True,
    require_all_tower_types: bool = True,
    mutually_exclusive_locations: bool = True,
    time_limit: float | None = 60.0,
    relaxed: bool = False,
) -> OptimizationResult:
    """Solve the linear tower placement model using PuLP.

    Parameters
    ----------
    relaxed : bool
        If True, relax binary tower-selection variables to continuous [0, 1].
        This converts the MILP into an LP that solves much faster and provides
        a valid lower bound on the MILP objective.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within the closed interval [0, 1].")

    candidate_count = len(candidates)
    pair_count = len(interference_pairs)
    city_count = int(coverage_matrix.shape[1])

    model = pulp.LpProblem(
        "Tower_Placement" if not relaxed else "Tower_Placement_LP_Relaxation",
        pulp.LpMinimize,
    )

    if relaxed:
        x = [pulp.LpVariable(f"x_{i}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous) for i in range(candidate_count)]
    else:
        x = [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(candidate_count)]
    y = [pulp.LpVariable(f"y_{p}", cat=pulp.LpContinuous, lowBound=0.0, upBound=1.0) for p in range(pair_count)]
    z = [] if hard_coverage else [pulp.LpVariable(f"z_{j}", cat=pulp.LpBinary) for j in range(city_count)]

    c_max = max(c.cost for c in candidates) if candidates else 1.0
    C_bar = alpha * c_max

    objective_expr = []
    for i in range(candidate_count):
        objective_expr.append(alpha * candidates[i].cost * x[i])
    for p in range(pair_count):
        objective_expr.append((1.0 - alpha) * interference_penalties[p] * y[p])
    if not hard_coverage:
        for j in range(city_count):
            objective_expr.append(-beta * C_bar * z[j])

    model += pulp.lpSum(objective_expr)

    for j in range(city_count):
        covering_candidates = np.flatnonzero(coverage_matrix[:, j])
        if hard_coverage:
            if covering_candidates.size == 0:
                raise ValueError(f"City index {j} cannot be covered by any generated candidate.")
            model += pulp.lpSum([x[i] for i in covering_candidates]) >= 1
        else:
            model += z[j] <= pulp.lpSum([x[i] for i in covering_candidates])

    if require_all_tower_types:
        tower_types = sorted({candidate.tower_type for candidate in candidates})
        if len(tower_types) < 2:
            raise ValueError("At least two tower types are required, but fewer were generated.")
        for tower_type in tower_types:
            type_indices = [i for i, candidate in enumerate(candidates) if candidate.tower_type == tower_type]
            model += pulp.lpSum([x[i] for i in type_indices]) >= 1

    if mutually_exclusive_locations:
        location_groups: dict[str, list[int]] = defaultdict(list)
        for i, candidate in enumerate(candidates):
            location_id = candidate.location_id or f"{candidate.latitude:.7f},{candidate.longitude:.7f}"
            location_groups[location_id].append(i)
        for indices in location_groups.values():
            if len(indices) > 1:
                model += pulp.lpSum([x[i] for i in indices]) <= 1

    for p, (left_index, right_index) in enumerate(interference_pairs):
        model += y[p] >= x[left_index] + x[right_index] - 1
        model += y[p] <= x[left_index]
        model += y[p] <= x[right_index]

    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=False)
    status = model.solve(solver)

    if candidate_count > 0 and x[0].varValue is None:
        raise RuntimeError(f"MILP solver did not return a solution: {pulp.LpStatus[status]}")

    solution_x = np.array([v.varValue if v.varValue is not None else 0.0 for v in x], dtype=float)
    solution_y = np.array([v.varValue if v.varValue is not None else 0.0 for v in y], dtype=float)

    selected_indices = np.flatnonzero(solution_x > 0.5)
    selected_candidates = [candidates[index] for index in selected_indices]
    pair_values = solution_y

    if selected_indices.size == 0:
        coverage_ratio = 0.0
    else:
        coverage_ratio = float(coverage_matrix[selected_indices].any(axis=0).mean())

    total_cost = float(sum([candidate.cost for candidate in selected_candidates]))
    total_interference = float(np.dot(interference_penalties, pair_values)) if pair_count else 0.0
    objective_value = pulp.value(model.objective)
    objective_value = float(objective_value) if objective_value is not None else 0.0

    return OptimizationResult(
        status=int(status),
        message=str(pulp.LpStatus[status]),
        objective_value=objective_value,
        total_cost=total_cost,
        total_interference=total_interference,
        coverage_ratio=coverage_ratio,
        selected_indices=selected_indices,
        selected_candidates=selected_candidates,
        pair_values=pair_values,
    )
