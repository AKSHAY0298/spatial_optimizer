from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pulp

from .candidates import TowerCandidate


_SOLVER_CACHE: dict[str, bool | None] = {"gurobi": None, "highs": None}


def _make_solver(time_limit: float | None, prefer: str | None = None):
    """Return the best available PuLP solver.

    Priority (default): Gurobi > HiGHS > CBC.
    Pass ``prefer="cbc"`` to force CBC (useful for comparisons).
    """
    global _SOLVER_CACHE

    # ── Honour explicit preference ──────────────────────────────────────
    if prefer == "cbc":
        return pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=False)
    if prefer == "gurobi":
        try:
            g = pulp.GUROBI(msg=False)
            if g.available():
                return pulp.GUROBI(timeLimit=time_limit, msg=False)
        except Exception:
            pass
        raise RuntimeError("Gurobi was requested (--solver gurobi) but is not available.")


    # ── Gurobi ───────────────────────────────────────────────────────────
    if _SOLVER_CACHE["gurobi"] is None:
        try:
            _SOLVER_CACHE["gurobi"] = pulp.GUROBI(msg=False).available()
        except Exception:
            _SOLVER_CACHE["gurobi"] = False
    if _SOLVER_CACHE["gurobi"]:
        return pulp.GUROBI(timeLimit=time_limit, msg=False)

    # ── HiGHS ────────────────────────────────────────────────────────────
    if _SOLVER_CACHE["highs"] is None:
        try:
            _SOLVER_CACHE["highs"] = pulp.HiGHS(msg=False).available()
        except Exception:
            _SOLVER_CACHE["highs"] = False
    if _SOLVER_CACHE["highs"]:
        return pulp.HiGHS(timeLimit=time_limit, msg=False)

    # ── CBC fallback ─────────────────────────────────────────────────────
    return pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=False)


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
    epsilon: float | None = None,
    solver: str | None = None,
) -> OptimizationResult:
    """Solve the linear tower placement model using PuLP.

    Two objective modes are supported:

    **Weighted-sum mode (default, epsilon=None):**
        minimize  α·Σ cost(i)·xᵢ  +  (1-α)·Σ Pₚ·yₚ

    **ε-constraint mode (epsilon is not None):**
        minimize  Σ cost(i)·xᵢ
        s.t.      Σ Pₚ·yₚ  ≤  ε

    The ε-constraint mode produces one point on the cost-vs-interference
    Pareto front. Sweeping ε traces the full trade-off curve.

    Parameters
    ----------
    relaxed : bool
        If True, relax binary tower-selection variables to continuous [0, 1].
    epsilon : float or None
        Interference budget. When set, the objective is pure cost minimization
        with interference constrained to ≤ epsilon. Alpha is ignored.
    """

    candidate_count = len(candidates)
    pair_count = len(interference_pairs)
    city_count = int(coverage_matrix.shape[1])
    use_epsilon = epsilon is not None

    if not use_epsilon and not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within the closed interval [0, 1].")

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

    # ── Objective ──────────────────────────────────────────────────────
    if use_epsilon:
        # Cost-only objective; interference is a constraint
        model += pulp.lpSum([candidates[i].cost * x[i] for i in range(candidate_count)])
    else:
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

    # ── Coverage constraints ───────────────────────────────────────────
    for j in range(city_count):
        covering_candidates = np.flatnonzero(coverage_matrix[:, j])
        if hard_coverage:
            if covering_candidates.size == 0:
                raise ValueError(f"City index {j} cannot be covered by any generated candidate.")
            model += pulp.lpSum([x[i] for i in covering_candidates]) >= 1
        else:
            model += z[j] <= pulp.lpSum([x[i] for i in covering_candidates])

    # ── Tower type requirements ────────────────────────────────────────
    if require_all_tower_types:
        tower_types = sorted({candidate.tower_type for candidate in candidates})
        if len(tower_types) < 2:
            raise ValueError("At least two tower types are required, but fewer were generated.")
        for tower_type in tower_types:
            type_indices = [i for i, candidate in enumerate(candidates) if candidate.tower_type == tower_type]
            model += pulp.lpSum([x[i] for i in type_indices]) >= 1

    # ── Mutual exclusion ───────────────────────────────────────────────
    if mutually_exclusive_locations:
        location_groups: dict[str, list[int]] = defaultdict(list)
        for i, candidate in enumerate(candidates):
            location_id = candidate.location_id or f"{candidate.latitude:.7f},{candidate.longitude:.7f}"
            location_groups[location_id].append(i)
        for indices in location_groups.values():
            if len(indices) > 1:
                model += pulp.lpSum([x[i] for i in indices]) <= 1

    # ── McCormick interference linearization ───────────────────────────
    # Only the lower bound is needed: y has a nonnegative objective (or budget)
    # coefficient and is minimized, so y = max(0, x_l + x_r - 1) at optimality.
    # The upper bounds y <= x_l and y <= x_r are redundant and only bloat CBC.
    for p, (left_index, right_index) in enumerate(interference_pairs):
        model += y[p] >= x[left_index] + x[right_index] - 1

    # ── Interference budget constraint (ε-constraint mode) ─────────────
    if use_epsilon and pair_count > 0:
        model += pulp.lpSum([interference_penalties[p] * y[p] for p in range(pair_count)]) <= epsilon

    # ── Solve ──────────────────────────────────────────────────────────
    solver = _make_solver(time_limit, prefer=solver)
    status = model.solve(solver)

    if candidate_count > 0 and x[0].varValue is None:
        raise RuntimeError(f"MILP solver did not return a solution: {pulp.LpStatus[status]}")

    solution_x = np.array([v.varValue if v.varValue is not None else 0.0 for v in x], dtype=float)
    solution_y = np.array([v.varValue if v.varValue is not None else 0.0 for v in y], dtype=float)

    selected_indices = np.flatnonzero(solution_x > 0.5)
    selected_candidates = [candidates[index] for index in selected_indices]
    if relaxed:
        pair_values = solution_y
    else:
        # Derive pair activation from the realized selection so reported
        # interference is exact even when the solver leaves slack in y.
        selected_mask = solution_x > 0.5
        pair_values = np.array(
            [1.0 if (selected_mask[li] and selected_mask[ri]) else 0.0 for li, ri in interference_pairs],
            dtype=float,
        )

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


def solve_with_lazy_interference(
    candidates: list[TowerCandidate],
    coverage_matrix: np.ndarray,
    interference_pairs: list[tuple[int, int]],
    interference_penalties: np.ndarray,
    *,
    alpha: float = 0.5,
    beta: float = 0.0,
    hard_coverage: bool = True,
    require_all_tower_types: bool = True,
    mutually_exclusive_locations: bool = True,
    time_limit: float | None = 60.0,
    epsilon: float | None = None,
    max_rounds: int = 30,
    solver: str | None = None,
) -> tuple[OptimizationResult, list[tuple[int, int]], np.ndarray]:
    """Solve the MILP with lazily generated interference pairs.

    Dense candidate menus can produce millions of overlapping pairs, but any
    optimal solution only selects a few dozen towers, so almost all y-variables
    are irrelevant. Instead of instantiating every pair up front, this routine:

      1. Solves the MILP with only the currently *active* pair set (initially
         empty), which is a relaxation of the full model.
      2. Finds pairs whose towers were both selected but are not yet in the
         model ("violated" pairs), activates them, and re-solves.
      3. Stops when the incumbent solution realizes no inactive pair — at that
         point the relaxation is exact for the incumbent, so it is optimal for
         the full model (up to the sub-solver's own optimality tolerance).

    Returns the final result plus the active pair subset and penalties that the
    result's ``pair_values`` are aligned with.
    """

    import dataclasses

    pair_array = np.asarray(interference_pairs, dtype=int).reshape(-1, 2)
    penalties = np.asarray(interference_penalties, dtype=float)
    active = np.zeros(len(pair_array), dtype=bool)

    result: OptimizationResult | None = None
    active_pairs: list[tuple[int, int]] = []
    active_penalties = np.zeros(0, dtype=float)
    violated = np.zeros(len(pair_array), dtype=bool)

    for _ in range(max_rounds):
        active_indices = np.flatnonzero(active)
        active_pairs = [tuple(pair) for pair in pair_array[active_indices]]
        active_penalties = penalties[active_indices]

        result = solve(
            candidates,
            coverage_matrix,
            active_pairs,
            active_penalties,
            alpha=alpha,
            beta=beta,
            hard_coverage=hard_coverage,
            require_all_tower_types=require_all_tower_types,
            mutually_exclusive_locations=mutually_exclusive_locations,
            time_limit=time_limit,
            relaxed=False,
            epsilon=epsilon,
            solver=solver,
        )

        if len(pair_array) == 0:
            break

        selected_mask = np.zeros(len(candidates), dtype=bool)
        selected_mask[result.selected_indices] = True
        left_selected = selected_mask[pair_array[:, 0]]
        right_selected = selected_mask[pair_array[:, 1]]
        violated = left_selected & right_selected & ~active
        if not violated.any():
            break
        active |= violated

    assert result is not None

    if violated.any():
        # Round cap hit before convergence: the model never charged some
        # realized pairs. Report the incumbent honestly by rebuilding the
        # metrics over active ∪ realized pairs.
        active |= violated
        active_indices = np.flatnonzero(active)
        active_pairs = [tuple(pair) for pair in pair_array[active_indices]]
        active_penalties = penalties[active_indices]
        selected_mask = np.zeros(len(candidates), dtype=bool)
        selected_mask[result.selected_indices] = True
        pair_values = (
            selected_mask[pair_array[active_indices, 0]]
            & selected_mask[pair_array[active_indices, 1]]
        ).astype(float)
        total_interference = float(np.dot(active_penalties, pair_values))
        objective_value = (
            result.total_cost
            if epsilon is not None
            else alpha * result.total_cost + (1.0 - alpha) * total_interference
        )
        result = dataclasses.replace(
            result,
            total_interference=total_interference,
            objective_value=objective_value,
            pair_values=pair_values,
        )

    return result, active_pairs, active_penalties
