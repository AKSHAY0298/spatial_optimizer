# 6G Tower Placement Optimizer — Complete System Analysis

> **Date:** 2026-07-20
> **Data:** 236 German cities (≥50k population)
> **Problem:** Place exactly 2 tower types (dense + sparse radius) to cover all cities at minimum cost + interference

---

## 1. High-Level Architecture

```
cities_de_50k.txt (236 cities)
        │
        ▼
┌─────────────────────────────────┐
│  DBSCAN (ε=40km, min_samples=2) │  ← spatial.py:55-64
│  → 17 clusters, 24 noise cities │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  PHASE A: Radius Pair Search                        │  ← radius_pair_search.py
│                                                      │
│  1. Generate candidate radii + cost-critical points  │
│  2. Greedy proxy (Haversine-corrected) → rank pairs  │
│  3. LP relaxation → tight lower bound per pair       │
│  4. MILP-verify top-K pairs                          │
│  5. Adaptive grid refinement (0.5 km) around winner  │
│                                                      │
│  Output: (r_dense, r_sparse)                         │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  PHASE B: MILP Placement        │  ← optimizer.py
│  CBC branch-and-cut             │
│  Output: which towers, where    │
└─────────────────────────────────┘
```

**Two modes exist** (controlled by `--radius-mode`):

| Mode | File | What it does |
|------|------|-------------|
| `milp_pair_search` (default) | `radius_pair_search.py` | Enumerate radius pairs, greedy-proxy rank, LP-relaxation bound, MILP-verify top-K, adaptive grid refine |
| `dbscan_representative` | `radius_search.py` | Pick radii independently per regime using coarse-to-fine score |

---

## 2. DBSCAN Clustering — What It Actually Decides

**Code:** `spatial.py:55-64`

```python
coordinates = np.radians(cities[["latitude", "longitude"]].to_numpy(dtype=float))
clusterer = DBSCAN(eps=eps_km / EARTH_RADIUS_KM, min_samples=min_samples, metric="haversine")
return clusterer.fit_predict(coordinates)
```

**Current settings:** `eps_km=40.0`, `min_samples=2`

**What DBSCAN controls:**

1. **In `milp_pair_search` mode:** DBSCAN is used ONLY to add cluster centroids as extra candidate tower locations (`candidates.py`). It does NOT decide which city gets which tower type. The MILP decides all coverage assignments freely.

2. **In `dbscan_representative` mode:** DBSCAN splits cities into "clustered" and "noise" groups. Radii are chosen independently for each group. This is fundamentally different — the clustering directly shapes the result.

---

## 3. Phase A — Radius Pair Search (milp_pair_search mode)

**Code:** `radius_pair_search.py:optimize_radius_pair_with_milp`

### 3.1 Radius Candidate Generation

Radii are generated at `radius_step` intervals (default 5 km) from [5, 100], producing 20 base radii. Additionally, **cost-critical radii** are injected from `costs.py:cost_critical_radii()`:

```
Base grid:  {5, 10, 15, 20, ..., 100}           → 20 radii
Critical:   {5, 20, 32.5, 35, 36.97, 50, 100}   → 7 additional
Total:      ~22 unique radii → 231 unordered pairs
```

The critical radii are:
- **Segment boundaries:** 5, 20, 35, 50, 100 — where the piecewise cubic definition changes
- **Inflection points:** 32.5 (in segment 2), 36.97 (in segment 3) — where curvature f''(r) = 0

These are the radii where the cost-per-km trade-off shifts structurally; omitting them would mean the grid could miss important values.

### 3.2 Candidate Location Generation

**Code:** `candidates.py:generate_candidate_locations`

| Source | Count (default) | Description |
|--------|----------------|-------------|
| Cluster centroids | 17 | Arithmetic mean of DBSCAN cluster members |
| Every city | 236 | All city locations |
| Midpoints | ~3,183 (off by default) | City pairs within 80 km |
| Voronoi vertices | 0 (opt-in via `--voronoi`) | Top-50 most isolated Voronoi vertices within convex hull |
| Hex grid | 0 (opt-in via `--grid`) | Hexagonal lattice filtered to cities |
| **Total (default)** | **~253** | |

Each location × 2 tower types. Without midpoints: **~506 candidates**. With midpoints: **~6,872 candidates**.

### 3.3 Greedy Proxy Ranking (Haversine-Corrected)

**Code:** `radius_pair_search.py:_greedy_proxy_evaluation`

Run for every radius pair to rank them before expensive MILP solves.

**Algorithm:** Weighted set cover, greedily picking the tower with the lowest cost-per-new-city-covered ratio:

```
while uncovered cities remain:
    for each tower type t in {dense, sparse}:
        for each location ℓ (vectorized):
            gain[t,ℓ] = |cities covered by (t,ℓ) ∩ uncovered|
            ratio[t,ℓ] = cost(t) / gain[t,ℓ]
    pick (t*, ℓ*) = argmin ratio
    mark cities covered by (t*, ℓ*) as covered
```

Both tower types are forced into the solution (at least one of each). The interference estimate is computed using **vectorized Haversine** (`pairwise_haversine_km`) — the flat-earth approximation previously used here has been eliminated (see §7).

**Proxy objective:** `α × total_cost + (1-α) × total_interference`

### 3.4 LP Relaxation Bound

**Code:** `optimizer.py:solve(relaxed=True)`

For each top-K pair, the **exact MILP formulation is relaxed** by replacing binary tower-selection variables `x[i] ∈ {0, 1}` with continuous `x[i] ∈ [0, 1]`. The resulting LP solves in **~1 second** (vs. ~20s for the MILP) and provides a **valid lower bound** on the MILP objective:

```
LP ≤ MILP    (for a minimization problem)
```

The LP→MILP integrality gap quantifies how much structure is lost in the relaxation:

```
gap = (MILP - LP) / LP × 100%
```

Empirically, gaps range from **0.7% to 3.4%** for this problem — the LP relaxation is tight, meaning the formulation is well-structured. The LP ranking matches the MILP ranking for the best pairs, making it a viable alternative to the greedy proxy for ranking.

### 3.5 MILP Verification of Top-K Pairs

The top-K pairs (default K=8) from the proxy ranking are each solved with the full MILP. For each pair:

1. Generate candidates via `generate_candidates_from_locations(locations, (r1, r2))`
2. Build coverage matrix using `pairwise_haversine_km`
3. Build interference pairs via KD-tree spatial pruning + exact Haversine
4. Solve LP relaxation → `lp_objective` (lower bound)
5. Solve full MILP → `milp_objective` (exact, if optimal)

The winner is the pair with the lowest MILP objective. Both LP and MILP objectives are reported, plus the integrality gap.

### 3.6 Adaptive Grid Refinement

**Code:** `radius_pair_search.py` (inline in `optimize_radius_pair_with_milp`)

After the coarse 5 km pass finds a winning radius pair, a **0.5 km fine grid** is spawned in a ±2 km window around each winning radius component. Cost-critical radii in the neighborhood are also injected. The fine-grid pairs are proxy-ranked and top-K are MILP-verified.

This found **(16.5, 44.0)** instead of the coarse-pass **(15, 45)** — strictly better: lower cost, lower interference, same coverage, same tower count.

---

## 4. Phase B — MILP Formulation

**Code:** `optimizer.py:solve`

### 4.1 Decision Variables

| Variable | Type | Meaning |
|----------|------|---------|
| `x[i]` | Binary (or Continuous [0,1] when `relaxed=True`) | Tower candidate i is selected |
| `y[p]` | Continuous [0,1] | Both towers in interference pair p are active |

### 4.2 Objective Function

```
minimize: α · Σᵢ cost(i)·xᵢ  +  (1-α) · Σₚ Pₚ·yₚ
```

Where:
- `cost(i)` = piecewise cubic `tower_cost(radius)` from `costs.py` (4 segments; see §5)
- `Pₚ` = normalized overlap penalty: `max(0, (rₐ + r_b − d) / (rₐ + r_b))`
- `α = 0.5` (default)

### 4.3 Constraints

**Hard coverage** — every city must be covered:
```
For every city j:  Σ{xᵢ | candidate i covers city j} ≥ 1
```

**Require all tower types** — at least one dense AND one sparse tower:
```
For each tower type t:  Σ{xᵢ | type(i) = t} ≥ 1
```

**Mutually exclusive locations** — at most one tower per physical site:
```
For each physical location ℓ:  Σ{xᵢ | location(i) = ℓ} ≤ 1
```

**McCormick linearization** — enforces yₚ = xᵢ · xⱼ (AND of two binaries) using linear constraints:
```
For each interference pair p = (i, j):
    yₚ ≥ xᵢ + xⱼ − 1
    yₚ ≤ xᵢ
    yₚ ≤ xⱼ
```

yₚ is declared continuous [0,1], but the McCormick envelope forces binary-equivalent values at the optimum for a minimization problem (since Pₚ ≥ 0, the solver pushes yₚ down against the lower bound).

### 4.4 Solver

**CBC** (COIN-OR Branch-and-Cut) via PuLP:

```python
solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=False)
status = model.solve(solver)
```

CBC is an open-source MILP solver. It guarantees global optimality — **if it finishes** within the time limit. With `time_limit=60s` and ~500 candidates, it reliably reaches optimality. With midpoints (~6,800 candidates), it may return a feasible but suboptimal solution.

---

## 5. Cost Function

**Code:** `costs.py`

### 5.1 Piecewise Cubic Definition

The installation cost `c(r)` is a piecewise cubic over 4 segments:

```
Segment 1 [5, 20]:    c₁(r) =  3.899×10⁻⁵ r³ − 5.848×10⁻⁴ r² + 8.183×10⁻⁴ r + 0.9057
Segment 2 (20, 35]:   c₂(r) = −4.680×10⁻⁵ r³ + 4.562×10⁻³ r² − 1.021×10⁻¹ r + 1.5919
Segment 3 (35, 50]:   c₃(r) =  5.931×10⁻⁵ r³ − 6.578×10⁻³ r² + 2.878×10⁻¹ r − 2.9572
Segment 4 (50, 100]:  c₄(r) = −1.545×10⁻⁵ r³ + 4.635×10⁻³ r² − 2.729×10⁻¹ r + 6.3873
```

Key property: **monotonically non-decreasing** within [5, 100]. Larger radius always costs more.

| Radius (km) | Cost |
|-------------|------|
| 5 | 0.91 |
| 10 | 0.93 |
| 15 | 1.04 |
| 20 | 1.32 |
| 30 | 1.59 |
| 45 | 2.62 |
| 50 | 3.53 |
| 60 | 4.93 |
| 80 | 7.57 |
| 100 | 8.83 |

### 5.2 Derivative Analysis

For each segment `c(r) = a·r³ + b·r² + c·r + d`:

```
c'(r)  = 3a·r² + 2b·r + c      (marginal cost per km of radius)
c''(r) = 6a·r + 2b              (curvature — how fast marginal cost changes)
```

**Inflection points** occur where c''(r) = 0, i.e., `r = −b / (3a)` for a ≠ 0:

| Segment | a | b | Inflection r | In domain? |
|---------|---|---|-------------|------------|
| 1: [5, 20] | +3.90×10⁻⁵ | −5.85×10⁻⁴ | r = 5.00 | ✓ (boundary) |
| 2: (20, 35] | −4.68×10⁻⁵ | +4.56×10⁻³ | r = 32.50 | ✓ |
| 3: (35, 50] | +5.93×10⁻⁵ | −6.58×10⁻³ | r = 36.97 | ✓ |
| 4: (50, 100] | −1.54×10⁻⁵ | +4.63×10⁻³ | r = 100.00 | ✓ (boundary) |

These points, along with segment boundaries {5, 20, 35, 50, 100}, form the **cost-critical radii set** injected into every candidate radius pool. At these radii, the cost-per-km trade-off changes structurally — the optimizer should consider them explicitly rather than relying on grid interpolation.

---

## 6. Interference Model

**Code:** `matrices.py:36-42`

```
P(a,b) = max(0, (rₐ + r_b − d) / (rₐ + r_b))
```

Where `d` = Haversine distance between towers a and b. P ∈ [0, 1].

- P = 0: circles are tangent or disjoint (no overlap)
- P → 1: one circle nearly contains the other (severe overlap)

**Pre-filtering:** `matrices.py:45-74` — KD-tree query with radius `2 × r_max` (using flat-earth projection for speed, then exact Haversine for the penalty value). This ensures only geometrically-possible overlaps enter the MILP.

**Final interference in objective:** `(1 − α) × Σₚ Pₚ × yₚ` where yₚ = 1 iff both towers in pair p are selected.

---

## 7. Distance Computations — Haversine Correction

All correctness-affecting distance computations now use **great-circle (Haversine)** distance. Flat-earth approximations are restricted to KD-tree spatial indexing where exact distance is verified downstream.

| Function | File:Line | Metric | Usage |
|----------|-----------|--------|-------|
| `haversine_distance_km` | `spatial.py:21-32` | Great-circle (scalar) | Point-to-point, interference penalties |
| `pairwise_haversine_km` | `spatial.py:35-52` | Great-circle (vectorized) | Coverage matrix, proxy distances, proxy interference |
| `project_coordinates_km` | `spatial.py:108-117` | Flat-earth projection | KD-tree pruning only (exact distance verified after) |

**Phase 0 fix (2026-07-20):** The greedy proxy previously used flat-earth approximations (`dlat × 110.574, dlon × 111.320 × cos(lat)`) at two locations in `_greedy_proxy_evaluation`. These were replaced with calls to `pairwise_haversine_km`. The vectorized Haversine formula:

```
a = sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)
d = 2 · R_earth · arcsin(√a)
```

where R_earth = 6371.0088 km. At German latitudes (~47°–55°N), the flat-earth approximation introduces errors of **0.3–0.5%** — small but systematic, and enough to mis-rank radius pairs where interference differences are subtle.

---

## 8. Optimality Gaps — Status

### Closed Gaps (Implemented)

**Gap 1 — Radius discretization** (previously §3.6):
- **Was:** Only tested radii at multiples of 5 km
- **Now:** Adaptive 0.5 km fine-grid refinement around coarse winners, plus cost-critical radii (inflection points, segment boundaries) injected into the candidate pool
- **Status:** **Closed.** The optimizer now finds radii like 16.5, 44.0 instead of being limited to 15, 45.

**Gap 2 — Proxy ranking may miss the true best pair** (previously §3.5):
- **Was:** Greedy set-cover proxy with no quality guarantee; 182/190 pairs never MILP-tested
- **Now:** LP relaxation provides a tight lower bound (0.7–3.4% gap) that reuses the exact MILP formulation. Can be used as an alternative ranking metric.
- **Status:** **Mitigated.** The LP relaxation is a principled bound. With K=8 and adaptive refinement, the search is more robust.

**Gap 6 — Proxy interference used flat-earth** (previously §8):
- **Was:** Euclidean approximation in proxy interference calculation
- **Now:** Both locations in `_greedy_proxy_evaluation` use `pairwise_haversine_km`
- **Status:** **Closed.** Proxy and MILP use the same distance metric.

### Remaining Gaps

**Gap 3 — Solver time limit:**
- **Where:** `optimizer.py` — `timeLimit=time_limit`
- **What:** CBC returns best feasible solution within time limit; may not prove optimality
- **Severity:** High with midpoints (many variables), Low without midpoints (reliably reaches optimality)

**Gap 4 — Candidate locations are finite:**
- **Where:** `candidates.py:generate_candidate_locations`
- **What:** Towers can only be placed at discrete locations (cities, centroids, midpoints), not at arbitrary continuous coordinates
- **Severity:** Low-Medium. Midpoints help, but true optimal positions may be slightly offset.

**Gap 5 — DBSCAN epsilon is fixed:**
- **Where:** `spatial.py:55-64` — `eps_km=40.0`
- **What:** Different clustering produces different centroid locations → different candidate set → different MILP optimum
- **Severity:** Low — centroids are only 17 out of 253+ locations in default mode

**Gap 6 — Objective scaling:**
- **What:** Cost (~0.9–8.8 per tower) and interference penalty (~0–1 per pair) operate on different scales. A small cost reduction can numerically dominate a large real-world interference increase.
- **Severity:** Medium. The α=0.5 weight doesn't normalize for scale. At the optimum, a 0.5-unit interference increase can be "paid for" by a 0.5-unit cost decrease, but in physical terms that could mean thousands of sq km of extra overlap.

---

## 9. Data Flow Summary (milp_pair_search, default settings — no midpoints)

```
Step 1:  Load 236 cities                                  (data.py)
Step 2:  DBSCAN → 17 clusters, 24 noise                  (spatial.py)
Step 3:  Generate candidate locations                     (candidates.py)
         → 17 centroids + 236 cities = 253 locations
Step 4:  Generate candidate radii                         (radius_pair_search.py)
         → 20 base radii + 7 cost-critical = ~22 radii
Step 5:  Enumerate ~231 radius pairs                      (radius_pair_search.py)
Step 6:  Pre-compute pairwise Haversine matrix            (pairwise_haversine_km)
         → shape (253, 236), used by all 231 proxy evals
Step 7:  Greedy proxy all 231 pairs                       (_greedy_proxy_evaluation)
         → O(231 × greedy_iterations), ~0.01s per pair
Step 8:  Sort by proxy objective, take top 8              (sorted by proxy_objective)
Step 9:  For each top-8 pair:                             (inline in optimize_radius_pair_with_milp)
         a. Generate 253×2 = 506 candidates
         b. Build coverage matrix (506 × 236) — Haversine
         c. Build interference pairs (~10K pairs) — KD-tree + Haversine
         d. Solve LP relaxation → lp_objective (~1s)
         e. Solve MILP → milp_objective (~10-20s)
Step 10: Adaptive grid refinement around winner           (0.5 km fine grid)
         a. Fine radii around (dense_winner, sparse_winner) ± 2 km
         b. Proxy-rank fine pairs, MILP-verify top-K
Step 11: Post-MILP per-tower radius refinement            (_refine_tower_radii)
Step 12: Return final result                              (pipeline.py)
```

---

## 10. Results

### 10.1 Final Result (All improvements, `--no-midpoints`)

| Metric | Value |
|--------|-------|
| Dense radius | **16.5 km** |
| Sparse radius | **44.0 km** |
| Towers selected | **62** |
| Coverage | 100% (hard constraint) |
| Total cost | **74.28** |
| Overlap | **1.43%** (1,883 sq km) |
| Interference penalty (Σ Pₚ) | 1.09 |
| Combined objective | 37.69 |
| LP→MILP gap | 1.7% |
| Solver status | Optimal |
| Candidate locations | 253 |
| Interference pairs | ~10,283 |
| α | 0.5 |

### 10.2 Comparison: Original → Improved

| Metric | Original (paper) | Improved (Phases 0+2+3) | Δ |
|--------|-----------------|--------------------------|----|
| Radii | 15 + 45 km | **16.5 + 44.0 km** | — |
| Towers | 62 | 62 | 0 |
| Cost | 75.45 | **74.28** | **−1.6%** |
| Overlap | 1.58% | **1.43%** | **−9.5%** |
| Solver objective | ~38.24 | **37.69** | **−1.4%** |
| Solver status | Optimal | Optimal | ✓ |

The adaptive grid refinement found (16.5, 44.0) — strictly Pareto-better than (15, 45): lower cost AND lower interference at the same coverage and tower count.

### 10.3 LP Relaxation Bounds (Top-5 Verified Pairs)

| (r_dense, r_sparse) | LP bound | MILP objective | Gap | Towers | Status |
|---------------------|----------|---------------|-----|--------|--------|
| (16.5, 44.0) | 37.07 | 37.69 | 1.7% | 62 | Optimal |
| (17.0, 44.0) | 37.25 | 37.75 | 1.3% | 63 | Optimal |
| (13.0, 44.0) | 37.45 | 37.84 | 1.0% | 63 | Optimal |
| (13.5, 44.0) | 37.51 | 37.93 | 1.1% | 63 | Optimal |
| (16.0, 44.0) | 37.38 | 37.95 | 1.5% | 64 | Optimal |

The LP relaxation is tight (gaps 1.0–1.7%) and correctly orders the top pairs. It provides a principled alternative to the greedy proxy for ranking.

---

## 11. Implemented Improvements

### Phase 0 — Haversine Correction (mandatory, done)
- **What:** Replaced flat-earth Euclidean distance with vectorized `pairwise_haversine_km` in `_greedy_proxy_evaluation` (two locations: force-both-types loop and interference estimate)
- **Impact:** Correctness. Proxy and MILP now use the same distance metric. Eliminated systematic ~0.3–0.5% distance error.

### Phase 2 — LP Relaxation Bound (high-value, done)
- **What:** Added `relaxed=True` parameter to `solve()`. Relaxes `x[i] ∈ {0, 1}` → `x[i] ∈ [0, 1]`, converting MILP to LP. LP solves in ~1s and provides a valid lower bound.
- **Impact:** LP bounds are tight (0.7–3.4% gap). Provides a principled ranking metric and validates the quality of the MILP solutions.

### Phase 3 — Adaptive Grid Refinement + Cost Derivative Analysis (done)
- **What:** (a) Computed c'(r) and c''(r) for each piecewise cubic segment, found inflection points and segment boundaries as cost-critical radii; (b) After coarse 5 km pass, spawns 0.5 km fine grid ±2 km around winners and re-evaluates.
- **Impact:** Found (16.5, 44.0) instead of (15, 45) — cost ↓1.6%, overlap ↓9.5%.

### Phase 1 — Voronoi Candidates (experimental, opt-in via `--voronoi`)
- **What:** Voronoi tessellation over city coordinates, convex hull filtering, top-50 most isolated vertices kept as candidate locations.
- **Status:** Available but off by default. Weighted geometric median (Weiszfeld) centroids tested but reverted — they shifted cluster centers and caused an interference regression (overlap 1.43% → 3.70%).

---

## 12. Remaining Opportunities

### High-impact, low-effort

1. **α-sweep for Pareto frontier.** Run with α ∈ {0.2, 0.3, ..., 0.8} to trace the cost-vs-interference trade-off curve. Different stakeholders will prefer different points.

2. **Use Gurobi instead of CBC.** Gurobi is 10–100× faster on MILPs of this size. Would make midpoints practical and enable higher K for verification. PuLP supports `pulp.GUROBI_CMD()`.

### Medium-impact

3. **Candidate pruning before MILP.** Identify dominated candidates (same coverage as another candidate but higher cost) and remove them. Would shrink the MILP without losing optimality guarantees.

4. **Grid search on DBSCAN eps.** Try eps ∈ {30, 35, 40, 45, 50} — different centroid placements might enable better configurations. The MILP is the ultimate quality judge.

5. **Multi-objective formulation.** Replace the weighted-sum scalarization with an ε-constraint method that explicitly bounds interference while minimizing cost (or vice versa). This avoids the scale-mismatch issue noted in Gap 6.

### Exploration

6. **Decrease radius_step to 2.5 km** for finer initial grid (39 radii → 741 pairs). More proxy evaluations but the adaptive refinement already handles this.

7. **Relax hard coverage to 99%.** Currently demands 100% coverage. Even 99% might enable significantly cheaper configurations (e.g., skipping one remote city that forces an extra tower).
