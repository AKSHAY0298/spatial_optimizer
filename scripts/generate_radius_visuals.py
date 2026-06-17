"""
Generate presentation visuals for the radius selection logic.
Produces:
    1. radius_bounds_visual.png   — Combined NN/pairwise histograms with P10/P95 lines
    2. radius_bounds_dense_nn.png — Dense nearest-neighbor histogram
    3. radius_bounds_dense_pw.png  — Dense pairwise histogram
    4. radius_bounds_sparse_nn.png — Sparse nearest-neighbor histogram
    5. radius_bounds_sparse_pw.png — Sparse pairwise histogram
    6. coarse_fine_score.png       — Score(r) landscape with coarse/fine annotations
"""
from __future__ import annotations
import math, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Make the core package importable
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from core.data import load_cities
from core.spatial import cluster_cities, pairwise_haversine_km
from core.costs import tower_cost
from core.radius_search import (
    _representatives_for_dense_search,
    _representatives_for_sparse_search,
)

# ── Load data & cluster ─────────────────────────────────────────────────
cities = load_cities()
labels = cluster_cities(cities, eps_km=40.0, min_samples=2)
city_points = cities[["latitude", "longitude"]].to_numpy(dtype=float)

labeled = cities.copy()
labeled["cluster_label"] = labels
clustered_pts = labeled[labeled["cluster_label"] >= 0][["latitude", "longitude"]].to_numpy(dtype=float)
noise_pts     = labeled[labeled["cluster_label"] <  0][["latitude", "longitude"]].to_numpy(dtype=float)


# =====================================================================
# FIGURE 1 — Nearest-neighbor + pairwise distance histograms
# =====================================================================
def nn_and_pairwise(points, label):
    dists = pairwise_haversine_km(points)
    n = len(points)
    upper_tri = dists[np.triu_indices(n, k=1)]
    positive = upper_tri[upper_tri > 0]

    masked = np.where(dists > 0, dists, np.inf)
    nn = masked.min(axis=1)
    nn = nn[np.isfinite(nn)]
    return nn, positive

nn_dense, pw_dense = nn_and_pairwise(clustered_pts, "Dense")
nn_sparse, pw_sparse = nn_and_pairwise(noise_pts, "Sparse")

fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#1a1a2e")
for ax in axes.flat:
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444")

titles = [
    ("Dense: Nearest-Neighbor Distances", nn_dense, 10, "NN"),
    ("Dense: Pairwise Distances",         pw_dense, 95, "Pairwise"),
    ("Sparse: Nearest-Neighbor Distances", nn_sparse, 10, "NN"),
    ("Sparse: Pairwise Distances",         pw_sparse, 95, "Pairwise"),
]

colors = ["#00e676", "#00e676", "#ffab00", "#ffab00"]


def plot_bounds_panel(ax, title, data, pct, dtype):
    ax.hist(data, bins=40, color="#00e676" if dtype == "NN" else "#ffab00",
            alpha=0.65, edgecolor="white", linewidth=0.3)
    pval = float(np.percentile(data, pct))
    ax.axvline(pval, color="#ff1744", linewidth=2, linestyle="--",
               label=f"$P_{{{pct}}}$ = {pval:.1f} km")
    if dtype == "NN":
        bound = max(5, int(math.floor(pval / 2.0)))
        ax.axvline(bound, color="#00bcd4", linewidth=1.5, linestyle=":",
                   label=f"$r_{{\\min}}$ bound → {bound} km")
    else:
        bound = min(100, int(math.ceil(pval)))
        ax.axvline(bound, color="#00bcd4", linewidth=1.5, linestyle=":",
                   label=f"$r_{{\\max}}$ bound → {bound} km")
    ax.set_title(title, color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Distance (km)", color="white", fontsize=9)
    ax.set_ylabel("Count", color="white", fontsize=9)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
    return pval, bound


def save_bounds_panel(title, data, pct, dtype, filename):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444")
    plot_bounds_panel(ax, title, data, pct, dtype)
    fig.tight_layout()
    fig.savefig(filename, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

save_bounds_panel("Dense: Nearest-Neighbor Distances", nn_dense, 10, "NN",
                  "images/radius_bounds_dense_nn.png")
save_bounds_panel("Dense: Pairwise Distances", pw_dense, 95, "Pairwise",
                  "images/radius_bounds_dense_pw.png")
save_bounds_panel("Sparse: Nearest-Neighbor Distances", nn_sparse, 10, "NN",
                  "images/radius_bounds_sparse_nn.png")
save_bounds_panel("Sparse: Pairwise Distances", pw_sparse, 95, "Pairwise",
                  "images/radius_bounds_sparse_pw.png")

for idx, (title, data, pct, dtype) in enumerate(titles):
    ax = axes.flat[idx]
    plot_bounds_panel(ax, title, data, pct, dtype)

fig.suptitle("How Spatial Statistics Drive the Radius Bounds", color="white",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("images/radius_bounds_visual.png", dpi=180, facecolor=fig.get_facecolor())
plt.close()
print("✓ Saved images/radius_bounds_visual.png")
print("✓ Saved images/radius_bounds_dense_nn.png")
print("✓ Saved images/radius_bounds_dense_pw.png")
print("✓ Saved images/radius_bounds_sparse_nn.png")
print("✓ Saved images/radius_bounds_sparse_pw.png")


# =====================================================================
# FIGURE 2 — Score landscape (coarse-to-fine)
# =====================================================================
dense_reps  = _representatives_for_dense_search(cities, labels)
sparse_reps = _representatives_for_sparse_search(cities, labels)

def compute_scores(reps, all_cities, rmin, rmax):
    dists = pairwise_haversine_km(reps, all_cities)
    radii = list(range(rmin, rmax + 1))
    scores = []
    for r in radii:
        cov = (dists <= r).sum(axis=1)
        avg_cov = max(1.0, float(cov.mean()))
        scores.append(tower_cost(r) / avg_cov)
    return radii, scores

# Dense bounds
d10_dense = float(np.percentile(nn_dense, 10))
dense_lb = max(5, int(math.floor(d10_dense / 2.0)))
dense_ub = min(100, int(math.ceil(float(np.percentile(pw_dense, 95)))))
if dense_lb > dense_ub:
    dense_lb = dense_ub

# Sparse bounds
d10_sparse = float(np.percentile(nn_sparse, 10))
sparse_lb = max(5, int(math.floor(d10_sparse / 2.0)))
sparse_ub = min(100, int(math.ceil(float(np.percentile(pw_sparse, 95)))))
if sparse_lb > sparse_ub:
    sparse_lb = sparse_ub

dense_radii, dense_scores = compute_scores(dense_reps, city_points, dense_lb, dense_ub)
sparse_radii, sparse_scores = compute_scores(sparse_reps, city_points, sparse_lb, sparse_ub)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1a1a2e")
for ax in (ax1, ax2):
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444")

def plot_score_panel(ax, radii, scores, lb, ub, best_r, best_s, label, color):
    ax.plot(radii, scores, color=color, linewidth=1.5, alpha=0.8, label="Score(r)")

    # Coarse grid
    coarse = list(range(lb, ub + 1, 15))
    if coarse[-1] != ub:
        coarse.append(ub)
    coarse_scores_vals = [scores[r - lb] for r in coarse if lb <= r <= ub]
    coarse_valid = [r for r in coarse if lb <= r <= ub]
    ax.scatter(coarse_valid, coarse_scores_vals, color="#ff1744", s=50, zorder=5,
               edgecolors="white", linewidths=0.5, label="Coarse grid (15 km)")

    # Top-3 coarse
    ranked = sorted(zip(coarse_valid, coarse_scores_vals), key=lambda x: x[1])
    for i, (r, s) in enumerate(ranked[:3]):
        lo = max(lb, r - 14)
        hi = min(ub, r + 14)
        ax.axvspan(lo, hi, alpha=0.12, color=["#00e676", "#ffab00", "#00bcd4"][i],
                   label=f"Fine window #{i+1}" if i < 2 else None)

    # Best
    ax.axvline(best_r, color="#ff1744", linewidth=2, linestyle="--")
    ax.scatter([best_r], [best_s], color="#ff1744", s=120, zorder=6,
               marker="*", edgecolors="white", linewidths=0.8,
               label=f"Best r = {best_r} km (score = {best_s:.3f})")

    ax.set_title(f"{label} Radius Search", color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Radius (km)", color="white", fontsize=10)
    ax.set_ylabel("Score = Cost(r) / Avg Coverage(r)", color="white", fontsize=10)
    leg = ax.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
                    loc="upper right")

# Find best
best_dense_idx = int(np.argmin(dense_scores))
best_dense_r = dense_radii[best_dense_idx]
best_dense_s = dense_scores[best_dense_idx]

best_sparse_idx = int(np.argmin(sparse_scores))
best_sparse_r = sparse_radii[best_sparse_idx]
best_sparse_s = sparse_scores[best_sparse_idx]

plot_score_panel(ax1, dense_radii, dense_scores, dense_lb, dense_ub,
                 best_dense_r, best_dense_s, "Dense", "#00e676")
plot_score_panel(ax2, sparse_radii, sparse_scores, sparse_lb, sparse_ub,
                 best_sparse_r, best_sparse_s, "Sparse", "#ffab00")

fig.suptitle("Coarse-to-Fine Score Landscape", color="white",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("images/coarse_fine_score.png", dpi=180, facecolor=fig.get_facecolor())
plt.close()
print("✓ Saved images/coarse_fine_score.png")

print(f"\nDense:  bounds=[{dense_lb}, {dense_ub}], best r={best_dense_r} km, score={best_dense_s:.4f}")
print(f"Sparse: bounds=[{sparse_lb}, {sparse_ub}], best r={best_sparse_r} km, score={best_sparse_s:.4f}")
