"""
Generate a side-by-side result visual:
  Left:  Zoomed-in cluster view with 27 km dense radius circles
  Right: Noise points with 89 km sparse radius circles + P95 pairwise line
"""
from __future__ import annotations
import math, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from core.data import load_cities
from core.spatial import cluster_cities, pairwise_haversine_km, haversine_distance_km

# ── Load & cluster ───────────────────────────────────────────────────────
cities = load_cities()
labels = cluster_cities(cities, eps_km=40.0, min_samples=2)

labeled = cities.copy()
labeled["cluster_label"] = labels

clustered = labeled[labeled["cluster_label"] >= 0]
noise     = labeled[labeled["cluster_label"] < 0]

DENSE_R  = 27  # km
SPARSE_R = 89  # km


def km_to_deg_lon(km, lat_deg):
    return km / (111.320 * math.cos(math.radians(lat_deg)))

def km_to_deg_lat(km):
    return km / 110.574


# =====================================================================
# FIGURE — side by side
# =====================================================================
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7), facecolor="#1a1a2e")

for ax in (ax_left, ax_right):
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.set_xlabel("Longitude", color="white", fontsize=10)
    ax.set_ylabel("Latitude", color="white", fontsize=10)

# ── LEFT: Dense clusters with 27 km radius ──────────────────────────────
cmap = plt.colormaps.get_cmap("tab20")

# Plot all clustered cities coloured by cluster
for cl, grp in clustered.groupby("cluster_label"):
    color = cmap(int(cl) % 20)
    ax_left.scatter(grp["longitude"], grp["latitude"],
                    c=[color], s=30, edgecolors="white", linewidths=0.4,
                    zorder=3, label=f"Cluster {int(cl)}" if int(cl) < 5 else None)
    # Centroid
    clat, clon = grp["latitude"].mean(), grp["longitude"].mean()
    ax_left.scatter(clon, clat, marker="*", c="gold", s=100, edgecolors="black",
                    linewidths=0.5, zorder=5)
    # 27 km radius circle around centroid
    rx = km_to_deg_lon(DENSE_R, clat)
    ry = km_to_deg_lat(DENSE_R)
    circle = mpatches.Ellipse((clon, clat), width=2*rx, height=2*ry,
                              linewidth=1.0, edgecolor="#00e676", facecolor="#00e676",
                              alpha=0.10, zorder=1)
    ax_left.add_patch(circle)
    border = mpatches.Ellipse((clon, clat), width=2*rx, height=2*ry,
                              linewidth=0.8, edgecolor="#00e676", facecolor="none",
                              alpha=0.5, zorder=2)
    ax_left.add_patch(border)

# Also show noise as faint X markers for context
ax_left.scatter(noise["longitude"], noise["latitude"],
                c="#888888", marker="x", s=20, linewidths=0.5, zorder=2, alpha=0.4)

# Zoom to the densest region (Ruhr / NRW area)
ax_left.set_xlim(5.8, 15.5)
ax_left.set_ylim(47.5, 55.2)
ax_left.set_aspect("equal")
ax_left.set_title(f"Dense Clusters — $r_{{\\mathrm{{dense}}}}$ = {DENSE_R} km Coverage",
                   color="white", fontsize=12, fontweight="bold")

# Legend
dense_patch = mpatches.Patch(facecolor="#00e676", alpha=0.3, edgecolor="#00e676",
                             label=f"{DENSE_R} km radius")
centroid_marker = plt.Line2D([], [], color="gold", marker="*", linestyle="None",
                             markersize=10, markeredgecolor="black", label="Centroid")
leg1 = ax_left.legend(handles=[dense_patch, centroid_marker], loc="upper left",
                      fontsize=8, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

# ── RIGHT: Noise points with 89 km radius + P95 lines ───────────────────
noise_pts = noise[["latitude", "longitude"]].to_numpy(dtype=float)

# Draw 89 km circles around each noise city
for _, row in noise.iterrows():
    lat, lon = row["latitude"], row["longitude"]
    rx = km_to_deg_lon(SPARSE_R, lat)
    ry = km_to_deg_lat(SPARSE_R)
    circle = mpatches.Ellipse((lon, lat), width=2*rx, height=2*ry,
                              linewidth=1.0, edgecolor="#ffab00", facecolor="#ffab00",
                              alpha=0.08, zorder=1)
    ax_right.add_patch(circle)
    border = mpatches.Ellipse((lon, lat), width=2*rx, height=2*ry,
                              linewidth=0.8, edgecolor="#ffab00", facecolor="none",
                              alpha=0.4, zorder=2)
    ax_right.add_patch(border)

ax_right.scatter(noise["longitude"], noise["latitude"],
                 c="#ffab00", marker="^", s=60, edgecolors="white", linewidths=0.5,
                 zorder=4, label="Noise city")

# Draw lines between pairs near the P95 pairwise distance
pw = pairwise_haversine_km(noise_pts)
n = len(noise_pts)
upper_tri = pw[np.triu_indices(n, k=1)]
positive = upper_tri[upper_tri > 0]
p95 = float(np.percentile(positive, 95))

# Draw lines for pairs within [P90, P95] to show what P95 looks like
p90 = float(np.percentile(positive, 90))
noise_lats = noise["latitude"].values
noise_lons = noise["longitude"].values
lines_drawn = 0
for i in range(n):
    for j in range(i+1, n):
        d = pw[i, j]
        if p90 <= d <= p95:
            ax_right.plot([noise_lons[i], noise_lons[j]],
                          [noise_lats[i], noise_lats[j]],
                          color="#ff1744", linewidth=0.8, linestyle="--", alpha=0.5, zorder=3)
            lines_drawn += 1

# Also plot clustered cities faintly for geographic context
ax_right.scatter(clustered["longitude"], clustered["latitude"],
                 c="#555555", s=8, alpha=0.2, zorder=1)

ax_right.set_xlim(5.8, 15.5)
ax_right.set_ylim(47.5, 55.2)
ax_right.set_aspect("equal")
ax_right.set_title(f"Noise Cities — $r_{{\\mathrm{{sparse}}}}$ = {SPARSE_R} km Coverage",
                   color="white", fontsize=12, fontweight="bold")

sparse_patch = mpatches.Patch(facecolor="#ffab00", alpha=0.3, edgecolor="#ffab00",
                              label=f"{SPARSE_R} km radius")
p95_line = plt.Line2D([], [], color="#ff1744", linestyle="--", linewidth=1.0,
                      label=f"$P_{{90}}$–$P_{{95}}$ pairs ({p90:.0f}–{p95:.0f} km)")
leg2 = ax_right.legend(handles=[sparse_patch, p95_line], loc="upper left",
                       fontsize=8, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

fig.suptitle("Selected Radii Applied to the Two Spatial Regimes", color="white",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("images/radius_result_visual.png", dpi=180, facecolor=fig.get_facecolor())
plt.close()
print(f"✓ Saved images/radius_result_visual.png  (P95 = {p95:.1f} km, lines drawn = {lines_drawn})")
