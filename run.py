from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from core.data import find_cities_file
from core.pipeline import Result, run


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _km_to_degrees_lon(km: float, latitude_deg: float) -> float:
    """Convert a distance in km to an approximate longitude-degree offset."""
    return km / (111.320 * math.cos(math.radians(latitude_deg)))


def _km_to_degrees_lat(km: float) -> float:
    """Convert a distance in km to an approximate latitude-degree offset."""
    return km / 110.574


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _calculate_circle_intersection_area(lat1: float, lon1: float, r1: float,
                                         lat2: float, lon2: float, r2: float) -> float:
    """Calculate the intersection area of two circles on Earth surface (in sq km²)."""
    from core.spatial import haversine_distance_km
    
    point_a = np.array([lat1, lon1])
    point_b = np.array([lat2, lon2])
    distance = haversine_distance_km(point_a, point_b)
    
    # If circles don't overlap, return 0
    if distance >= r1 + r2:
        return 0.0
    
    # If one circle is completely inside the other
    if distance <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    
    # Partial overlap - use formula for intersection of two circles
    part1 = r1**2 * math.acos((distance**2 + r1**2 - r2**2) / (2 * distance * r1))
    part2 = r2**2 * math.acos((distance**2 + r2**2 - r1**2) / (2 * distance * r2))
    part3 = 0.5 * math.sqrt((r1 + r2 - distance) * (r1 - r2 + distance) * 
                            (-r1 + r2 + distance) * (r1 + r2 + distance))
    
    return part1 + part2 - part3


def _calculate_optimization_metrics(selected_candidates, candidates, interference_pairs, pair_penalties, result):
    """Calculate total cost, coverage area, and overlap metrics.
    
    Returns:
        (total_cost, total_coverage_area_sq_km, overlap_area_sq_km, 
         overlap_percentage, interference_penalty_sum, total_objective)
    """
    total_cost = sum(c.cost for c in selected_candidates)
    
    # Total individual coverage area
    total_coverage_area_sq_km = sum(math.pi * (c.radius_km ** 2) for c in selected_candidates)
    
    # Calculate actual overlap area from selected interference pairs
    overlap_area_sq_km = 0.0
    interference_penalty_sum = 0.0
    selected_idx_set = set(result.optimization.selected_indices.tolist())
    pair_values = result.optimization.pair_values
    
    for pair_idx, (li, ri) in enumerate(interference_pairs):
        if li in selected_idx_set and ri in selected_idx_set and pair_values[pair_idx] > 0.5:
            # Calculate actual overlap area
            c_left = candidates[li]
            c_right = candidates[ri]
            overlap = _calculate_circle_intersection_area(
                c_left.latitude, c_left.longitude, c_left.radius_km,
                c_right.latitude, c_right.longitude, c_right.radius_km
            )
            overlap_area_sq_km += overlap
            
            # Sum the normalized overlap penalties for solver
            interference_penalty_sum += pair_penalties[pair_idx]
    
    # Overlap as percentage of total coverage
    overlap_percentage = (overlap_area_sq_km / total_coverage_area_sq_km * 100) if total_coverage_area_sq_km > 0 else 0.0
    
    # Combined objective: cost + sum of normalized overlap penalties
    total_objective = total_cost + interference_penalty_sum
    
    return total_cost, total_coverage_area_sq_km, overlap_area_sq_km, overlap_percentage, interference_penalty_sum, total_objective


def plot_result(result: Result, total_cost: float = None, overlap_percentage: float = None) -> None:
    """Render a map with cities, selected towers, coverage radii, and interference."""

    fig, ax = plt.subplots(figsize=(16, 12), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    # ── City points (coloured by cluster label) ──────────────────────────
    cities = result.cities
    labels = result.labels
    unique_labels = sorted(set(labels))
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(len(unique_labels), 1))

    for label in unique_labels:
        mask = labels == label
        subset = cities[mask]
        if label < 0:
            color = "#888888"
            marker = "x"
            size = 22
            zorder = 2
            lbl = "Noise city"
        else:
            color = cmap(label % 20)
            marker = "o"
            size = 20
            zorder = 2
            lbl = f"Cluster {label}"
        scatter_kwargs = dict(
            c=[color], marker=marker, s=size, linewidths=0.5,
            zorder=zorder, label=lbl,
        )
        if label >= 0:
            scatter_kwargs["edgecolors"] = "white"
        ax.scatter(
            subset["longitude"], subset["latitude"],
            **scatter_kwargs,
        )

    # ── Selected towers + coverage circles ───────────────────────────────
    selected = result.optimization.selected_candidates
    tower_lons = [t.longitude for t in selected]
    tower_lats = [t.latitude for t in selected]

    for tower in selected:
        rx = _km_to_degrees_lon(tower.radius_km, tower.latitude)
        ry = _km_to_degrees_lat(tower.radius_km)
        fill_color = "#00e676" if tower.tower_type == "dense" else "#ffab00"
        circle = mpatches.Ellipse(
            (tower.longitude, tower.latitude),
            width=2 * rx, height=2 * ry,
            linewidth=0.8, edgecolor=fill_color, facecolor=fill_color,
            alpha=0.12, zorder=1,
        )
        ax.add_patch(circle)
        border = mpatches.Ellipse(
            (tower.longitude, tower.latitude),
            width=2 * rx, height=2 * ry,
            linewidth=0.6, edgecolor=fill_color, facecolor="none",
            alpha=0.45, zorder=3,
        )
        ax.add_patch(border)

    ax.scatter(
        tower_lons, tower_lats,
        c="#ff1744", marker="^", s=70, linewidths=0.6,
        edgecolors="white", zorder=5, label="Selected tower",
    )

    # ── Interference lines ───────────────────────────────────────────────
    pair_values = result.optimization.pair_values
    selected_idx_set = set(result.optimization.selected_indices.tolist())
    interference_drawn = 0

    for pair_idx, (li, ri) in enumerate(result.interference_pairs):
        if li in selected_idx_set and ri in selected_idx_set and pair_values[pair_idx] > 0.5:
            c_left = result.candidates[li]
            c_right = result.candidates[ri]
            ax.plot(
                [c_left.longitude, c_right.longitude],
                [c_left.latitude, c_right.latitude],
                color="#ff1744", linewidth=1.2, linestyle="--", alpha=0.7, zorder=4,
            )
            interference_drawn += 1

    # ── Legend & labels ──────────────────────────────────────────────────
    dense_patch = mpatches.Patch(facecolor="#00e676", alpha=0.35, edgecolor="#00e676",
                                  label=f"Dense radius ({result.radius_plan.dense_radius_km} km)")
    sparse_patch = mpatches.Patch(facecolor="#ffab00", alpha=0.35, edgecolor="#ffab00",
                                   label=f"Sparse radius ({result.radius_plan.sparse_radius_km} km)")
    interference_line = plt.Line2D([], [], color="#ff1744", linestyle="--", linewidth=1.2,
                                    label=f"Interference ({interference_drawn} pairs)")
    tower_marker = plt.Line2D([], [], color="#ff1744", marker="^", linestyle="None",
                               markersize=8, markeredgecolor="white", label="Selected tower")
    city_marker = plt.Line2D([], [], color="white", marker="o", linestyle="None",
                              markersize=5, markeredgecolor="white", label="City (clustered)")
    noise_marker = plt.Line2D([], [], color="#888888", marker="x", linestyle="None",
                               markersize=6, label="City (noise)")

    handles = [city_marker, noise_marker, tower_marker, dense_patch, sparse_patch, interference_line]
    legend = ax.legend(
        handles=handles, loc="upper left", fontsize=8,
        facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
        framealpha=0.9,
    )
    legend.get_frame().set_linewidth(0.5)

    ax.set_xlabel("Longitude", color="white", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Latitude", color="white", fontsize=12, fontweight="bold", labelpad=10)
    
    # Build title with cost and overlap metrics
    title_text = f"6G Tower Placement  —  {len(selected)} towers  |  Coverage {result.optimization.coverage_ratio:.2%}"
    if total_cost is not None and overlap_percentage is not None:
        title_text += f"\nCost: {total_cost:.2f}  |  Overlap: {overlap_percentage:.2f}%"
    else:
        title_text += f"  |  Objective {result.optimization.objective_value:.2f}"
    
    ax.set_title(
        title_text,
        color="white", fontsize=14, fontweight="bold", pad=20,
    )
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")

    ax.set_aspect("equal")
    
    # Adjust layout to prevent text cutoff
    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95)
    plt.tight_layout(rect=[0.10, 0.10, 0.95, 0.92])
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 6G tower placement Algorithm.")
    parser.add_argument("--cities", type=str, default=None, help="Path to cities_de_50k.txt")
    parser.add_argument("--eps-km", type=float, default=40.0, help="DBSCAN epsilon in kilometers")
    parser.add_argument("--min-samples", type=int, default=3, help="DBSCAN min_samples parameter")
    parser.add_argument("--alpha", type=float, default=0.2, help="Weight for cost vs interference (0=interference only, 1=cost only)")
    parser.add_argument("--beta", type=float, default=0.5, help="Coverage incentive in [0, 1] (0=ignore coverage, 1=cover at any cost")
    parser.add_argument("--time-limit", type=float, default=60.0, help="MILP solver time limit in seconds")
    parser.add_argument("--output", type=str, default=None, help="Path to output .txt file (one line per tower: latitude, longitude, radius)")
    parser.add_argument("--no-plot", action="store_true", help="Skip the map visualisation")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cities_file = Path(args.cities) if args.cities else find_cities_file()
    result = run(
        cities_file,
        eps_km=args.eps_km,
        min_samples=args.min_samples,
        alpha=args.alpha,
        beta=args.beta,
        time_limit=args.time_limit,
    )

    print(f"Cities loaded: {len(result.cities)}")
    print(f"Clusters found: {result.cluster_profile.cluster_count}")
    print(f"Noise cities: {result.cluster_profile.noise_count}")
    print(
        f"Dense radius: {result.radius_plan.dense_radius_km} km "
        f"(score={result.radius_plan.dense_score:.4f}, bounds={result.radius_plan.dense_bounds_km})"
    )
    print(
        f"Sparse radius: {result.radius_plan.sparse_radius_km} km "
        f"(score={result.radius_plan.sparse_score:.4f}, bounds={result.radius_plan.sparse_bounds_km})"
    )
    print(f"Candidates: {len(result.candidates)}")
    print(f"Interference pairs: {len(result.interference_pairs)}")
    print(f"Solver status: {result.optimization.status} - {result.optimization.message}")
    
    # Calculate and display actual cost and overlap metrics
    total_cost, total_coverage_area_sq_km, overlap_area_sq_km, overlap_percentage, interference_penalty_sum, total_objective = _calculate_optimization_metrics(
        result.optimization.selected_candidates,
        result.candidates,
        result.interference_pairs,
        result.interference_penalties,
        result
    )
    print(f"\n📊 OPTIMIZATION METRICS:")
    print(f"Coverage ratio: {result.optimization.coverage_ratio:.3f}")
    print(f"Selected towers: {len(result.optimization.selected_candidates)}")
    print(f"\n💰 COSTS & INTERFERENCE:") 
    print(f"Total tower cost: {total_cost:.4f}")
    print(f"Total coverage area: {total_coverage_area_sq_km:,.2f} sq km")
    print(f"Overlap area: {overlap_area_sq_km:,.2f} sq km ({overlap_percentage:.2f}% of coverage)")
    print(f"Interference penalty (sum): {interference_penalty_sum:.4f}")
    print(f"Combined objective (cost + penalty): {total_objective:.4f}")
    print(f"Solver objective: {result.optimization.objective_value:.4f}")
    print(f"\n🗼 SELECTED TOWERS (showing first 20 of {len(result.optimization.selected_candidates)}):")

    for candidate in result.optimization.selected_candidates[:20]:
        print(
            f"- {candidate.name} | {candidate.source} | type={candidate.tower_type} | "
            f"radius={candidate.radius_km} km | cost={candidate.cost:.3f}"
        )

    if len(result.optimization.selected_candidates) > 20:
        print(f"... {len(result.optimization.selected_candidates) - 20} more selected towers")

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", encoding="utf8") as fh:
            for cand in result.optimization.selected_candidates:
                # Required format: latitude, longitude, radius (in km)
                fh.write(f"{cand.latitude:.5f}, {cand.longitude:.5f}, {int(cand.radius_km)}\n")
        print(f"Wrote {len(result.optimization.selected_candidates)} towers to {out_path}")

    if not args.no_plot:
        plot_result(result, total_cost, overlap_percentage)

if __name__ == "__main__":
    main()