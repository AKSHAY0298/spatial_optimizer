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


def plot_result(
    result: Result,
    total_cost: float = None,
    overlap_percentage: float = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Render a map with cities, selected towers, coverage radii, and interference."""

    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#2b2b3d")
    ax.set_facecolor("#1e1e2e")

    # ── Coverage circles (draw first so cities appear on top) ────────────
    selected = result.optimization.selected_candidates
    dense_radius_km = result.radius_plan.dense_radius_km
    sparse_radius_km = result.radius_plan.sparse_radius_km

    for tower in selected:
        rx = _km_to_degrees_lon(tower.radius_km, tower.latitude)
        ry = _km_to_degrees_lat(tower.radius_km)
        if tower.tower_type == "dense":
            fill_color = "#00e676"
            edge_color = "#00c853"
        else:
            fill_color = "#ffd740"
            edge_color = "#ffab00"

        circle = mpatches.Ellipse(
            (tower.longitude, tower.latitude),
            width=2 * rx, height=2 * ry,
            linewidth=1.0, edgecolor=edge_color, facecolor=fill_color,
            alpha=0.18, zorder=1,
        )
        ax.add_patch(circle)
        border = mpatches.Ellipse(
            (tower.longitude, tower.latitude),
            width=2 * rx, height=2 * ry,
            linewidth=0.8, edgecolor=edge_color, facecolor="none",
            alpha=0.5, zorder=2,
        )
        ax.add_patch(border)

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
                color="#ff5252", linewidth=1.0, linestyle="--", alpha=0.7, zorder=4,
            )
            interference_drawn += 1

    # ── City points (coloured by cluster label) ──────────────────────────
    cities = result.cities
    labels = result.labels
    unique_labels = sorted(set(labels))
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(unique_labels), 1))

    clustered_plotted = False
    noise_plotted = False
    for label in unique_labels:
        mask = labels == label
        subset = cities[mask]
        if label < 0:
            ax.scatter(
                subset["longitude"], subset["latitude"],
                c="#e040fb", marker="^", s=35, linewidths=0.5,
                edgecolors="white", zorder=6,
                label="City (noise)" if not noise_plotted else None,
            )
            noise_plotted = True
        else:
            color = cmap(label % 10)
            ax.scatter(
                subset["longitude"], subset["latitude"],
                c=[color], marker="o", s=30, linewidths=0.4,
                edgecolors="white", zorder=6,
                label="City (clustered)" if not clustered_plotted else None,
            )
            clustered_plotted = True

    # ── Selected tower markers ───────────────────────────────────────────
    tower_lons = [t.longitude for t in selected]
    tower_lats = [t.latitude for t in selected]
    ax.scatter(
        tower_lons, tower_lats,
        c="#ff5722", marker="^", s=80, linewidths=0.7,
        edgecolors="white", zorder=7, label="Selected tower",
    )

    # ── Legend ───────────────────────────────────────────────────────────
    dense_r_display = int(dense_radius_km) if dense_radius_km == int(dense_radius_km) else f"{dense_radius_km:.1f}"
    sparse_r_display = int(sparse_radius_km) if sparse_radius_km == int(sparse_radius_km) else f"{sparse_radius_km:.1f}"

    legend_handles = [
        plt.Line2D([], [], color="gray", marker="o", linestyle="None",
                   markersize=6, markeredgecolor="white", label="City (clustered)"),
        plt.Line2D([], [], color="#e040fb", marker="^", linestyle="None",
                   markersize=6, markeredgecolor="white", label="City (noise)"),
        plt.Line2D([], [], color="#ff5722", marker="^", linestyle="None",
                   markersize=8, markeredgecolor="white", label="Selected tower"),
        mpatches.Patch(facecolor="#00e676", alpha=0.4, edgecolor="#00c853",
                       label=f"Dense radius ({dense_r_display} km)"),
        mpatches.Patch(facecolor="#ffd740", alpha=0.4, edgecolor="#ffab00",
                       label=f"Sparse radius ({sparse_r_display} km)"),
        plt.Line2D([], [], color="#ff5252", linestyle="--", linewidth=1.2,
                   label=f"Interference ({interference_drawn} pairs)"),
    ]
    legend = ax.legend(
        handles=legend_handles, loc="upper left", fontsize=9,
        facecolor="#2b2b3d", edgecolor="#555", labelcolor="white",
        framealpha=0.92,
    )
    legend.get_frame().set_linewidth(0.6)

    # ── Axis labels and title ────────────────────────────────────────────
    ax.set_xlabel("Longitude", color="white", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Latitude", color="white", fontsize=11, fontweight="bold", labelpad=8)

    title_text = f"6G Tower Placement  —  {len(selected)} towers  |  Coverage {result.optimization.coverage_ratio:.2%}"
    if total_cost is not None and overlap_percentage is not None:
        title_text += f"\nCost: {total_cost:.2f}  |  Overlap: {overlap_percentage:.2f}%"
    else:
        title_text += f"\n Objective: {result.optimization.objective_value:.2f}"

    ax.set_title(title_text, color="white", fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#555")

    ax.set_aspect("equal")
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Wrote visualization to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 6G tower placement Algorithm.")
    parser.add_argument("--cities", type=str, default=None, help="Path to cities_de_50k.txt")
    parser.add_argument("--eps-km", type=float, default=40.0, help="DBSCAN epsilon in kilometers")
    parser.add_argument("--min-samples", type=int, default=2, help="DBSCAN min_samples parameter")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for cost vs interference (0=interference only, 1=cost only)")
    parser.add_argument("--beta", type=float, default=0.0, help="Legacy coverage incentive; ignored when hard coverage is enabled")
    parser.add_argument(
        "--radius-mode",
        choices=("milp_pair_search", "dbscan_representative"),
        default="milp_pair_search",
        help="How to choose the two tower radii",
    )
    parser.add_argument(
        "--radius-milp-top-k",
        type=int,
        default=8,
        help="Number of proxy-ranked radius pairs to verify with the MILP",
    )
    parser.add_argument(
        "--radius-step",
        type=float,
        default=5.0,
        help="Step size for candidate radii (1.0=integers, 0.5=half-km resolution)",
    )
    parser.add_argument(
        "--radius-time-limit",
        type=float,
        default=20.0,
        help="MILP time limit in seconds for each radius pair during radius search",
    )
    parser.add_argument("--time-limit", type=float, default=60.0, help="MILP solver time limit in seconds")
    parser.add_argument("--no-midpoints", action="store_true", help="Disable midpoint candidate locations")
    parser.add_argument("--voronoi", action="store_true", help="Enable Voronoi vertex candidate locations (experimental)")
    parser.add_argument(
        "--midpoint-max-distance",
        type=float,
        default=80.0,
        help="Max distance (km) between cities to generate midpoint candidates",
    )
    parser.add_argument("--grid", action="store_true", help="Enable hexagonal grid candidate locations")
    parser.add_argument("--grid-spacing", type=float, default=50.0, help="Grid spacing in km")
    parser.add_argument("--refine", action="store_true", help="Enable per-tower radius refinement (WARNING: produces more than 2 radii)")
    parser.add_argument("--output", type=str, default=None, help="Path to output .txt file (one line per tower: latitude, longitude, radius)")
    parser.add_argument("--plot-output", type=str, default=None, help="Path to save the visualization image, for example images/final_topology.png")
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
        radius_mode=args.radius_mode,
        radius_milp_top_k=args.radius_milp_top_k,
        radius_step=args.radius_step,
        radius_time_limit=args.radius_time_limit,
        time_limit=args.time_limit,
        include_midpoints=not args.no_midpoints,
        midpoint_max_distance_km=args.midpoint_max_distance,
        include_voronoi=args.voronoi,
        include_grid=args.grid,
        grid_spacing_km=args.grid_spacing,
        refine_radii=args.refine,
    )

    print(f"Cities loaded: {len(result.cities)}")
    print(f"Clusters found: {result.cluster_profile.cluster_count}")
    print(f"Noise cities: {result.cluster_profile.noise_count}")
    print(f"Radius mode: {result.radius_plan.search_method}")
    if result.radius_plan.evaluated_radius_pairs:
        print(f"Radius pairs ranked: {result.radius_plan.evaluated_radius_pairs}")
        print(f"Radius pairs solved with MILP: {result.radius_plan.solved_radius_pairs}")
    print(
        f"Dense radius: {result.radius_plan.dense_radius_km} km "
        f"(proxy={result.radius_plan.proxy_objective or result.radius_plan.dense_score:.4f}, bounds={result.radius_plan.dense_bounds_km})"
    )
    print(
        f"Sparse radius: {result.radius_plan.sparse_radius_km} km "
        f"(objective={result.radius_plan.pair_objective or result.radius_plan.sparse_score:.4f}, bounds={result.radius_plan.sparse_bounds_km})"
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
    print("\nOPTIMIZATION METRICS:")
    print(f"Coverage ratio: {result.optimization.coverage_ratio:.3f}")
    print(f"Selected towers: {len(result.optimization.selected_candidates)}")
    print("\nCOSTS & INTERFERENCE:")
    print(f"Total tower cost: {total_cost:.4f}")
    print(f"Total coverage area: {total_coverage_area_sq_km:,.2f} sq km")
    print(f"Overlap area: {overlap_area_sq_km:,.2f} sq km ({overlap_percentage:.2f}% of coverage)")
    print(f"Interference penalty (sum): {interference_penalty_sum:.4f}")
    print(f"Combined objective (cost + penalty): {total_objective:.4f}")
    print(f"Solver objective: {result.optimization.objective_value:.4f}")
    if result.radius_pair_evaluations:
        solved = [item for item in result.radius_pair_evaluations if item.solver_status is not None]
        solved = sorted(solved, key=lambda item: float("inf") if item.milp_objective is None else item.milp_objective)
        print("\nBEST RADIUS PAIRS TESTED (LP relaxation → MILP):")
        for item in solved[:5]:
            lp_str = f"LP={item.lp_objective:.4f}" if item.lp_objective is not None else "LP=N/A"
            gap_str = ""
            if item.lp_objective is not None and item.milp_objective is not None and item.lp_objective > 0:
                gap = (item.milp_objective - item.lp_objective) / item.lp_objective * 100
                gap_str = f" | gap={gap:.1f}%"
            print(
                f"- ({item.dense_radius_km}, {item.sparse_radius_km}) km | "
                f"{lp_str} → MILP={item.milp_objective:.4f}{gap_str} | towers={item.selected_tower_count} | "
                f"cost={item.total_cost:.4f} | interference={item.total_interference:.4f} | "
                f"status={item.solver_status}"
            )

    print(f"\nSELECTED TOWERS (showing first 20 of {len(result.optimization.selected_candidates)}):")

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

    # Determine output path for visualization
    plot_output = args.plot_output
    if plot_output is None and not args.no_plot:
        # Default to images/vis.png if no explicit output path provided
        images_dir = Path("images")
        images_dir.mkdir(exist_ok=True)
        plot_output = str(images_dir / "vis.png")

    if not args.no_plot:
        plot_result(result, total_cost, overlap_percentage, plot_output, show=True)
    elif plot_output:
        plot_result(result, total_cost, overlap_percentage, plot_output, show=False)

if __name__ == "__main__":
    main()
