"""
Interactive UI for 6G Tower Placement Optimization.

Drag sliders, press Run, see live results.
Uses core/ as backend — no core logic modified.
"""

from __future__ import annotations

import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from core.data import find_cities_file
from core.pipeline import Result, run


def _km_to_degrees_lon(km: float, latitude_deg: float) -> float:
    return km / (111.320 * math.cos(math.radians(latitude_deg)))


def _km_to_degrees_lat(km: float) -> float:
    return km / 110.574


# ---------------------------------------------------------------------------
# Map drawing
# ---------------------------------------------------------------------------


def _draw_map(ax: plt.Axes, result: Result) -> None:
    ax.clear()
    ax.set_facecolor("#16213e")

    cities = result.cities
    labels = result.labels
    unique_labels = sorted(set(labels))
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(len(unique_labels), 1))

    for label in unique_labels:
        mask = labels == label
        subset = cities[mask]
        if label < 0:
            color, mkr, sz, zo = "#888888", "x", 22, 2
        else:
            color, mkr, sz, zo = cmap(label % 20), "o", 20, 2
        kw = dict(c=[color], marker=mkr, s=sz, linewidths=0.5, zorder=zo)
        if label >= 0:
            kw["edgecolors"] = "white"
        ax.scatter(subset["longitude"], subset["latitude"], **kw)

    selected = result.optimization.selected_candidates
    for tower in selected:
        rx = _km_to_degrees_lon(tower.radius_km, tower.latitude)
        ry = _km_to_degrees_lat(tower.radius_km)
        fc = "#00e676" if tower.tower_type == "dense" else "#ffab00"
        ax.add_patch(mpatches.Ellipse(
            (tower.longitude, tower.latitude), 2 * rx, 2 * ry,
            lw=0.8, ec=fc, fc=fc, alpha=0.12, zorder=1,
        ))
        ax.add_patch(mpatches.Ellipse(
            (tower.longitude, tower.latitude), 2 * rx, 2 * ry,
            lw=0.6, ec=fc, fc="none", alpha=0.45, zorder=3,
        ))

    tlons = [t.longitude for t in selected]
    tlats = [t.latitude for t in selected]
    ax.scatter(tlons, tlats, c="#ff1744", marker="^", s=70,
               linewidths=0.6, edgecolors="white", zorder=5)

    pair_values = result.optimization.pair_values
    sel = set(result.optimization.selected_indices.tolist())
    n_int = 0
    for pi, (li, ri) in enumerate(result.interference_pairs):
        if li in sel and ri in sel and pair_values[pi] > 0.5:
            a, b = result.candidates[li], result.candidates[ri]
            ax.plot([a.longitude, b.longitude], [a.latitude, b.latitude],
                    color="#ff1744", lw=1.2, ls="--", alpha=0.7, zorder=4)
            n_int += 1

    # compact legend
    rp = result.radius_plan
    handles = [
        mpatches.Patch(fc="#00e676", alpha=0.35, ec="#00e676",
                       label=f"Dense {rp.dense_radius_km:.0f} km"),
        mpatches.Patch(fc="#ffab00", alpha=0.35, ec="#ffab00",
                       label=f"Sparse {rp.sparse_radius_km:.0f} km"),
        plt.Line2D([], [], color="#ff1744", ls="--", lw=1.2,
                   label=f"Interference ({n_int})"),
        plt.Line2D([], [], color="#ff1744", marker="^", ls="None",
                   markersize=7, mec="white", label="Tower"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=7,
                    facecolor="#1a1a2e", edgecolor="#444",
                    labelcolor="white", framealpha=0.9, ncol=2)
    leg.get_frame().set_linewidth(0.5)

    opt = result.optimization
    ax.set_title(
        f"{len(selected)} towers  |  Coverage {opt.coverage_ratio:.0%}  |  "
        f"Objective {opt.objective_value:.2f}",
        color="white", fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Longitude", color="white", fontsize=10, fontweight="bold")
    ax.set_ylabel("Latitude", color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.set_aspect("equal")
    ax.set_xlim(5.5, 15.5)
    ax.set_ylim(47.0, 55.0)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------


def main() -> None:
    cities_file = find_cities_file()

    # --- initial run ---------------------------------------------------------
    result = run(cities_file, eps_km=45.0, min_samples=2, alpha=0.5, beta=0.0, radius_mode="milp_pair_search")

    # --- figure layout using GridSpec ----------------------------------------
    fig = plt.figure(figsize=(18, 10.5), facecolor="#1a1a2e")
    fig.canvas.manager.set_window_title("6G Tower Placement — Interactive UI")

    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.02,
                          left=0.03, right=0.98, top=0.94, bottom=0.06)

    # Left panel — controls
    gs_left = gs[0, 0].subgridspec(8, 1, hspace=0.6)
    ax_sliders = [fig.add_subplot(gs_left[i, 0]) for i in range(5)]
    ax_btn = fig.add_subplot(gs_left[5, 0])
    ax_info = fig.add_subplot(gs_left[6:, 0])

    for ax in ax_sliders:
        ax.set_facecolor("#1a1a2e")
    ax_btn.set_facecolor("#1a1a2e")
    ax_info.set_facecolor("#16213e")
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_color("#444")

    # Map — right side
    ax_map = fig.add_subplot(gs[0, 1])

    # --- sliders -------------------------------------------------------------
    slider_style = dict(color="#00bcd4")
    label_style = dict(color="white", fontsize=9, fontweight="bold")
    val_style = dict(color="#aaaaaa", fontsize=8)

    s_eps   = Slider(ax_sliders[0], "eps-km", 10.0, 100.0, valinit=45.0, valstep=1.0,  **slider_style)
    s_min   = Slider(ax_sliders[1], "min-samples", 1, 10, valinit=2, valstep=1, valfmt="%d", **slider_style)
    s_alpha = Slider(ax_sliders[2], "alpha", 0.0, 1.0, valinit=0.5, valstep=0.01,         **slider_style)
    s_beta  = Slider(ax_sliders[3], "beta", 0.0, 1.0, valinit=0.25, valstep=0.01,         **slider_style)
    s_time  = Slider(ax_sliders[4], "time-limit (s)", 5.0, 300.0, valinit=60.0, valstep=5.0, **slider_style)

    for s in [s_eps, s_min, s_alpha, s_beta, s_time]:
        s.label.set(**label_style)
        s.valtext.set(**val_style)

    # --- button --------------------------------------------------------------
    btn = Button(ax_btn, "Run Optimization", color="#00bcd4", hovercolor="#0097a7")
    btn.label.set(color="white", fontsize=11, fontweight="bold")

    # --- initial draw --------------------------------------------------------
    _draw_map(ax_map, result)

    def _make_info(r: Result) -> str:
        opt = r.optimization
        rp = r.radius_plan
        cost = sum(c.cost for c in opt.selected_candidates)
        return (
            f"r_dense  = {rp.dense_radius_km:.0f} km\n"
            f"r_sparse = {rp.sparse_radius_km:.0f} km\n"
            f"Towers   = {len(opt.selected_candidates)}\n"
            f"Coverage = {opt.coverage_ratio:.1%}\n"
            f"Cost     = {cost:.2f}\n"
            f"Clusters = {r.cluster_profile.cluster_count}\n"
            f"Noise    = {r.cluster_profile.noise_count}\n"
            f"Candidates = {len(r.candidates)}\n"
            f"Solver   = {opt.status}"
        )

    info_text = ax_info.text(
        0.08, 0.96, _make_info(result),
        transform=ax_info.transAxes, ha="left", va="top",
        fontsize=9.5, color="white", fontfamily="monospace",
    )

    # --- callback ------------------------------------------------------------
    def _on_run(_event):
        try:
            info_text.set_text("Running ...")
            fig.canvas.draw_idle()

            r = run(
                cities_file,
                eps_km=float(s_eps.val),
                min_samples=int(s_min.val),
                alpha=float(s_alpha.val),
                beta=float(s_beta.val),
                time_limit=float(s_time.val),
                radius_mode="milp_pair_search",
            )
            _draw_map(ax_map, r)
            info_text.set_text(_make_info(r))
            fig.canvas.draw_idle()
        except Exception as exc:
            info_text.set_text(f"Error:\n{exc}")
            fig.canvas.draw_idle()

    btn.on_clicked(_on_run)

    fig.suptitle("6G Tower Placement — Interactive Parameter Explorer",
                 color="white", fontsize=15, fontweight="bold")
    plt.show()


if __name__ == "__main__":
    main()
