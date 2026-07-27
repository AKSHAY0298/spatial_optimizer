"""Render the cost-vs-interference trade-off from a pareto_sweep.py CSV.

Three categories are drawn, because the honest picture needs all of them:
  * non-dominated, epsilon-respecting solutions  -- the actual Pareto front
  * dominated, epsilon-respecting solutions      -- context
  * solutions whose realized interference exceeds the epsilon they were solved
    under (the lazy cutting-plane loop hit its round cap) -- NOT valid points
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FRONT = "#2a78d6"    # categorical slot 1
INVALID = "#eb6834"  # categorical slot 2
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d7d2"


def load(path: Path):
    valid, invalid = [], []
    with path.open(encoding="utf8") as fh:
        for row in csv.DictReader(fh):
            if not row["cost"]:
                continue
            point = {
                "eps": float(row["epsilon"]),
                "cost": float(row["cost"]),
                "interf": float(row["interference"]),
                "towers": int(row["towers"]),
            }
            respects = row.get("respects_epsilon", "True")
            (valid if respects in ("True", "true", "") else invalid).append(point)
    return valid, invalid


def non_dominated(points):
    """Points with no other point at both lower-or-equal cost and interference."""
    front, best_cost = [], float("inf")
    for p in sorted(points, key=lambda q: (q["interf"], q["cost"])):
        if p["cost"] < best_cost - 1e-9:
            front.append(p)
            best_cost = p["cost"]
    return front


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", required=True)
    args = ap.parse_args()

    valid, invalid = load(Path(args.csv))
    front = non_dominated(valid)
    front_keys = {(round(p["interf"], 9), round(p["cost"], 9)) for p in front}
    dominated = [p for p in valid if (round(p["interf"], 9), round(p["cost"], 9)) not in front_keys]

    fig, ax = plt.subplots(figsize=(6.6, 4.3), facecolor="white")
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    if invalid:
        ax.scatter([p["interf"] for p in invalid], [p["cost"] for p in invalid],
                   marker="x", s=58, linewidths=1.6, color=INVALID, zorder=3,
                   label="Round cap hit (violates $\\varepsilon$)")
    if dominated:
        ax.scatter([p["interf"] for p in dominated], [p["cost"] for p in dominated],
                   marker="o", s=34, facecolors="white", edgecolors=INK_SOFT,
                   linewidths=1.0, zorder=4, label="Feasible, dominated")
    if len(front) > 1:
        ordered = sorted(front, key=lambda p: p["interf"])
        ax.step([p["interf"] for p in ordered], [p["cost"] for p in ordered],
                where="post", color=FRONT, linewidth=2.0, zorder=5)
    ax.scatter([p["interf"] for p in front], [p["cost"] for p in front],
               marker="o", s=78, color=FRONT, edgecolors="white", linewidths=1.2,
               zorder=6, label="Pareto-optimal")

    for p in front:
        ax.annotate(f"{p['towers']} towers\ncost {p['cost']:.2f}",
                    xy=(p["interf"], p["cost"]), xytext=(9, 9),
                    textcoords="offset points", fontsize=8.5, color=INK_SOFT,
                    linespacing=1.35)

    ax.set_xlabel("Realised interference  $\\sum_p P_p y_p$", fontsize=10.5, color=INK)
    ax.set_ylabel("Total installation cost", fontsize=10.5, color=INK)
    ax.set_title(args.title, fontsize=11.5, color=INK, pad=10)
    ax.tick_params(labelsize=9, colors=INK_SOFT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    ax.legend(fontsize=8.5, frameon=False, loc="best", labelcolor=INK_SOFT)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"front={len(front)} dominated={len(dominated)} invalid={len(invalid)} -> {out}")


if __name__ == "__main__":
    main()
