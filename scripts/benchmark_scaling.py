from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data import find_cities_file, load_cities
from core.pipeline import run


def _make_subset_file(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in df.itertuples(index=False):
            fh.write(f"{row.city}, {row.latitude}, {row.longitude}\n")


def benchmark(
    cities_file: Path,
    sample_sizes: list[int],
    repeats: int,
    seed: int,
    radius_step: float,
    radius_milp_top_k: int,
    radius_time_limit: float,
    time_limit: float,
    out_csv: Path,
    out_plot_time: Path,
    out_plot_model: Path,
) -> pd.DataFrame:
    full = load_cities(cities_file)
    rng = np.random.default_rng(seed)
    tmp_dir = Path("images") / "benchmark_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []

    for n in sample_sizes:
        if n > len(full):
            continue

        # Keep benchmark deterministic across repeats for fair timing.
        subset = full.sample(n=n, random_state=seed).sort_values("city").reset_index(drop=True)
        subset_file = tmp_dir / f"cities_subset_{n}.txt"
        _make_subset_file(subset, subset_file)

        for rep in range(repeats):
            t0 = time.perf_counter()
            try:
                result = run(
                    subset_file,
                    eps_km=40.0,
                    min_samples=2,
                    alpha=0.5,
                    beta=0.0,
                    radius_mode="milp_pair_search",
                    radius_milp_top_k=radius_milp_top_k,
                    radius_step=radius_step,
                    radius_time_limit=radius_time_limit,
                    time_limit=time_limit,
                    include_midpoints=False,
                    include_grid=False,
                    refine_radii=False,
                )
                t1 = time.perf_counter()

                rows.append(
                    {
                        "n_cities": n,
                        "repeat": rep + 1,
                        "runtime_sec": t1 - t0,
                        "candidates": len(result.candidates),
                        "interference_pairs": len(result.interference_pairs),
                        "selected_towers": len(result.optimization.selected_candidates),
                        "objective": float(result.optimization.objective_value),
                        "status": result.optimization.message,
                    }
                )
            except RuntimeError as exc:
                t1 = time.perf_counter()
                rows.append(
                    {
                        "n_cities": n,
                        "repeat": rep + 1,
                        "runtime_sec": t1 - t0,
                        "candidates": np.nan,
                        "interference_pairs": np.nan,
                        "selected_towers": np.nan,
                        "objective": np.nan,
                        "status": f"FAILED: {exc}",
                    }
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No benchmark rows generated.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)

    solved = frame[~frame["status"].astype(str).str.startswith("FAILED:")].copy()
    if solved.empty:
        raise RuntimeError("All benchmark runs failed; increase solver time limits.")

    agg = solved.groupby("n_cities", as_index=False).agg(
        runtime_mean=("runtime_sec", "mean"),
        runtime_std=("runtime_sec", "std"),
        candidates_mean=("candidates", "mean"),
        pairs_mean=("interference_pairs", "mean"),
        selected_mean=("selected_towers", "mean"),
    )

    # Plot 1: Runtime scaling.
    out_plot_time.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        agg["n_cities"],
        agg["runtime_mean"],
        yerr=agg["runtime_std"].fillna(0.0),
        marker="o",
        capsize=4,
        color="#0b2e5f",
        label="mean runtime ± std",
    )
    x = agg["n_cities"].to_numpy(dtype=float)
    y = agg["runtime_mean"].to_numpy(dtype=float)
    if len(x) >= 2:
        coeff = np.polyfit(x, y, deg=2)
        xfit = np.linspace(x.min(), x.max(), 200)
        yfit = coeff[0] * xfit**2 + coeff[1] * xfit + coeff[2]
        plt.plot(xfit, yfit, "--", color="#c94c4c", label="quadratic trend")
    plt.title("Runtime Scaling vs Number of Cities (strict mode)")
    plt.xlabel("Number of cities (n)")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot_time, dpi=220)
    plt.close()

    # Plot 2: Model size scaling.
    out_plot_model.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(agg["n_cities"], agg["candidates_mean"], marker="o", label="MILP candidates", color="#2b7a0b")
    plt.plot(agg["n_cities"], agg["pairs_mean"], marker="s", label="interference pairs", color="#8b1d3a")
    plt.plot(agg["n_cities"], agg["selected_mean"], marker="^", label="selected towers", color="#1d5f8b")
    plt.title("Model Size Growth vs Number of Cities")
    plt.xlabel("Number of cities (n)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot_model, dpi=220)
    plt.close()

    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark runtime/model scaling of strict solver.")
    parser.add_argument("--cities", type=str, default=None, help="Path to cities_de_50k.txt")
    parser.add_argument(
        "--sizes",
        type=str,
        default="60,100,140,180,220,236",
        help="Comma-separated sample sizes",
    )
    parser.add_argument("--repeats", type=int, default=2, help="Number of repeats per sample size")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--radius-step", type=float, default=5.0, help="Radius step for pair search")
    parser.add_argument("--radius-top-k", type=int, default=8, help="Top-k radius pairs for MILP verification")
    parser.add_argument("--radius-time-limit", type=float, default=20.0, help="Seconds for each radius-pair MILP solve")
    parser.add_argument("--time-limit", type=float, default=60.0, help="Seconds for final MILP solve")
    parser.add_argument(
        "--out-csv",
        type=str,
        default="images/benchmark_scaling.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--out-time-plot",
        type=str,
        default="images/benchmark_runtime_scaling.png",
        help="Output runtime plot path",
    )
    parser.add_argument(
        "--out-model-plot",
        type=str,
        default="images/benchmark_model_scaling.png",
        help="Output model-size plot path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cities_file = Path(args.cities) if args.cities else find_cities_file()
    sizes = [int(part.strip()) for part in args.sizes.split(",") if part.strip()]

    frame = benchmark(
        cities_file=cities_file,
        sample_sizes=sizes,
        repeats=args.repeats,
        seed=args.seed,
        radius_step=args.radius_step,
        radius_milp_top_k=args.radius_top_k,
        radius_time_limit=args.radius_time_limit,
        time_limit=args.time_limit,
        out_csv=Path(args.out_csv),
        out_plot_time=Path(args.out_time_plot),
        out_plot_model=Path(args.out_model_plot),
    )

    print("Benchmark complete")
    print(f"Rows: {len(frame)}")
    print(f"CSV: {args.out_csv}")
    print(f"Runtime plot: {args.out_time_plot}")
    print(f"Model plot: {args.out_model_plot}")


if __name__ == "__main__":
    main()
