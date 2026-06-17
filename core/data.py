from __future__ import annotations

from pathlib import Path

import pandas as pd


def find_cities_file(start: Path | None = None) -> Path:
    """Locate the German city data file by searching upward from a start path."""

    search_roots = []
    if start is not None:
        start_path = Path(start).resolve()
        search_roots.append(start_path if start_path.is_dir() else start_path.parent)
    search_roots.append(Path.cwd())
    search_roots.append(Path(__file__).resolve().parent)

    seen: set[Path] = set()
    for root in search_roots:
        for candidate_root in [root, *root.parents]:
            if candidate_root in seen:
                continue
            seen.add(candidate_root)
            for relative_path in (
                Path("data") / "cities_de_50k.txt",
                Path("cities_de_50k.txt"),
            ):
                candidate = candidate_root / relative_path
                if candidate.exists():
                    return candidate

    raise FileNotFoundError("Could not locate cities_de_50k.txt")


def load_cities(file_path: str | Path | None = None) -> pd.DataFrame:
    """Load the city table with normalized column names and numeric coordinates."""

    cities_file = find_cities_file() if file_path is None else Path(file_path)
    frame = pd.read_csv(
        cities_file,
        header=None,
        names=["city", "latitude", "longitude"],
        skipinitialspace=True,
    )
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="raise")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="raise")
    return frame