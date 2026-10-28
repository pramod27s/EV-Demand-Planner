"""Loading the raw Palo Alto EV charging CSV.

The CSV is downloaded manually by the student; this module never downloads data.
"""
from __future__ import annotations

import sys

import pandas as pd

from src.config import resolve

DATASET_URL = (
    "https://data.cityofpaloalto.org/dataviews/257812/"
    "electric-vehicle-charging-station-usage-july-2011-dec-2020/"
)


def load_raw(cfg: dict) -> pd.DataFrame:
    """Read the raw CSV named in ``cfg['data']['raw_path']``.

    Exits with a clear message (and the download URL) if the file is missing.
    """
    path = resolve(cfg["data"]["raw_path"])
    if not path.exists():
        print(
            f"\n[ERROR] Raw dataset not found at:\n    {path}\n\n"
            "Download 'Electric Vehicle Charging Station Usage (July 2011 - Dec 2020)' from\n"
            f"    {DATASET_URL}\n"
            "(mirrors exist on Kaggle / ORNL Open Energy Hub) and save it at the path above.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")   # the official file starts with a BOM
    df.columns = [c.strip() for c in df.columns]
    return df


def check_columns(df: pd.DataFrame, cfg: dict) -> None:
    """Fail early if a column mapped in ``config.yaml -> columns`` is absent."""
    required = {k: v for k, v in cfg["columns"].items() if isinstance(v, str)}
    missing = {k: v for k, v in required.items() if v not in df.columns}
    if missing:
        raise KeyError(
            f"Columns not found in CSV: {missing}. Available columns: {list(df.columns)}. "
            "Fix the names in config.yaml -> columns."
        )
