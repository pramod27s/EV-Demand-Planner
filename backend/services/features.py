"""Serve-time feature building.

Reuses the exact feature functions of ``src/data/preprocess.py`` (calendar features, the
rare-category mapping and the fixed-level model frame), so a prediction request is
encoded identically to the training data.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data.preprocess import apply_category_levels, time_features, to_model_frame


def build_features(*, station: dict, port_type: str, plug_type: str, port_number: int,
                   start: datetime, user_prev_sessions: int, user_prev_mean_kwh: float | None,
                   meta: dict) -> tuple[pd.DataFrame, dict]:
    """One-row model frame for a plug-in event, plus the raw feature values used."""
    row = time_features(pd.Series([pd.Timestamp(start)]))
    row["station"] = station["name"]
    row["port_type"] = port_type
    row["plug_type"] = plug_type
    row["port_number"] = port_number
    row["lat"] = station["lat"]
    row["lon"] = station["lon"]
    row["driver_zip_known"] = 1
    row["user_prev_sessions"] = user_prev_sessions
    no_history = user_prev_mean_kwh is None or user_prev_sessions == 0
    row["user_prev_mean_kwh"] = meta["global_train_mean_kwh"] if no_history else user_prev_mean_kwh
    row["station_prev_mean_kwh"] = station["station_prev_mean_kwh"]

    row = apply_category_levels(row, meta["categories"])
    used = {k: (v.item() if hasattr(v, "item") else v) for k, v in row[meta["features"]].iloc[0].items()}
    return to_model_frame(row, meta), used
