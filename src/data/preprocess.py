"""Preprocessing pipeline: raw Palo Alto CSV -> leakage-free train/val/test splits.

Run with ``python -m src.data.preprocess``. Steps (row counts are logged after each):

1. load & parse ``Start Date``          6. feature engineering (plug-in-time info only)
2. keep 2012-01-01 .. 2019-12-31          7. rare categories -> "other" (levels fitted on train)
3. clean target (missing, <= 0, outliers) 8. chronological 70/15/15 split
4. drop leakage + identifier columns      9. (scaling for Linear Regression is done in
5. fill missing values                       ``src.models.baselines`` and fitted on train)
                                          10. save parquet splits + feature metadata

The feature helpers (:func:`time_features`, :func:`apply_category_levels`,
:func:`to_model_frame`) are imported by the FastAPI backend so that train-time and
serve-time features are computed by exactly the same code.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, processed_dir, results_dir, set_seeds

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
START = "start_time"
TARGET = "energy_kwh"
OTHER = "other"
UNKNOWN = "unknown"
MISSING = "missing"

TIME_FEATURES = ["hour", "hour_sin", "hour_cos", "dayofweek", "is_weekend",
                 "month", "month_sin", "month_cos", "year"]
NUMERIC_FEATURES = TIME_FEATURES + [
    "port_number", "lat", "lon", "driver_zip_known",
    "user_prev_sessions", "user_prev_mean_kwh", "station_prev_mean_kwh",
]
CATEGORICAL_FEATURES = ["station", "port_type", "plug_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Steps 1-5: cleaning
# ---------------------------------------------------------------------------
def parse_start(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Parse the start timestamp into ``start_time``; drop rows that cannot be parsed."""
    col, fmt = cfg["columns"]["start"], cfg["data"].get("date_format")
    raw = df[col].astype(str).str.strip()
    if fmt:
        ts = pd.to_datetime(raw, format=fmt, errors="coerce")
    else:
        try:
            ts = pd.to_datetime(raw, errors="raise")
        except (ValueError, TypeError):
            # Mirrors mix formats (e.g. "7/29/2011 20:17" and "2011-07-29 20:17:00").
            ts = pd.to_datetime(raw, format="mixed", errors="coerce")
    out = df.copy()
    out[START] = ts
    return out[out[START].notna()]


def filter_period(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Keep 2012-2019: 2011 has very few sessions and 2020 is distorted by COVID-19."""
    lo = pd.Timestamp(cfg["data"]["start_date"])
    hi = pd.Timestamp(cfg["data"]["end_date"]) + pd.Timedelta(days=1)  # inclusive end date
    return df[(df[START] >= lo) & (df[START] < hi)]


def clean_target(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, float]:
    """Drop missing / non-positive energy and the top ``1 - outlier_quantile`` outliers.

    Returns the cleaned frame and the outlier threshold (kWh), which is logged.
    """
    out = df.copy()
    out[TARGET] = pd.to_numeric(out[cfg["columns"]["energy"]], errors="coerce")
    out = out[out[TARGET].notna() & (out[TARGET] > 0)]
    threshold = float(out[TARGET].quantile(cfg["data"]["outlier_quantile"]))
    out = out[out[TARGET] <= threshold]
    return out, threshold


def _matches(column: str, names: list[str]) -> bool:
    c = column.strip().lower()
    return any(c.startswith(n.strip().lower()) for n in names)


def drop_unused_columns(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Drop leakage columns (known only after the session ends) and identifier columns."""
    names = cfg["columns"]["leakage"] + cfg["columns"]["identifiers"]
    keep_raw = {v for v in cfg["columns"].values() if isinstance(v, str)}
    dropped = [c for c in df.columns if _matches(c, names) and c not in keep_raw]
    return df.drop(columns=dropped), dropped


def fill_missing(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """User ID / driver postal code missing -> "unknown"; categorical missing -> "missing"."""
    c = cfg["columns"]
    out = df.copy()
    for key in ("user", "driver_zip"):
        s = out[c[key]]
        # User IDs may be read as floats ("123.0"); normalise to clean strings.
        s = s.map(lambda v: UNKNOWN if pd.isna(v) or str(v).strip() == "" else
                  str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v).strip())
        out[c[key]] = s
    for key in ("station", "port_type", "plug_type"):
        out[c[key]] = out[c[key]].fillna(MISSING).astype(str).str.strip().replace("", MISSING)
    for key in ("port_number", "lat", "lon"):
        out[c[key]] = pd.to_numeric(out[c[key]], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# Step 6: feature engineering (shared with the backend)
# ---------------------------------------------------------------------------
def time_features(ts: pd.Series) -> pd.DataFrame:
    """Calendar features of the plug-in time, including cyclical sin/cos encodings."""
    ts = pd.to_datetime(ts)
    hour, month = ts.dt.hour, ts.dt.month
    dow = ts.dt.dayofweek
    return pd.DataFrame({
        "hour": hour,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dayofweek": dow,
        "is_weekend": (dow >= 5).astype(int),
        "month": month,
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "year": ts.dt.year,
    }, index=ts.index)


def train_cut(n: int, split: dict) -> tuple[int, int]:
    """Row positions where the validation and test blocks start."""
    n_train = int(round(n * split["train"]))
    n_val = int(round(n * split["val"]))
    return n_train, n_train + n_val


def prev_count_and_mean(energy: pd.Series, groups: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Count and mean of *strictly earlier* rows in the same group (rows already time-sorted).

    Equivalent to ``groupby(...).expanding().mean().shift(1)`` but vectorised:
    previous mean = (cumulative sum - current value) / number of previous rows.
    """
    g = energy.groupby(groups, sort=False)
    prev_count = g.cumcount()
    prev_sum = g.cumsum() - energy
    prev_mean = prev_sum / prev_count.replace(0, np.nan)
    return prev_count, prev_mean


def engineer_features(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, float]:
    """Build every model feature from information available at plug-in time.

    Rows are sorted chronologically first so history features only see the past.
    Returns the feature frame and the global *train* mean kWh used as fallback.
    """
    c = cfg["columns"]
    df = df.sort_values(START, kind="mergesort").reset_index(drop=True)

    n_train, _ = train_cut(len(df), cfg["data"]["split"])
    global_mean = float(df[TARGET].iloc[:n_train].mean())

    out = time_features(df[START])
    out[START] = df[START]
    out[TARGET] = df[TARGET]
    out["station"] = df[c["station"]]
    out["port_type"] = df[c["port_type"]]
    out["plug_type"] = df[c["plug_type"]]
    out["port_number"] = df[c["port_number"]]
    out["lat"] = df[c["lat"]]
    out["lon"] = df[c["lon"]]
    out["driver_zip_known"] = (df[c["driver_zip"]] != UNKNOWN).astype(int)
    out["user_id"] = df[c["user"]]          # kept for the web-app sample file; never a model feature

    user = df[c["user"]]
    count, mean = prev_count_and_mean(df[TARGET], user)
    unknown = user == UNKNOWN
    out["user_prev_sessions"] = count.where(~unknown, 0).astype(int)
    out["user_prev_mean_kwh"] = mean.where(~unknown).fillna(global_mean)

    _, st_mean = prev_count_and_mean(df[TARGET], df[c["station"]])
    out["station_prev_mean_kwh"] = st_mean.fillna(global_mean)
    return out, global_mean


# ---------------------------------------------------------------------------
# Steps 7-8: rare categories and chronological split
# ---------------------------------------------------------------------------
def chronological_split(df: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """First 70% of sessions -> train, next 15% -> val, last 15% -> test (no shuffling)."""
    df = df.sort_values(START, kind="mergesort").reset_index(drop=True)
    a, b = train_cut(len(df), split)
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def fit_category_levels(train: pd.DataFrame, min_count: int) -> dict[str, list[str]]:
    """Categorical levels seen at least ``min_count`` times in *train*, plus "other"."""
    levels = {}
    for col in CATEGORICAL_FEATURES:
        counts = train[col].value_counts()
        kept = sorted(str(v) for v in counts[counts >= min_count].index if v != OTHER)
        levels[col] = kept + [OTHER]
    return levels


def apply_category_levels(df: pd.DataFrame, levels: dict[str, list[str]]) -> pd.DataFrame:
    """Map rare or unseen categories to "other"."""
    out = df.copy()
    for col, lv in levels.items():
        allowed = set(lv)
        out[col] = out[col].astype(str).where(out[col].astype(str).isin(allowed), OTHER)
    return out


def to_model_frame(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Ordered feature matrix for tree models; categoricals use fixed ``category`` levels.

    Fixed levels keep the category codes identical at train and serve time.
    """
    X = df[meta["features"]].copy()
    for col in meta["categorical"]:
        X[col] = pd.Categorical(X[col].astype(str), categories=meta["categories"][col])
    for col in meta["numeric"]:
        X[col] = X[col].astype(float)
    return X


def assert_no_leakage(columns: list[str], cfg: dict) -> None:
    """Guard: no final feature may come from a column known only after the session."""
    bad = [c for c in columns if _matches(c, cfg["columns"]["leakage"])
           or c in {cfg["columns"]["energy"], TARGET}]
    assert not bad, f"Leakage columns in features: {bad}"


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_pipeline(raw: pd.DataFrame, cfg: dict) -> dict:
    """Run steps 1-8 in memory. Returns splits, feature metadata and the step log."""
    steps: list[dict] = [{"step": "raw CSV", "rows": len(raw)}]

    def record(name: str, frame: pd.DataFrame) -> None:
        steps.append({"step": name, "rows": len(frame)})
        log.info("%-38s %9d rows", name, len(frame))

    df = parse_start(raw, cfg);                 record("parse start date", df)
    df = filter_period(df, cfg);                record("filter 2012-2019", df)
    df, threshold = clean_target(df, cfg);      record("clean target + outliers", df)
    log.info("outlier threshold (%.3f quantile) = %.3f kWh", cfg["data"]["outlier_quantile"], threshold)
    df, dropped = drop_unused_columns(df, cfg); record("drop leakage/identifier columns", df)
    df = fill_missing(df, cfg);                 record("fill missing values", df)
    feats, global_mean = engineer_features(df, cfg); record("feature engineering", feats)

    train, val, test = chronological_split(feats, cfg["data"]["split"])
    levels = fit_category_levels(train, cfg["data"]["rare_category_min"])
    train, val, test = (apply_category_levels(x, levels) for x in (train, val, test))
    assert_no_leakage(FEATURES, cfg)

    ranges = {name: {"rows": len(x), "start": str(x[START].min()), "end": str(x[START].max())}
              for name, x in (("train", train), ("val", val), ("test", test))}
    for name, r in ranges.items():
        log.info("%-5s %8d rows  %s -> %s", name, r["rows"], r["start"], r["end"])

    meta = {
        "features": FEATURES,
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "categories": levels,
        "rare_mapping": {"min_count": cfg["data"]["rare_category_min"], "other_label": OTHER},
        "global_train_mean_kwh": global_mean,
        "target": TARGET,
    }
    info = {
        "steps": steps,
        "outlier_threshold_kwh": threshold,
        "dropped_columns": dropped,
        "splits": ranges,
        "target_stats": {k: float(v) for k, v in feats[TARGET].describe().items()},
    }
    return {"train": train, "val": val, "test": test, "meta": meta, "info": info}


def load_splits(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load the processed parquet splits and feature metadata."""
    d = processed_dir(cfg)
    if not (d / "train.parquet").exists():
        raise FileNotFoundError(f"No processed data in {d}. Run: python -m src.data.preprocess")
    splits = [pd.read_parquet(d / f"{n}.parquet") for n in ("train", "val", "test")]
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return (*splits, meta)


def save_outputs(result: dict, out_dir: Path) -> None:
    """Step 10: write parquet splits, ``feature_list.json``, metadata and the step log."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        result[name].to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "feature_list.json").write_text(json.dumps(result["meta"]["features"], indent=2))
    (out_dir / "feature_meta.json").write_text(json.dumps(result["meta"], indent=2))
    (out_dir / "preprocess_log.json").write_text(json.dumps(result["info"], indent=2))


def main() -> None:
    from src.data import eda
    from src.data.load import check_columns, load_raw

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    set_seeds(cfg["seed"])
    raw = load_raw(cfg)
    check_columns(raw, cfg)
    result = run_pipeline(raw, cfg)
    save_outputs(result, processed_dir(cfg))
    eda.make_all(raw, result, cfg, results_dir(cfg) / "eda")
    print(f"Saved processed splits to {processed_dir(cfg)} and EDA figures to {results_dir(cfg) / 'eda'}")


if __name__ == "__main__":
    main()
