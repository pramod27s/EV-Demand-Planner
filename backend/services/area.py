"""Analyse a user's own area CSV: energy estimates, per-location demand and SMA-suggested
charger sites.

Two kinds of file are accepted (columns are recognised by common names, see ``ALIASES``):

* **Session files** — one row per charging session. Required: start time, latitude, longitude.
  Every session gets an energy estimate from the SMA-tuned model, built with the same feature
  functions as training (``src/data/preprocess.py``); the model is never retrained here.
* **Demand files** — kWh already measured per location (and optionally per month), e.g. the
  monthly consumption Indian utilities publish. Required: latitude, longitude, energy; no start
  time. The model is not needed: the measured kWh drive the demand map and the placement.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from src.data.preprocess import (UNKNOWN, apply_category_levels, prev_count_and_mean, time_features,
                                 to_model_frame)
from src.evaluation.metrics import all_metrics
from src.placement import haversine_km, solve_placement

PALO_ALTO = (37.44, -122.16)
FAR_FROM_TRAINING_KM = 50
DAYS_PER_MONTH = 30.44

# canonical name -> accepted column names (compared after lower-casing and removing non-alphanumerics)
ALIASES = {
    "start": ["startdate", "starttime", "start", "startdatetime", "timestamp", "datetime", "plugintime", "date"],
    "lat": ["latitude", "lat"],
    "lon": ["longitude", "lon", "lng", "long"],
    "station": ["stationname", "station", "stationid", "site", "sitename", "location", "locationname"],
    "port_type": ["porttype", "level", "chargerlevel", "chargertype"],
    "plug_type": ["plugtype", "plug", "connector", "connectortype"],
    "port_number": ["portnumber", "port", "portno"],
    "user": ["userid", "user", "driverid", "driver", "customerid"],
    "energy": ["energykwh", "energy", "kwh", "energyconsumed", "units", "unitskwh", "demandkwh"],
    "month": ["month", "period", "billingmonth", "yearmonth"],
}
SESSION_REQUIRED = ["start", "lat", "lon"]
DEMAND_REQUIRED = ["lat", "lon", "energy"]
DEFAULTS = {"port_type": "Level 2", "plug_type": "J1772", "port_number": 1}


class AreaInputError(ValueError):
    """The uploaded file cannot be analysed (shown to the user as a 422)."""


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def read_csv(raw: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", low_memory=False)
    except Exception as exc:  # noqa: BLE001 - any parse failure is a user error
        raise AreaInputError(f"Could not read the file as CSV: {exc}") from exc
    if df.empty:
        raise AreaInputError("The file has no rows.")
    return df


def find_columns(df: pd.DataFrame) -> dict[str, str]:
    """canonical name -> column in the file (first match wins)."""
    by_norm = {_norm(c): c for c in df.columns}
    found = {}
    for key, names in ALIASES.items():
        for n in names:
            if n in by_norm:
                found[key] = by_norm[n]
                break
    return found


def is_demand_file(cols: dict[str, str]) -> bool:
    return "start" not in cols and all(k in cols for k in DEMAND_REQUIRED)


def require(cols: dict[str, str], required: list[str], df: pd.DataFrame) -> None:
    missing = [k for k in required if k not in cols]
    if missing:
        accepted = {k: ALIASES[k][:4] for k in missing}
        raise AreaInputError(f"Missing required column(s) {missing}. Accepted names include: {accepted}. "
                             f"Columns in your file: {list(df.columns)[:20]}")


def _locations_by_coords(out: pd.DataFrame) -> pd.Series:
    """Names for rows without a location column: group by coordinates (~10 m)."""
    key = out["lat"].round(4).astype(str) + "," + out["lon"].round(4).astype(str)
    names = {k: f"Location {i + 1}" for i, k in enumerate(pd.unique(key))}
    return key.map(names)


def _place(loc: pd.DataFrame, k: int, cfg: dict) -> tuple[dict, int]:
    """SMA placement on the highest-demand locations (weights = total kWh)."""
    top = loc.head(cfg["max_locations"])
    k = int(min(k, cfg["max_sites"], len(top)))
    place = solve_placement(top["lat"], top["lon"], top["total_kwh"], k, cfg["pop_size"], cfg["max_iter"], cfg["seed"])
    return {"k": k, "sites": place["sites"], "objective_km": place["objective_km"],
            "convergence_km": place["convergence_km"], "evaluations": place["evaluations"],
            "locations_considered": int(len(top))}, k


def _centre(loc: pd.DataFrame) -> tuple[float, float]:
    w = loc["total_kwh"] + 1e-9
    return float(np.average(loc["lat"], weights=w)), float(np.average(loc["lon"], weights=w))


def _val(c: str, v):
    if isinstance(v, (str, list)):
        return v
    return None if pd.isna(v) else (int(v) if c in ("sessions", "peak_hour") else float(v))


# ---------------------------------------------------------------------------------
# Session files
# ---------------------------------------------------------------------------------
def clean(df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """Canonical columns; rows without a valid time or coordinates are dropped."""
    out = pd.DataFrame({
        "start_time": pd.to_datetime(df[cols["start"]].astype(str), format="mixed", errors="coerce"),
        "lat": pd.to_numeric(df[cols["lat"]], errors="coerce"),
        "lon": pd.to_numeric(df[cols["lon"]], errors="coerce"),
    })
    for key in ("port_type", "plug_type"):
        out[key] = df[cols[key]].fillna(DEFAULTS[key]).astype(str).str.strip() if key in cols else DEFAULTS[key]
    out["port_number"] = (pd.to_numeric(df[cols["port_number"]], errors="coerce").fillna(1)
                          if "port_number" in cols else DEFAULTS["port_number"])
    out["user"] = df[cols["user"]].astype(str).where(df[cols["user"]].notna(), UNKNOWN) if "user" in cols else UNKNOWN
    if "energy" in cols:
        out["energy_kwh"] = pd.to_numeric(df[cols["energy"]], errors="coerce")
    if "station" in cols:
        out["location"] = df[cols["station"]].astype(str).str.strip()
    out = out[out["start_time"].notna() & out["lat"].between(-90, 90) & out["lon"].between(-180, 180)]
    if "location" not in out:
        out["location"] = _locations_by_coords(out)
    return out.sort_values("start_time", kind="mergesort").reset_index(drop=True)


def build_features(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Model features for every row, using the training-time feature functions."""
    gm = meta["global_train_mean_kwh"]
    f = time_features(df["start_time"])
    f["station"] = df["location"]          # unseen station names map to "other"
    for c in ("port_type", "plug_type", "port_number", "lat", "lon"):
        f[c] = df[c]
    f["driver_zip_known"] = 1
    has_energy = "energy_kwh" in df and df["energy_kwh"].notna().any()
    if has_energy:
        e = df["energy_kwh"].fillna(gm)
        cnt, mean = prev_count_and_mean(e, df["user"])
        known = df["user"] != UNKNOWN
        f["user_prev_sessions"] = cnt.where(known, 0)
        f["user_prev_mean_kwh"] = mean.where(known).fillna(gm)
        f["station_prev_mean_kwh"] = prev_count_and_mean(e, df["location"])[1].fillna(gm)
    else:
        f["user_prev_sessions"] = 0
        f["user_prev_mean_kwh"] = gm
        f["station_prev_mean_kwh"] = gm
    return to_model_frame(apply_category_levels(f, meta["categories"]), meta)


def predict_frame(df_in: pd.DataFrame, cols: dict[str, str], model, meta: dict) -> pd.DataFrame:
    """Cleaned session rows with a ``predicted_kwh`` column."""
    require(cols, SESSION_REQUIRED, df_in)
    df = clean(df_in, cols)
    if df.empty:
        raise AreaInputError("No usable rows: every row is missing a valid start time or latitude/longitude.")
    df["predicted_kwh"] = np.clip(model.predict(build_features(df, meta)), 0, None)
    return df


def analyse_sessions(df_in: pd.DataFrame, cols: dict[str, str], model, meta: dict, k: int, cfg: dict) -> dict:
    df = predict_frame(df_in, cols, model, meta)
    n_in = len(df_in)
    days = max(1.0, (df["start_time"].max() - df["start_time"].min()).total_seconds() / 86400)
    df["hour"] = df["start_time"].dt.hour

    g = df.groupby("location", sort=False)
    loc = pd.DataFrame({
        "lat": g["lat"].median(), "lon": g["lon"].median(), "sessions": g.size(),
        "total_kwh": g["predicted_kwh"].sum(), "mean_kwh": g["predicted_kwh"].mean(),
    })
    loc["daily_kwh"] = loc["total_kwh"] / days
    by_hour = df.groupby(["location", "hour"])["predicted_kwh"]
    loc["peak_hour"] = by_hour.sum().unstack().idxmax(axis=1)
    hourly_mean = by_hour.mean().unstack().reindex(columns=range(24))
    loc["port_type"] = g["port_type"].agg(lambda x: x.mode().iat[0])
    loc["plug_type"] = g["plug_type"].agg(lambda x: x.mode().iat[0])
    has_actual = "energy_kwh" in df and df["energy_kwh"].notna().any()
    if has_actual:
        loc["mean_actual_kwh"] = g["energy_kwh"].mean()
    loc = loc.sort_values("total_kwh", ascending=False)

    hourly = df.groupby("hour").agg(sessions=("predicted_kwh", "size"), total_kwh=("predicted_kwh", "sum"),
                                    mean_kwh=("predicted_kwh", "mean")).reindex(range(24), fill_value=0)
    placement, _ = _place(loc, k, cfg)
    centre = _centre(loc)
    dist_pa = float(haversine_km(centre[0], centre[1], *PALO_ALTO))

    summary = {
        "mode": "sessions",
        "rows_in": n_in, "rows_used": int(len(df)), "rows_dropped": int(n_in - len(df)),
        "n_locations": int(len(loc)), "date_start": str(df["start_time"].min()), "date_end": str(df["start_time"].max()),
        "days": round(days, 1), "total_kwh": float(df["predicted_kwh"].sum()), "mean_kwh": float(df["predicted_kwh"].mean()),
        "daily_kwh": float(df["predicted_kwh"].sum() / days), "has_actual": bool(has_actual),
        "centre": {"lat": centre[0], "lon": centre[1]}, "distance_from_training_km": dist_pa,
        "far_from_training": dist_pa > FAR_FROM_TRAINING_KM,
    }
    if has_actual:
        m = df["energy_kwh"].notna()
        summary["accuracy"] = all_metrics(df.loc[m, "energy_kwh"].to_numpy(), df.loc[m, "predicted_kwh"].to_numpy())

    locations = [{"name": str(name), **{c: _val(c, v) for c, v in row.items()},
                  "hourly_mean_kwh": [None if pd.isna(v) else float(v) for v in hourly_mean.loc[name]]}
                 for name, row in loc.iterrows()]
    return {
        "summary": summary,
        "locations": locations,
        "hourly": [{"hour": int(h), **{c: float(v) for c, v in r.items()}} for h, r in hourly.iterrows()],
        "monthly": [],
        "placement": placement,
    }


# ---------------------------------------------------------------------------------
# Demand files (measured kWh per location, optionally per month)
# ---------------------------------------------------------------------------------
def analyse_demand(df_in: pd.DataFrame, cols: dict[str, str], k: int, cfg: dict) -> dict:
    out = pd.DataFrame({
        "lat": pd.to_numeric(df_in[cols["lat"]], errors="coerce"),
        "lon": pd.to_numeric(df_in[cols["lon"]], errors="coerce"),
        "kwh": pd.to_numeric(df_in[cols["energy"]], errors="coerce"),
    })
    if "month" in cols:
        out["month"] = pd.to_datetime(df_in[cols["month"]].astype(str), format="mixed", errors="coerce").dt.to_period("M")
    if "station" in cols:
        out["location"] = df_in[cols["station"]].astype(str).str.strip()
    out = out[out["lat"].between(-90, 90) & out["lon"].between(-180, 180) & (out["kwh"] > 0)]
    if "month" in out:
        out = out[out["month"].notna()]
    if out.empty:
        raise AreaInputError("No usable rows: every row is missing coordinates or a positive energy value.")
    if "location" not in out:
        out["location"] = _locations_by_coords(out)

    n_months = int(out["month"].nunique()) if "month" in out else None
    days = n_months * DAYS_PER_MONTH if n_months else 1.0
    g = out.groupby("location", sort=False)
    loc = pd.DataFrame({"lat": g["lat"].median(), "lon": g["lon"].median(), "sessions": g.size(),
                        "total_kwh": g["kwh"].sum()})
    loc["mean_kwh"] = loc["total_kwh"] / (n_months or 1)      # kWh per month (or per file if undated)
    loc["daily_kwh"] = loc["total_kwh"] / days
    loc = loc.sort_values("total_kwh", ascending=False)

    monthly, per_loc = [], {}
    if n_months:
        m = out.groupby("month")["kwh"].sum().sort_index()
        monthly = [{"month": str(p), "total_kwh": float(v)} for p, v in m.items()]
        lm = out.groupby(["location", "month"])["kwh"].sum()
        for (name, p), v in lm.items():
            per_loc.setdefault(name, []).append({"month": str(p), "kwh": float(v)})

    placement, _ = _place(loc, k, cfg)
    centre = _centre(loc)
    summary = {
        "mode": "demand",
        "rows_in": int(len(df_in)), "rows_used": int(len(out)), "rows_dropped": int(len(df_in) - len(out)),
        "n_locations": int(len(loc)),
        "date_start": str(out["month"].min()) if n_months else "", "date_end": str(out["month"].max()) if n_months else "",
        "months": n_months, "days": round(days, 1), "total_kwh": float(out["kwh"].sum()),
        "mean_kwh": float(loc["mean_kwh"].mean()), "daily_kwh": float(out["kwh"].sum() / days), "has_actual": True,
        "centre": {"lat": centre[0], "lon": centre[1]},
        "distance_from_training_km": float(haversine_km(centre[0], centre[1], *PALO_ALTO)),
        "far_from_training": False,          # measured kWh: the Palo Alto model is not used
    }
    locations = [{"name": str(name), **{c: _val(c, v) for c, v in row.items()}, "peak_hour": None,
                  "port_type": "", "plug_type": "", "hourly_mean_kwh": [None] * 24,
                  "monthly_kwh": per_loc.get(name, [])} for name, row in loc.iterrows()]
    return {"summary": summary, "locations": locations, "hourly": [], "monthly": monthly, "placement": placement}


# ---------------------------------------------------------------------------------
# Entry points used by the router
# ---------------------------------------------------------------------------------
def analyse(raw: bytes, model, meta: dict, k: int, placement_cfg: dict) -> dict:
    df_in = read_csv(raw)
    cols = find_columns(df_in)
    if is_demand_file(cols):
        return analyse_demand(df_in, cols, k, placement_cfg)
    return analyse_sessions(df_in, cols, model, meta, k, placement_cfg)


def predictions_csv(raw: bytes, model, meta: dict) -> str:
    df_in = read_csv(raw)
    cols = find_columns(df_in)
    if is_demand_file(cols):
        raise AreaInputError("This is a demand file (measured kWh, no sessions), so there are no session estimates to download.")
    df = predict_frame(df_in, cols, model, meta)
    out_cols = ["start_time", "location", "lat", "lon", "port_type", "plug_type", "port_number"]
    if "energy_kwh" in df:
        out_cols.append("energy_kwh")
    out = df[out_cols + ["predicted_kwh"]].copy()
    out["predicted_kwh"] = out["predicted_kwh"].round(3)
    return out.to_csv(index=False)
