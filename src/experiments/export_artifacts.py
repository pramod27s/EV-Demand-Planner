"""Export everything the web app needs into ``results/models`` and ``results/api``.

Run with ``python -m src.experiments.export_artifacts`` after ``summarize``. The web app
never trains or tunes: it only reads these files, so its numbers match the report.
"""
from __future__ import annotations

import math
import shutil

import numpy as np
import pandas as pd

from src.config import load_config, processed_dir, results_dir
from src.data.preprocess import OTHER, START, TARGET
from src.experiments.common import load_data, read_json, write_json
from src.experiments.summarize import OPT_ORDER, best_runs, load_model, load_runs, tuned_model_path
from src.optimizers import LABELS

API_MODELS = ["sma", "pso", "ga"]   # tuned models served by /api/predict (plus the default)


def _clean(obj):
    """Replace NaN/inf with None so the JSON is valid."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def stations_table(full: pd.DataFrame, test: pd.DataFrame, test_pred: np.ndarray, meta: dict) -> list[dict]:
    """Per-station metadata, usage statistics and the latest history feature."""
    test = test.assign(pred=test_pred)
    rows = []
    for name in meta["categories"]["station"]:
        if name == OTHER:
            continue
        g = full[full.station == name].sort_values(START)
        if g.empty:
            continue
        t = test[test.station == name]
        hourly = g.groupby("hour")[TARGET].mean().reindex(range(24))
        rows.append({
            "name": name,
            "lat": float(g["lat"].median()),
            "lon": float(g["lon"].median()),
            "port_types": sorted(v for v in g["port_type"].unique() if v != OTHER),
            "plug_types": sorted(v for v in g["plug_type"].unique() if v != OTHER),
            "port_numbers": sorted(int(p) for p in g["port_number"].dropna().unique()),
            "session_count": int(len(g)),
            "first_session": str(g[START].iloc[0]),
            "last_session": str(g[START].iloc[-1]),
            "mean_actual_kwh": float(t[TARGET].mean()) if len(t) else float(g[TARGET].mean()),
            "mean_predicted_kwh": float(t["pred"].mean()) if len(t) else None,
            "test_sessions": int(len(t)),
            # value the NEXT session at this station would see (mean of all sessions so far)
            "station_prev_mean_kwh": float(g[TARGET].mean()),
            "hourly_mean_kwh": [None if np.isnan(v) else float(v) for v in hourly.to_numpy()],
        })
    return rows


def run(cfg: dict) -> None:
    out = results_dir(cfg)
    models_dir, api_dir = out / "models", out / "api"
    models_dir.mkdir(parents=True, exist_ok=True)
    api_dir.mkdir(parents=True, exist_ok=True)

    runs, quick = load_runs(out)
    best = best_runs(runs)
    data = load_data(cfg)
    meta = data.meta
    comp = pd.read_csv(out / "final_comparison.csv")

    # ---- models + feature metadata ------------------------------------------------
    shutil.copy(out / "baselines" / "xgb_default.json", models_dir / "xgb_default.json")
    for k in API_MODELS:
        if k in best:
            shutil.copy(tuned_model_path(out, best[k]), models_dir / f"xgb_{k}.json")
    write_json(models_dir / "feature_meta.json", meta)

    # ---- predictions on the test set ------------------------------------------------
    Xte, yte = data.X("test"), data.y("test")
    pred_default = load_model(models_dir / "xgb_default.json").predict(Xte)
    pred_sma = load_model(models_dir / "xgb_sma.json").predict(Xte)

    full = pd.concat([data.train, data.val, data.test], ignore_index=True)
    write_json(api_dir / "stations.json", _clean(stations_table(full, data.test, pred_sma, meta)))

    write_json(api_dir / "comparison.json", _clean({"quick": quick, "rows": comp.to_dict(orient="records")}))

    conv = {}
    for k in [k for k in OPT_ORDER if k in best]:
        mine = sorted([r for r in runs if r["optimizer"] == k], key=lambda r: r["seed"])
        arr = np.array([r["convergence"] for r in mine])
        conv[k] = {"label": LABELS[k], "mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
                   "seeds": {str(r["seed"]): r["convergence"] for r in mine}}
        for r in mine:
            write_json(api_dir / f"history_{k}_seed{r['seed']}.json", {
                "optimizer": k, "label": LABELS[k], "seed": r["seed"],
                "history": [{"iteration": h["iteration"], "best_fitness": h["best_fitness"],
                             "mean_fitness": h["mean_fitness"], "best_params": h["best_params"]}
                            for h in r["history"]],
            })
    write_json(api_dir / "convergence.json", conv)

    write_json(api_dir / "hyperparams.json", {
        k: {"label": LABELS[k], "seed": best[k]["seed"], "best_val_rmse": best[k]["best_val_rmse"],
            "best_iteration": best[k]["best_iteration"], "params": best[k]["best_params"]}
        for k in OPT_ORDER if k in best})

    idx = np.sort(np.random.default_rng(cfg["seed"]).choice(len(yte), size=min(2000, len(yte)), replace=False))
    write_json(api_dir / "test_sample.json", {
        "actual": yte[idx].round(4).tolist(),
        "default": pred_default[idx].round(4).tolist(),
        "sma": pred_sma[idx].round(4).tolist(),
    })

    # Sample file for the "Analyse my area" page: a continuous stretch of test sessions (so driver
    # history can be rebuilt from the file itself), with raw-style column names and the actual energy.
    test_known = data.test[data.test.station != OTHER].sort_values(START)
    cols = {"station": "Station Name", "lat": "Latitude", "lon": "Longitude", START: "Start Date",
            "port_type": "Port Type", "plug_type": "Plug Type", "port_number": "Port Number",
            "user_id": "User ID", TARGET: "Energy (kWh)"}
    test_known.head(8000).rename(columns=cols)[[v for k, v in cols.items() if k in test_known]].to_csv(
        api_dir / "sample_upload.csv", index=False)

    gain = load_model(models_dir / "xgb_sma.json").get_booster().get_score(importance_type="gain")
    top = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:15]
    write_json(api_dir / "feature_importance.json", [{"feature": f, "gain": float(g)} for f, g in top])

    info = read_json(processed_dir(cfg) / "preprocess_log.json")
    write_json(api_dir / "dataset_summary.json", _clean({
        "steps": info["steps"], "splits": info["splits"], "target_stats": info["target_stats"],
        "outlier_threshold_kwh": info["outlier_threshold_kwh"],
        "dropped_columns": info["dropped_columns"], "n_sessions": int(len(full)),
        "n_stations": len([s for s in meta["categories"]["station"] if s != OTHER]),
        "quick": quick,
    }))
    print(f"Exported models to {models_dir} and API files to {api_dir}" + ("  [QUICK]" if quick else ""))


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()
