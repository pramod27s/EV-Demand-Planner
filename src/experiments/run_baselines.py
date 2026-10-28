"""Train the reference models and the default-XGBoost baseline; evaluate on the TEST set.

Run with ``python -m src.experiments.run_baselines`` -> ``results/baseline_metrics.csv``.
"""
from __future__ import annotations

import time

import pandas as pd

from src.config import load_config, results_dir, set_seeds
from src.evaluation.metrics import all_metrics
from src.experiments.common import load_data, write_json
from src.models.baselines import make_default_xgb, make_linear_regression, make_random_forest


def run(cfg: dict) -> pd.DataFrame:
    seed = cfg["seed"]
    set_seeds(seed)
    data = load_data(cfg)
    out = results_dir(cfg)
    (out / "baselines").mkdir(parents=True, exist_ok=True)

    Xtr, ytr = data.X("train"), data.y("train")
    Xva, yva = data.X("val"), data.y("val")
    Xte, yte = data.X("test"), data.y("test")
    raw_tr, raw_va, raw_te = (data.train[data.meta["features"]], data.val[data.meta["features"]],
                              data.test[data.meta["features"]])

    rows = []

    def record(name, model, fit, Xv, Xt):
        t0 = time.perf_counter()
        fit()
        train_time = time.perf_counter() - t0
        row = {"model": name, **all_metrics(yte, model.predict(Xt)),
               "val_rmse": all_metrics(yva, model.predict(Xv))["rmse"], "train_time_s": train_time}
        rows.append(row)
        print(f"{name:<20} test RMSE={row['rmse']:.4f}  R2={row['r2']:.4f}  ({train_time:.1f}s)")

    lr = make_linear_regression(data.meta)
    record("Linear Regression", lr, lambda: lr.fit(raw_tr, ytr), raw_va, raw_te)

    rf = make_random_forest(data.meta, cfg, seed)
    record("Random Forest", rf, lambda: rf.fit(raw_tr, ytr), raw_va, raw_te)
    del rf  # large in memory and not needed later

    xgb = make_default_xgb(seed)
    xgb.set_params(eval_metric="rmse")
    record("XGBoost (default)", xgb,
           lambda: xgb.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False), Xva, Xte)
    xgb.save_model(out / "baselines" / "xgb_default.json")
    ev = xgb.evals_result()
    write_json(out / "baselines" / "xgb_default_evals.json",
               {"train_rmse": ev["validation_0"]["rmse"], "val_rmse": ev["validation_1"]["rmse"]})

    df = pd.DataFrame(rows)
    df.to_csv(out / "baseline_metrics.csv", index=False)
    print(f"Saved {out / 'baseline_metrics.csv'}")
    return df


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()
