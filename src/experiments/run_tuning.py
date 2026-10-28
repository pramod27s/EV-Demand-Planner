"""Tune XGBoost with SMA, PSO, GA and Random Search (identical evaluation budget).

For each optimiser and seed:
1. run the optimiser on the XGBoost fitness (validation RMSE, training subsample);
2. decode the best position into hyperparameters;
3. retrain on the FULL training set with early stopping on validation; evaluate on TEST;
4. save ``results/tuning/{optimizer}_seed{seed}.json`` and the model.

Resumable: a run whose JSON already exists (in the same mode) is skipped.

    python -m src.experiments.run_tuning            # full experiment (config.yaml -> tuning)
    python -m src.experiments.run_tuning --quick    # smoke test: pop 6, iter 5, 1 seed, 10k rows
    python -m src.experiments.run_tuning --optimizers sma pso --seeds 1 2
"""
from __future__ import annotations

import argparse
import time

from src.config import load_config, results_dir, set_seeds
from src.evaluation.metrics import all_metrics, rmse
from src.experiments.common import load_data, read_json, write_json
from src.models.baselines import fit_xgb
from src.models.xgb_space import SearchSpace, XGBFitness
from src.optimizers import LABELS, make_optimizer


def tuning_settings(cfg: dict, quick: bool) -> dict:
    s = dict(cfg["tuning"])
    if quick:
        s.update(cfg["quick"])
    return s


def run_one(key: str, seed: int, data, cfg: dict, settings: dict, quick: bool, mats: dict) -> dict:
    space = SearchSpace()
    set_seeds(seed)
    fitness = XGBFitness(mats["Xtr"], mats["ytr"], mats["Xva"], mats["yva"], seed=seed,
                         sample_size=settings["tuning_sample_size"],
                         early_stopping_rounds=settings["early_stopping_rounds"], space=space)
    opt = make_optimizer(key, settings["pop_size"], settings["max_iter"], seed, cfg)
    res = opt.optimize(fitness, space.dim)
    best_params = space.decode(res.best_position)

    # Re-fit the best configuration on the full training set; evaluate once on test.
    t0 = time.perf_counter()
    model = fit_xgb(best_params, mats["Xtr"], mats["ytr"], mats["Xva"], mats["yva"], seed,
                    settings["early_stopping_rounds"], track_train=True)
    final_train_time = time.perf_counter() - t0
    ev = model.evals_result()

    models_dir = results_dir(cfg) / "tuning" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(models_dir / f"{key}_seed{seed}.json")

    history = [{**h, "best_params": space.decode(h["best_position"])} for h in res.history]
    return {
        "optimizer": key,
        "label": LABELS[key],
        "seed": seed,
        "quick": quick,
        "settings": settings,
        "best_params": best_params,
        "best_position": res.best_position.tolist(),
        "best_val_rmse": res.best_fitness,                     # tuning fitness (subsample model)
        "final_val_rmse": rmse(mats["yva"], model.predict(mats["Xva"])),
        "best_iteration": int(model.best_iteration),
        "test_metrics": all_metrics(mats["yte"], model.predict(mats["Xte"])),
        "convergence": res.convergence,
        "history": history,
        "runtime_s": res.runtime_s,
        "evaluations": res.evaluations,
        "real_evaluations": fitness.real_evaluations,
        "cache_hits": fitness.cache_hits,
        "final_train_time_s": final_train_time,
        "learning_curve": {"train_rmse": ev["validation_0"]["rmse"], "val_rmse": ev["validation_1"]["rmse"]},
        "extras": res.extras,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    ap.add_argument("--optimizers", nargs="+", help="subset of: sma pso ga random")
    ap.add_argument("--seeds", nargs="+", type=int, help="override the seeds")
    args = ap.parse_args(argv)
    run(load_config(), args.quick, args.optimizers, args.seeds)


def run(cfg: dict, quick: bool = False, keys: list[str] | None = None, seeds: list[int] | None = None) -> None:
    settings = tuning_settings(cfg, quick)
    keys = keys or settings["optimizers"]
    seeds = seeds or settings["seeds"]

    data = load_data(cfg)
    mats = {"Xtr": data.X("train"), "ytr": data.y("train"), "Xva": data.X("val"),
            "yva": data.y("val"), "Xte": data.X("test"), "yte": data.y("test")}
    out = results_dir(cfg) / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Tuning {keys} x seeds {seeds} | pop={settings['pop_size']} iter={settings['max_iter']} "
          f"sample={settings['tuning_sample_size']}{' [QUICK]' if quick else ''}")

    for key in keys:
        for seed in seeds:
            path = out / f"{key}_seed{seed}.json"
            if path.exists() and read_json(path).get("quick") == quick:
                print(f"  skip {key} seed {seed} (exists)")
                continue
            t0 = time.perf_counter()
            result = run_one(key, seed, data, cfg, settings, quick, mats)
            write_json(path, result)
            m = result["test_metrics"]
            print(f"  {LABELS[key]:<7} seed {seed}: val RMSE={result['best_val_rmse']:.4f} "
                  f"test RMSE={m['rmse']:.4f} R2={m['r2']:.4f} "
                  f"({result['real_evaluations']} trainings, {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
