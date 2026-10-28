"""Turn baseline + tuning results into tables, figures and ``results/summary.md``.

Run with ``python -m src.experiments.summarize`` (after run_baselines and run_tuning).

Best seed per optimiser = the run with the lowest *validation* RMSE (the test set is never
used for model selection).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.config import load_config, processed_dir, results_dir
from src.evaluation.stats import runs_frame, tuning_summary, wilcoxon_vs
from src.experiments.common import load_data, read_json, write_json
from src.models.xgb_space import SearchSpace
from src.optimizers import LABELS
from src.viz import plots

OPT_ORDER = ["random", "ga", "pso", "sma"]


# ---------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------
def load_runs(out: Path) -> tuple[list[dict], bool]:
    """All tuning JSONs of one mode: full runs if any exist, otherwise quick runs."""
    runs = [read_json(p) for p in sorted((out / "tuning").glob("*_seed*.json"))]
    if not runs:
        raise FileNotFoundError("No tuning results. Run: python -m src.experiments.run_tuning")
    full = [r for r in runs if not r["quick"]]
    return (full, False) if full else (runs, True)


def best_runs(runs: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for r in runs:
        k = r["optimizer"]
        if k not in best or r["best_val_rmse"] < best[k]["best_val_rmse"]:
            best[k] = r
    return best


def load_model(path: Path) -> XGBRegressor:
    m = XGBRegressor()
    m.load_model(path)
    return m


def tuned_model_path(out: Path, run: dict) -> Path:
    return out / "tuning" / "models" / f"{run['optimizer']}_seed{run['seed']}.json"


# ---------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------
def final_comparison(baseline: pd.DataFrame, runs_df: pd.DataFrame, best: dict[str, dict]) -> pd.DataFrame:
    """Rows: LR, RF, default XGB, XGB + Random/GA/PSO/SMA (best seed + mean ± std over seeds)."""
    default_rmse = float(baseline.loc[baseline.model == "XGBoost (default)", "rmse"].iloc[0])
    rows = []
    for _, b in baseline.iterrows():
        rows.append({"model": b["model"], "rmse": b["rmse"], "mae": b["mae"], "r2": b["r2"],
                     "smape": b["smape"], "train_time_s": b["train_time_s"], "n_seeds": 1})
    for k in [k for k in OPT_ORDER if k in best]:
        r, g = best[k], runs_df[runs_df.optimizer == k]
        row = {"model": f"XGBoost + {LABELS[k]}", **{m: r["test_metrics"][m] for m in ("rmse", "mae", "r2", "smape")},
               "train_time_s": r["final_train_time_s"], "tuning_time_s": r["runtime_s"],
               "best_seed": r["seed"], "n_seeds": len(g)}
        for m in ("rmse", "mae", "r2", "smape"):
            row[f"{m}_mean"] = g[f"test_{m}"].mean()
            row[f"{m}_std"] = g[f"test_{m}"].std(ddof=1) if len(g) > 1 else 0.0
        rows.append(row)
    df = pd.DataFrame(rows)
    df["rmse_improvement_pct"] = 100 * (default_rmse - df["rmse"]) / default_rmse
    if "rmse_mean" in df:
        df["rmse_mean_improvement_pct"] = 100 * (default_rmse - df["rmse_mean"]) / default_rmse
    return df


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------
FIGURE_CAPTIONS = {
    "convergence_mean.png": "Mean ± std best-so-far validation RMSE per iteration for SMA, PSO, GA and Random Search.",
    "convergence_sma_runs.png": "Convergence of every SMA seed.",
    "learning_curve_default_vs_sma.png": "Train/validation RMSE per boosting round: default vs SMA-tuned XGBoost.",
    "baseline_vs_optimized_bar.png": "Test RMSE, MAE and R² of the baselines and the optimised XGBoost models.",
    "boxplot_test_rmse.png": "Distribution of test RMSE across seeds for each optimiser.",
    "pred_vs_actual_sma.png": "Predicted vs actual energy on the test set (SMA-tuned XGBoost, 5,000 sessions).",
    "residuals_sma.png": "Residual histogram and residuals vs predicted (SMA-tuned XGBoost, test set).",
    "feature_importance_sma.png": "Top-15 gain importances of the SMA-tuned XGBoost.",
    "hyperparams_comparison.png": "Best hyperparameters of each optimiser, normalised to the search range.",
    "runtime_vs_rmse.png": "Mean tuning runtime vs mean test RMSE per optimiser.",
    "sma_weights_explained.png": "SMA weights W (vein thickness) vs fitness rank for one iteration.",
    "benchmarks/*.png": "Convergence of the optimisers on Sphere, Rastrigin, Ackley, Rosenbrock and Griewank.",
}


def run(cfg: dict) -> None:
    out = results_dir(cfg)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(out / "baseline_metrics.csv")
    runs, quick = load_runs(out)
    best = best_runs(runs)
    runs_df = runs_frame(runs)

    # ---- tables -----------------------------------------------------------------
    summary = tuning_summary(runs_df)
    summary.to_csv(out / "tuning_summary.csv", index=False)
    wil = wilcoxon_vs(runs_df, others=[k for k in ("pso", "ga", "random") if k in best])
    wil.to_csv(out / "wilcoxon.csv", index=False)
    comp = final_comparison(baseline, runs_df, best)
    comp.to_csv(out / "final_comparison.csv", index=False)
    space = SearchSpace()
    hp = pd.DataFrame([{"optimizer": LABELS[k], "seed": best[k]["seed"], **best[k]["best_params"],
                        "best_iteration": best[k]["best_iteration"]} for k in OPT_ORDER if k in best])
    hp.to_csv(out / "best_hyperparameters.csv", index=False)

    # ---- figures ----------------------------------------------------------------
    curves = {k: [r["convergence"] for r in runs if r["optimizer"] == k] for k in best}
    plots.convergence_mean(curves, cfg, fig_dir / "convergence_mean.png")
    if "sma" in best:
        sma_runs = sorted([r for r in runs if r["optimizer"] == "sma"], key=lambda r: r["seed"])
        plots.convergence_runs([r["convergence"] for r in sma_runs], [r["seed"] for r in sma_runs],
                               "sma", cfg, fig_dir / "convergence_sma_runs.png")
        default_lc = read_json(out / "baselines" / "xgb_default_evals.json")
        plots.learning_curves(default_lc, best["sma"]["learning_curve"], cfg,
                              fig_dir / "learning_curve_default_vs_sma.png")

        data = load_data(cfg)
        Xte, yte = data.X("test"), data.y("test")
        sma_model = load_model(tuned_model_path(out, best["sma"]))
        yhat = sma_model.predict(Xte)
        idx = np.random.default_rng(cfg["seed"]).choice(len(yte), size=min(5000, len(yte)), replace=False)
        plots.pred_vs_actual(yte[idx], yhat[idx], cfg, fig_dir / "pred_vs_actual_sma.png")
        plots.residuals(yte, yhat, cfg, fig_dir / "residuals_sma.png")
        gain = sma_model.get_booster().get_score(importance_type="gain")
        plots.feature_importance(gain, cfg, fig_dir / "feature_importance_sma.png")
        snap = best["sma"].get("extras", {}).get("weights_snapshot")
        if snap:
            plots.sma_weights(snap, cfg, fig_dir / "sma_weights_explained.png")

    plots.comparison_bars(comp, cfg, fig_dir / "baseline_vs_optimized_bar.png")
    default_rmse = float(baseline.loc[baseline.model == "XGBoost (default)", "rmse"].iloc[0])
    plots.boxplot_test_rmse(runs_df, cfg, fig_dir / "boxplot_test_rmse.png", default_rmse)
    norm = {k: space.normalise(best[k]["best_params"]) for k in best}
    plots.hyperparams_comparison(norm, space.names, cfg, fig_dir / "hyperparams_comparison.png")
    summary_keyed = summary.assign(key=[k for k in summary["optimizer"].map({v: k for k, v in LABELS.items()})])
    plots.runtime_vs_rmse(summary_keyed, cfg, fig_dir / "runtime_vs_rmse.png")

    write_summary_md(cfg, out, comp, baseline, summary, wil, hp, quick)
    print(f"Wrote tables, figures and {out / 'summary.md'}" + ("  [QUICK-mode results]" if quick else ""))


def write_summary_md(cfg, out, comp, baseline, summary, wil, hp, quick) -> None:
    info = read_json(processed_dir(cfg) / "preprocess_log.json")
    lines = ["# Results summary", ""]
    if quick:
        lines += ["> **Note:** these numbers come from `run_tuning --quick` (smoke test), not the full experiment.", ""]

    lines += ["## Dataset", "", "Rows after each preprocessing step:", "",
              md_table(pd.DataFrame(info["steps"]), "{:.0f}"), "",
              f"Outlier threshold (99.5th percentile): **{info['outlier_threshold_kwh']:.2f} kWh**.", "",
              "Chronological splits:", "",
              md_table(pd.DataFrame([{"split": k, **v} for k, v in info["splits"].items()]), "{:.0f}"), ""]

    lines += ["## Baseline models (test set, no optimisation)", "",
              md_table(baseline[["model", "rmse", "mae", "r2", "smape", "train_time_s"]]), ""]

    cols = ["model", "rmse", "mae", "r2", "smape", "rmse_improvement_pct", "train_time_s"]
    extra = [c for c in ("rmse_mean", "rmse_std", "n_seeds") if c in comp]
    lines += ["## Final comparison (test set)", "",
              "Tuned rows use each optimiser's best seed (selected on validation RMSE); "
              "`rmse_mean`/`rmse_std` are over all seeds. `rmse_improvement_pct` is relative to default XGBoost.", "",
              md_table(comp[cols + extra]), ""]

    lines += ["## Tuning summary (across seeds)", "",
              md_table(summary[["optimizer", "n_runs", "val_rmse_mean", "val_rmse_std", "test_rmse_mean",
                                "test_rmse_std", "test_rmse_best", "test_rmse_worst", "test_r2_mean",
                                "runtime_s_mean", "real_evaluations_mean"]]), ""]

    lines += ["## Best hyperparameters", "", md_table(hp), ""]

    lines += ["## Wilcoxon signed-rank test (test RMSE, paired by seed)", "",
              md_table(wil), "",
              "With 5 seeds the smallest possible two-sided p-value is 0.0625, so the test has low power; "
              "p-values are reported as they are.", ""]

    bench = out / "benchmarks" / "benchmark_stats.csv"
    if bench.exists():
        lines += ["## Benchmark functions (D = 10, 10 seeds, final best fitness)", "",
                  md_table(pd.read_csv(bench), "{:.4g}"), ""]

    lines += ["## Figures", ""] + [f"- `figures/{f}` — {c}" if not f.startswith("benchmarks")
                                   else f"- `{f}` — {c}" for f, c in FIGURE_CAPTIONS.items()]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    run(load_config())


if __name__ == "__main__":
    main()
