"""Generates SMA_XGBoost_EV_Demand.ipynb (kept as code so the notebook is easy to review).

    python scripts/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/SMA_XGBoost_EV_Demand.ipynb
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Slime Mould Algorithm–Optimised XGBoost for EV Charging Energy Demand
**Bio-Inspired Algorithms (23DS7BSBIO) AAT — B.E. CSE (Data Science), BMSCE Bengaluru**

End-to-end walkthrough for the demo. All logic lives in `src/`; this notebook imports it and
shows the results produced by the pipeline (`python -m src.experiments.*`).

**Problem.** Given information available at the moment an EV plugs in — station and port, plug type,
time and day, and the driver's past charging behaviour — predict the energy (kWh) the session will consume.
XGBoost is the model; the Slime Mould Algorithm (SMA) tunes its hyperparameters and is compared with PSO,
GA, random search and the untuned model."""),
    code("""import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
from IPython.display import Image, display, Markdown
from src.config import load_config, results_dir
cfg = load_config()
RES = results_dir(cfg)
pd.set_option("display.precision", 4)
print("Results directory:", RES)"""),

    md("""## 1. Dataset and preprocessing
City of Palo Alto *EV Charging Station Usage (July 2011 – Dec 2020)*. We keep 2012–2019, clean the target,
drop **leakage** columns (known only after the session ends, e.g. charging time, fees, GHG savings) and
split **chronologically** 70/15/15."""),
    code("""info = json.loads((ROOT / cfg["paths"]["processed_dir"] / "preprocess_log.json").read_text())
display(pd.DataFrame(info["steps"]))
print(f"Outlier threshold: {info['outlier_threshold_kwh']:.2f} kWh")
display(pd.DataFrame(info["splits"]).T)
print("Dropped columns:", ", ".join(info["dropped_columns"]))"""),
    code("""from src.data.preprocess import FEATURES, assert_no_leakage, load_splits
train, val, test, meta = load_splits(cfg)
assert_no_leakage(meta["features"], cfg)          # raises if any leakage column slipped in
print(len(FEATURES), "features:", FEATURES)
train[FEATURES + ["energy_kwh"]].head()"""),
    md("### Exploratory analysis"),
    code("""for f in ["target_distribution", "sessions_per_year", "mean_energy_by_hour", "mean_energy_by_weekday"]:
    display(Image(filename=str(RES / "eda" / f"{f}.png"), width=720))"""),

    md("""## 2. Baselines (no optimisation)
Linear Regression and Random Forest as references; **default XGBoost** is the baseline the optimisers must beat."""),
    code("""pd.read_csv(RES / "baseline_metrics.csv")"""),

    md("""## 3. The Slime Mould Algorithm
*Physarum polycephalum* reinforces tubes leading to rich food. SMA models this with fitness-ranked
**weights** (vein thickness), an **approach** move toward the best food, a **wrap/contract** move and
random **relocation** (probability z = 0.03). The implementation is `src/optimizers/sma.py` (pure NumPy)."""),
    code("""from src.optimizers import SMA, make_optimizer
from src.optimizers.benchmarks import BENCHMARKS
import inspect
print(inspect.getsource(SMA.optimize))"""),
    md("### Sanity check on benchmark functions (D = 10, 10 seeds)"),
    code("""res = SMA(pop_size=20, max_iter=200, seed=1).optimize(BENCHMARKS["sphere"], dim=5)
print(f"SMA on Sphere (D=5): best fitness = {res.best_fitness:.2e} after {res.evaluations} evaluations")
bench = RES / "benchmarks" / "benchmark_stats.csv"
display(pd.read_csv(bench)) if bench.exists() else print("run: python -m src.experiments.run_benchmarks")"""),
    md("""### Vein thickness: the weight W of every cell
The better half of the population gets W > 1, the worse half W < 1."""),
    code("""sma = SMA(6, 1, seed=0)
S = np.array([5.2, 4.9, 5.0, 5.6, 5.1, 5.3])        # e.g. validation RMSEs of 6 cells
order = np.argsort(S)
W = sma.compute_weights(S, order, dim=3)
pd.DataFrame(W, columns=["W_d1", "W_d2", "W_d3"]).assign(fitness=S, rank=np.argsort(order)).sort_values("rank")"""),

    md("""## 4. The XGBoost search space (9-D)
Every optimiser searches u ∈ [0,1]^9; `SearchSpace.decode` maps it to hyperparameters
(learning rate and the L1/L2 penalties are log-uniform)."""),
    code("""from src.models.xgb_space import SearchSpace
space = SearchSpace()
pd.DataFrame({"u = 0": space.decode(np.zeros(9)), "u = 0.5": space.decode(np.full(9, .5)), "u = 1": space.decode(np.ones(9))})"""),
    md("""### A tiny live tuning run
Fitness = validation RMSE of XGBoost trained on a training subsample. (The real experiment uses
pop 15 × 20 iterations × 5 seeds with a 40k-row subsample; here 4 × 3 on 5k rows so it runs in seconds.)"""),
    code("""from src.experiments.common import Data
from src.models.xgb_space import XGBFitness
d = Data(train, val, test, meta)
fit = XGBFitness(d.X("train"), d.y("train"), d.X("val"), d.y("val"), seed=1, sample_size=5000, early_stopping_rounds=20)
demo = SMA(pop_size=4, max_iter=3, seed=1).optimize(fit, space.dim)
print("convergence (val RMSE):", [round(c, 4) for c in demo.convergence])
print("best hyperparameters:", space.decode(demo.best_position))"""),

    md("## 5. Results of the full experiment"),
    code("""summary_md = RES / "summary.md"
if summary_md.exists() and "quick" in summary_md.read_text()[:300]:
    display(Markdown("> **Note:** results below are from the quick smoke test."))
comp = pd.read_csv(RES / "final_comparison.csv")
comp[["model", "rmse", "mae", "r2", "smape", "rmse_improvement_pct", "train_time_s"]]"""),
    code("""display(pd.read_csv(RES / "tuning_summary.csv")[["optimizer", "n_runs", "val_rmse_mean", "test_rmse_mean", "test_rmse_std", "runtime_s_mean"]])
display(pd.read_csv(RES / "wilcoxon.csv"))"""),
    code("""for f in ["convergence_mean", "convergence_sma_runs", "learning_curve_default_vs_sma", "baseline_vs_optimized_bar",
          "boxplot_test_rmse", "pred_vs_actual_sma", "residuals_sma", "feature_importance_sma",
          "hyperparams_comparison", "runtime_vs_rmse", "sma_weights_explained"]:
    p = RES / "figures" / f"{f}.png"
    if p.exists():
        display(Markdown(f"**{f}**")); display(Image(filename=str(p), width=820))"""),
    md("""## 6. Known limitations
* Palo Alto data — patterns may differ from Indian cities; the method transfers, the numbers may not.
* Per-session energy is very noisy (driver behaviour), so R² is moderate; the focus is the improvement from tuning.
* Tuning uses a training subsample for speed; tuned hyperparameters are re-fitted on the full training set.
* Only 5 seeds per optimiser; the Wilcoxon test has limited power.
* In the normalised [0,1] space SMA's wrap move contracts toward the lower bounds — a property of the original algorithm."""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent.parent / "notebooks" / "SMA_XGBoost_EV_Demand.ipynb"
nbf.write(nb, out)
print("wrote", out)
