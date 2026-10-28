"""All result figures (matplotlib only, PNG at ``config.yaml -> figures.dpi``).

Colours follow the spec (SMA green, PSO orange, GA purple, Random grey, Default blue).
Green and orange are close for red-green colour-blind readers, so every line chart also
gives each optimiser its own line style and marker, and bars carry value labels: colour is
never the only way to tell series apart.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ORDER = ["random", "ga", "pso", "sma"]
LABEL = {"sma": "SMA", "pso": "PSO", "ga": "GA", "random": "Random", "default": "Default"}
STYLE = {   # secondary encoding (line style + marker) next to colour
    "sma": {"linestyle": "-", "marker": "o"},
    "pso": {"linestyle": "--", "marker": "s"},
    "ga": {"linestyle": "-.", "marker": "^"},
    "random": {"linestyle": ":", "marker": "v"},
    "default": {"linestyle": "-", "marker": "D"},
}
INK, MUTED = "#1f2937", "#6b7280"

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9ca3af", "axes.labelcolor": INK, "axes.titleweight": "bold",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#e5e7eb", "grid.linewidth": 0.8, "lines.linewidth": 2, "axes.axisbelow": True,
    "legend.frameon": False, "font.size": 10,
})


def _colors(cfg: dict) -> dict:
    return cfg["figures"]["colors"]


def _save(fig, path: Path, cfg: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=cfg["figures"]["dpi"])
    plt.close(fig)
    return path


def _int_x(ax) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _markevery(n: int) -> int:
    return max(1, n // 10)


def _label_bars(ax, bars, fmt="{:.3f}") -> None:
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, color=INK, xytext=(0, 2),
                    textcoords="offset points")


# 1 ---------------------------------------------------------------------------------
def convergence_mean(curves: dict[str, list[list[float]]], cfg: dict, path: Path) -> Path:
    """Mean ± std of best-so-far validation RMSE per iteration, all optimisers."""
    c = _colors(cfg)
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in [k for k in ORDER if k in curves]:
        arr = np.array(curves[key])
        it = np.arange(1, arr.shape[1] + 1)
        mean, std = arr.mean(0), arr.std(0)
        ax.plot(it, mean, color=c[key], label=f"{LABEL[key]} (n={len(arr)})",
                markevery=_markevery(len(it)), markersize=6, **STYLE[key])
        ax.fill_between(it, mean - std, mean + std, color=c[key], alpha=0.12, linewidth=0)
    ax.set(title="Convergence: best validation RMSE so far (mean ± std over seeds)",
           xlabel="Iteration", ylabel="Validation RMSE (kWh)")
    _int_x(ax)
    ax.legend()
    return _save(fig, path, cfg)


# 2 ---------------------------------------------------------------------------------
def convergence_runs(curves: list[list[float]], seeds: list[int], key: str, cfg: dict, path: Path) -> Path:
    """Every seed's convergence curve for one optimiser."""
    color = _colors(cfg)[key]
    fig, ax = plt.subplots(figsize=(8, 5))
    shades = np.linspace(0.45, 1.0, len(curves))
    for curve, seed, a in zip(curves, seeds, shades):
        it = np.arange(1, len(curve) + 1)
        ax.plot(it, curve, color=color, alpha=a, label=f"seed {seed}")
        ax.annotate(f"seed {seed}", (it[-1], curve[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=MUTED, va="center")
    ax.set(title=f"{LABEL[key]} convergence for every seed", xlabel="Iteration",
           ylabel="Best validation RMSE (kWh)")
    _int_x(ax)
    ax.legend(ncol=2, fontsize=8)
    return _save(fig, path, cfg)


# 3 ---------------------------------------------------------------------------------
def learning_curves(default: dict, tuned: dict, cfg: dict, path: Path, tuned_key: str = "sma") -> Path:
    """Train / validation RMSE per boosting round: default XGBoost vs tuned XGBoost."""
    c = _colors(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (name, curves, color) in zip(axes, [("Default XGBoost", default, c["default"]),
                                               (f"XGBoost + {LABEL[tuned_key]}", tuned, c[tuned_key])]):
        r = np.arange(1, len(curves["train_rmse"]) + 1)
        ax.plot(r, curves["train_rmse"], color=color, linestyle="--", label="train")
        ax.plot(r, curves["val_rmse"], color=color, label="validation")
        best = int(np.argmin(curves["val_rmse"]))
        ax.scatter([best + 1], [curves["val_rmse"][best]], color=color, s=50, zorder=3,
                   edgecolor="white", linewidth=2)
        ax.annotate(f"best val {curves['val_rmse'][best]:.3f}\n@ round {best + 1}",
                    (best + 1, curves["val_rmse"][best]), xytext=(6, 10), textcoords="offset points",
                    fontsize=8, color=INK)
        ax.set(title=name, xlabel="Boosting round")
        ax.legend()
    axes[0].set_ylabel("RMSE (kWh)")
    fig.suptitle("Learning curves", fontweight="bold")
    return _save(fig, path, cfg)


# 4 ---------------------------------------------------------------------------------
MODEL_COLOR_KEY = {"Linear Regression": "lr", "Random Forest": "rf", "XGBoost (default)": "default",
                   "XGBoost + Random": "random", "XGBoost + GA": "ga", "XGBoost + PSO": "pso",
                   "XGBoost + SMA": "sma"}


def comparison_bars(comp: pd.DataFrame, cfg: dict, path: Path) -> Path:
    """Test RMSE / MAE / R² per model (one panel per metric; never a dual axis)."""
    c = _colors(cfg)
    models = [m for m in MODEL_COLOR_KEY if m in set(comp["model"]) and m != "XGBoost + Random"]
    comp = comp.set_index("model").loc[models]
    colors = [c[MODEL_COLOR_KEY[m]] for m in models]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (col, title) in zip(axes, [("rmse", "RMSE (kWh, lower is better)"),
                                       ("mae", "MAE (kWh, lower is better)"),
                                       ("r2", "R² (higher is better)")]):
        vals = comp[col].to_numpy()
        bars = ax.bar(range(len(models)), vals, color=colors, edgecolor="white", linewidth=2)
        _label_bars(ax, bars)
        lo, hi = vals.min(), vals.max()
        pad = (hi - lo) * 0.6 + 1e-6
        ax.set_ylim(max(0, lo - pad), hi + pad * 0.6)
        ax.set_xticks(range(len(models)), [m.replace("XGBoost", "XGB") for m in models],
                      rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Test-set performance: baselines vs optimised XGBoost (y-axis zoomed)", fontweight="bold")
    return _save(fig, path, cfg)


# 5 ---------------------------------------------------------------------------------
def boxplot_test_rmse(runs: pd.DataFrame, cfg: dict, path: Path, default_rmse: float | None = None) -> Path:
    c = _colors(cfg)
    keys = [k for k in ORDER if k in set(runs.optimizer)]
    data = [runs[runs.optimizer == k]["test_rmse"].to_numpy() for k in keys]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops={"color": INK})
    for patch, k in zip(bp["boxes"], keys):
        patch.set(facecolor=c[k], alpha=0.35, edgecolor=c[k])
    rng = np.random.default_rng(0)
    for i, (d, k) in enumerate(zip(data, keys), start=1):
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(d)), d, color=c[k], s=40, zorder=3,
                   edgecolor="white", linewidth=1.5, marker=STYLE[k]["marker"])
    if default_rmse is not None:
        ax.axhline(default_rmse, color=c["default"], linestyle="--", linewidth=1.5)
        ax.annotate("default XGBoost", (len(keys) + 0.45, default_rmse), color=INK, fontsize=8,
                    va="bottom", ha="right")
    ax.set_xticks(range(1, len(keys) + 1), [LABEL[k] for k in keys])
    ax.set(title="Test RMSE across seeds", ylabel="Test RMSE (kWh)")
    return _save(fig, path, cfg)


# 6 ---------------------------------------------------------------------------------
def pred_vs_actual(y: np.ndarray, yhat: np.ndarray, cfg: dict, path: Path, key: str = "sma") -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, yhat, s=8, alpha=0.25, color=_colors(cfg)[key], linewidth=0)
    lim = [0, max(y.max(), yhat.max()) * 1.02]
    ax.plot(lim, lim, color=INK, linewidth=1.2, linestyle="--", label="y = x")
    ax.set(xlim=lim, ylim=lim, xlabel="Actual energy (kWh)", ylabel="Predicted energy (kWh)",
           title=f"Predicted vs actual — XGBoost + {LABEL[key]} (test, n={len(y):,})")
    ax.legend()
    return _save(fig, path, cfg)


# 7 ---------------------------------------------------------------------------------
def residuals(y: np.ndarray, yhat: np.ndarray, cfg: dict, path: Path, key: str = "sma") -> Path:
    color = _colors(cfg)[key]
    res = y - yhat
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(res, bins=80, color=color, edgecolor="white", linewidth=0.5)
    axes[0].axvline(0, color=INK, linewidth=1)
    axes[0].set(title="Residual distribution", xlabel="Actual − predicted (kWh)", ylabel="Sessions")
    axes[1].scatter(yhat, res, s=6, alpha=0.2, color=color, linewidth=0)
    axes[1].axhline(0, color=INK, linewidth=1)
    axes[1].set(title="Residuals vs predicted", xlabel="Predicted (kWh)", ylabel="Residual (kWh)")
    fig.suptitle(f"Residuals — XGBoost + {LABEL[key]} (test)", fontweight="bold")
    return _save(fig, path, cfg)


# 8 ---------------------------------------------------------------------------------
def feature_importance(imp: dict[str, float], cfg: dict, path: Path, key: str = "sma", top: int = 15) -> Path:
    items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:top][::-1]
    names, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, vals, color=_colors(cfg)[key], edgecolor="white", linewidth=2)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.0f}", (v, i), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK)
    ax.set(title=f"Top-{top} feature importances (gain) — XGBoost + {LABEL[key]}", xlabel="Gain")
    ax.grid(axis="y", visible=False)
    return _save(fig, path, cfg)


# 9 ---------------------------------------------------------------------------------
def hyperparams_comparison(norm: dict[str, np.ndarray], names: list[str], cfg: dict, path: Path) -> Path:
    """Best hyperparameters per optimiser on a common 0-1 scale (0 = lower bound, 1 = upper)."""
    c = _colors(cfg)
    keys = [k for k in ORDER if k in norm]
    width = 0.8 / len(keys)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(13, 5))
    hatches = {"sma": "", "pso": "//", "ga": "\\\\", "random": ".."}
    for i, k in enumerate(keys):
        ax.bar(x + (i - (len(keys) - 1) / 2) * width, norm[k], width, color=c[k], label=LABEL[k],
               edgecolor="white", linewidth=1.5, hatch=hatches[k])
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set(ylim=(0, 1.05), ylabel="Normalised value (0 = low bound, 1 = high bound)",
           title="Best hyperparameters found by each optimiser (best seed)")
    ax.legend(ncol=len(keys))
    ax.grid(axis="x", visible=False)
    return _save(fig, path, cfg)


# 10 --------------------------------------------------------------------------------
def runtime_vs_rmse(summary: pd.DataFrame, cfg: dict, path: Path) -> Path:
    c = _colors(cfg)
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in summary.iterrows():
        k = r["key"]
        ax.errorbar(r["runtime_s_mean"] / 60, r["test_rmse_mean"], yerr=r["test_rmse_std"],
                    fmt=STYLE[k]["marker"], color=c[k], markersize=10, capsize=4,
                    markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(LABEL[k], (r["runtime_s_mean"] / 60, r["test_rmse_mean"]), xytext=(8, 6),
                    textcoords="offset points", color=INK)
    ax.set(title="Tuning cost vs test RMSE (mean ± std over seeds)",
           xlabel="Mean tuning runtime (minutes)", ylabel="Mean test RMSE (kWh)")
    return _save(fig, path, cfg)


# 11 --------------------------------------------------------------------------------
def sma_weights(snapshot: dict, cfg: dict, path: Path) -> Path:
    """SMA vein-thickness weights W against fitness rank for one iteration."""
    S = np.array(snapshot["fitness"])
    W = np.array(snapshot["weights"])
    rank = np.array(snapshot["rank"])                     # 0 = best
    N = len(S)
    color = _colors(cfg)["sma"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for d in range(W.shape[1]):
        axes[0].scatter(rank, W[:, d], s=18, color=color, alpha=0.45, linewidth=0)
    axes[0].plot(np.sort(rank), W[np.argsort(rank)].mean(1), color=INK, marker="o", label="mean over dimensions")
    axes[0].axhline(1, color=MUTED, linestyle="--", linewidth=1)
    axes[0].axvline(N / 2 - 0.5, color=MUTED, linestyle=":", linewidth=1)
    axes[0].annotate("thicker veins (W > 1)\nbetter half", (0, W.max()), fontsize=8, color=INK, va="top")
    axes[0].annotate("thinner veins (W < 1)\nworse half", (N - 1, W.min()), fontsize=8, color=INK,
                     ha="right", va="bottom")
    axes[0].set(title=f"SMA weights W vs fitness rank (iteration {snapshot['iteration']})",
                xlabel="Fitness rank (0 = best)", ylabel="Weight W")
    axes[0].legend(loc="upper right")
    order = np.argsort(rank)
    axes[1].bar(np.arange(N), S[order], color=color, edgecolor="white", linewidth=2)
    axes[1].set(title="Fitness of the same slime-mould cells", xlabel="Fitness rank (0 = best)",
                ylabel="Validation RMSE (kWh)", ylim=(S.min() - (S.max() - S.min()) * 0.3, S.max() * 1.01))
    axes[1].grid(axis="x", visible=False)
    fig.suptitle("Slime mould 'vein thickness': W = 1 ± r·log10((bF − S)/(bF − wF) + 1)", fontweight="bold")
    return _save(fig, path, cfg)


# 12 --------------------------------------------------------------------------------
def plot_benchmark_convergence(fname: str, per_opt: dict[str, list[list[float]]], cfg: dict, path: Path) -> Path:
    c = _colors(cfg)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key in [k for k in ORDER if k in per_opt]:
        arr = np.maximum(np.array(per_opt[key]), 1e-300)
        it = np.arange(1, arr.shape[1] + 1)
        med = np.median(arr, 0)
        ax.semilogy(it, med, color=c[key], label=LABEL[key], markevery=_markevery(len(it)),
                    markersize=6, **STYLE[key])
        ax.fill_between(it, np.percentile(arr, 25, 0), np.percentile(arr, 75, 0), color=c[key],
                        alpha=0.12, linewidth=0)
    ax.set(title=f"{fname} (D = {cfg['benchmarks']['dim']}): median best fitness, IQR band",
           xlabel="Iteration", ylabel="Best fitness (log scale)")
    ax.legend()
    return _save(fig, path, cfg)
