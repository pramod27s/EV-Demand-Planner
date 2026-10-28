"""Statistics over tuning runs: per-optimiser summaries and paired Wilcoxon tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["rmse", "mae", "r2", "smape"]


def runs_frame(runs: list[dict]) -> pd.DataFrame:
    """One row per (optimiser, seed) with validation and test metrics."""
    return pd.DataFrame([{
        "optimizer": r["optimizer"], "label": r["label"], "seed": r["seed"],
        "val_rmse": r["best_val_rmse"], "final_val_rmse": r["final_val_rmse"],
        **{f"test_{m}": r["test_metrics"][m] for m in METRICS},
        "runtime_s": r["runtime_s"], "evaluations": r["evaluations"],
        "real_evaluations": r["real_evaluations"],
    } for r in runs])


def tuning_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, best and worst of validation RMSE and test metrics per optimiser."""
    rows = []
    for (key, label), g in df.groupby(["optimizer", "label"], sort=False):
        row = {"optimizer": label, "n_runs": len(g)}
        for col in ["val_rmse"] + [f"test_{m}" for m in METRICS]:
            v = g[col].to_numpy()
            higher_better = col == "test_r2"
            row[f"{col}_mean"] = v.mean()
            row[f"{col}_std"] = v.std(ddof=1) if len(v) > 1 else 0.0
            row[f"{col}_best"] = v.max() if higher_better else v.min()
            row[f"{col}_worst"] = v.min() if higher_better else v.max()
        row["runtime_s_mean"] = g["runtime_s"].mean()
        row["evaluations_mean"] = g["evaluations"].mean()
        row["real_evaluations_mean"] = g["real_evaluations"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def wilcoxon_vs(df: pd.DataFrame, reference: str = "sma", others=("pso", "ga", "random"),
                metric: str = "test_rmse") -> pd.DataFrame:
    """Paired (by seed) two-sided Wilcoxon signed-rank test of ``reference`` vs each other optimiser.

    With only 5 seeds the smallest attainable two-sided p-value is 0.0625, so the test has
    low power; p-values are reported as they are.
    """
    ref = df[df.optimizer == reference].set_index("seed")[metric]
    rows = []
    for o in others:
        other = df[df.optimizer == o].set_index("seed")[metric]
        seeds = ref.index.intersection(other.index)
        a, b = ref.loc[seeds].to_numpy(), other.loc[seeds].to_numpy()
        row = {"comparison": f"{reference.upper()} vs {o.upper() if o != 'random' else 'Random'}",
               "metric": metric, "n_pairs": len(seeds),
               f"{reference}_mean": a.mean() if len(a) else np.nan,
               "other_mean": b.mean() if len(b) else np.nan,
               "sma_wins": int((a < b).sum()), "statistic": np.nan, "p_value": np.nan}
        if len(seeds) >= 2 and np.any(a != b):
            res = wilcoxon(a, b, alternative="two-sided")
            row["statistic"], row["p_value"] = float(res.statistic), float(res.pvalue)
            row["note"] = "low power (few seeds)" if len(seeds) < 10 else ""
        else:
            row["note"] = "not enough paired runs for a test"
        rows.append(row)
    return pd.DataFrame(rows)
