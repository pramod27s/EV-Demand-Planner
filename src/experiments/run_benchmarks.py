"""Benchmark-function experiment: verifies SMA, PSO, GA and Random Search are correct.

D = 10, pop 30, 200 iterations, 10 seeds (see ``config.yaml -> benchmarks``).
Outputs ``results/benchmarks/benchmark_stats.csv`` (mean, std, best, worst of the final
best fitness), ``benchmark_curves.json`` and one convergence plot per function.

Run with ``python -m src.experiments.run_benchmarks``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load_config, results_dir
from src.experiments.common import write_json
from src.optimizers import LABELS, make_optimizer
from src.optimizers.benchmarks import BENCHMARKS

OPTIMIZERS = ["sma", "pso", "ga", "random"]


def run(cfg: dict) -> pd.DataFrame:
    from src.viz.plots import plot_benchmark_convergence

    b = cfg["benchmarks"]
    out = results_dir(cfg) / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    jobs = [(f, o) for f in b["functions"] for o in OPTIMIZERS]
    for fname, key in tqdm(jobs, desc="benchmarks"):
        finals, runs = [], []
        for seed in b["seeds"]:
            res = make_optimizer(key, b["pop_size"], b["max_iter"], seed, cfg).optimize(
                BENCHMARKS[fname], b["dim"])
            finals.append(res.best_fitness)
            runs.append(res.convergence)
        finals = np.array(finals)
        rows.append({"function": BENCHMARKS[fname].name, "optimizer": LABELS[key],
                     "mean": finals.mean(), "std": finals.std(ddof=1) if len(finals) > 1 else 0.0,
                     "best": finals.min(), "worst": finals.max()})
        curves.setdefault(fname, {})[key] = runs

    df = pd.DataFrame(rows)
    df.to_csv(out / "benchmark_stats.csv", index=False)
    write_json(out / "benchmark_curves.json", curves)
    for fname, per_opt in curves.items():
        plot_benchmark_convergence(BENCHMARKS[fname].name, per_opt, cfg, out / f"{fname}.png")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return df


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()
