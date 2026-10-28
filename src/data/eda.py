"""Exploratory data analysis figures written to ``results/eda/``."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.preprocess import NUMERIC_FEATURES, START, TARGET  # noqa: E402

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _save(fig, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def target_distribution(df: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df[TARGET], bins=80, color="#2563eb")
    axes[0].set(title="Energy per session", xlabel="kWh", ylabel="Sessions")
    axes[1].hist(np.log1p(df[TARGET]), bins=80, color="#16a34a")
    axes[1].set(title="log(1 + energy)", xlabel="log(1 + kWh)", ylabel="Sessions")
    _save(fig, out / "target_distribution.png", dpi)


def sessions_per_year(df: pd.DataFrame, out: Path, dpi: int) -> None:
    counts = df[START].dt.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color="#2563eb")
    ax.set_xticks(counts.index)
    ax.set(title="Charging sessions per year", xlabel="Year", ylabel="Sessions")
    _save(fig, out / "sessions_per_year.png", dpi)


def mean_energy_by(df: pd.DataFrame, col: str, labels, title: str, fname: str, out: Path, dpi: int) -> None:
    m = df.groupby(col)[TARGET].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(m.index, m.values, marker="o", color="#16a34a")
    if labels is not None:
        ax.set_xticks(range(len(labels)), labels)
    ax.set(title=title, ylabel="Mean kWh")
    ax.grid(alpha=0.3)
    _save(fig, out / fname, dpi)


def correlation_heatmap(df: pd.DataFrame, out: Path, dpi: int) -> None:
    cols = [c for c in NUMERIC_FEATURES if df[c].nunique() > 1] + [TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90)
    ax.set_yticks(range(len(cols)), cols)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation of numeric features (train)")
    _save(fig, out / "correlation_heatmap.png", dpi)


def missing_values(raw: pd.DataFrame, out: Path, dpi: int) -> None:
    frac = raw.isna().mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(frac.index, frac.values, color="#ea580c")
    ax.set_xticks(range(len(frac)), frac.index, rotation=90)
    ax.set(title="Missing values in the raw CSV", ylabel="% missing")
    _save(fig, out / "missing_values.png", dpi)


def make_all(raw: pd.DataFrame, result: dict, cfg: dict, out: Path) -> None:
    """Create every EDA figure from the raw frame and the processed splits."""
    out.mkdir(parents=True, exist_ok=True)
    dpi = cfg["figures"]["dpi"]
    full = pd.concat([result["train"], result["val"], result["test"]], ignore_index=True)
    missing_values(raw, out, dpi)
    target_distribution(full, out, dpi)
    sessions_per_year(full, out, dpi)
    mean_energy_by(full, "hour", None, "Mean energy by plug-in hour", "mean_energy_by_hour.png", out, dpi)
    mean_energy_by(full, "dayofweek", DAYS, "Mean energy by weekday", "mean_energy_by_weekday.png", out, dpi)
    correlation_heatmap(result["train"], out, dpi)
