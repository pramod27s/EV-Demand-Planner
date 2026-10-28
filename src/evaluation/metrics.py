"""Regression metrics used throughout the project (RMSE is the primary metric)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root mean squared error: sqrt(mean((y - yhat)^2))."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute error: mean(|y - yhat|)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat)))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """Coefficient of determination (sklearn ``r2_score``)."""
    return float(r2_score(y, yhat))


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Symmetric MAPE in %: 100 * mean(2|y - yhat| / (|y| + |yhat| + 1e-8)).

    Used instead of MAPE, which explodes for sessions with very small kWh.
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(100 * np.mean(2 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + 1e-8)))


def all_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    return {"rmse": rmse(y, yhat), "mae": mae(y, yhat), "r2": r2(y, yhat), "smape": smape(y, yhat)}
