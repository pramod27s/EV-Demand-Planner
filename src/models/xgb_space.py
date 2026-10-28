"""XGBoost hyperparameter search space and the fitness function the optimisers minimise.

Every optimiser searches a normalised vector ``u ∈ [0,1]^9``; :class:`SearchSpace` decodes
it into hyperparameters (linear or log-uniform). Fitness = validation RMSE of XGBoost
trained with those hyperparameters on a fixed training subsample (for speed).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.metrics import rmse
from src.models.baselines import fit_xgb


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    log: bool = False      # log-uniform decoding
    integer: bool = False


class SearchSpace:
    """The 9-D XGBoost search space (spec §6.1)."""

    PARAMS = (
        Param("n_estimators", 100, 1000, integer=True),
        Param("max_depth", 3, 12, integer=True),
        Param("learning_rate", 0.01, 0.3, log=True),
        Param("subsample", 0.5, 1.0),
        Param("colsample_bytree", 0.5, 1.0),
        Param("min_child_weight", 1, 10),
        Param("gamma", 0, 5),
        Param("reg_lambda", 1e-3, 10, log=True),
        Param("reg_alpha", 1e-3, 10, log=True),
    )

    @property
    def dim(self) -> int:
        return len(self.PARAMS)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.PARAMS]

    @staticmethod
    def _round(x: float) -> float:
        """Round floats to 4 significant digits (keeps the cache key and the model consistent)."""
        return 0.0 if x == 0 else float(f"{x:.4g}")

    def decode(self, u: np.ndarray) -> dict:
        """Map ``u ∈ [0,1]^9`` to hyperparameters.

        linear:      lo + u·(hi - lo)
        log-uniform: 10 ** (log10(lo) + u·(log10(hi) - log10(lo)))
        """
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        out = {}
        for p, ui in zip(self.PARAMS, u):
            if p.log:
                v = 10 ** (math.log10(p.lo) + ui * (math.log10(p.hi) - math.log10(p.lo)))
            else:
                v = p.lo + ui * (p.hi - p.lo)
            out[p.name] = int(round(v)) if p.integer else self._round(v)
        return out

    def normalise(self, params: dict) -> np.ndarray:
        """Inverse of :meth:`decode` (used to plot hyperparameters on a common 0-1 scale)."""
        u = []
        for p in self.PARAMS:
            v = params[p.name]
            if p.log:
                u.append((math.log10(v) - math.log10(p.lo)) / (math.log10(p.hi) - math.log10(p.lo)))
            else:
                u.append((v - p.lo) / (p.hi - p.lo))
        return np.clip(np.array(u), 0, 1)


class XGBFitness:
    """Callable fitness: ``u -> validation RMSE`` with memoisation.

    * Trains on a fixed random subsample of the training set (``sample_size`` rows, drawn
      once with the run's seed) and evaluates on the full validation set.
    * Early stopping (``early_stopping_rounds``) on the validation set.
    * Results are cached by the rounded decoded hyperparameters, so identical
      configurations are never re-trained; ``real_evaluations`` counts actual trainings.
    """

    def __init__(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame,
                 y_val: np.ndarray, seed: int, sample_size: int, early_stopping_rounds: int,
                 space: SearchSpace | None = None):
        self.space = space or SearchSpace()
        rng = np.random.default_rng(seed)
        n = len(X_train)
        idx = np.sort(rng.choice(n, size=min(sample_size, n), replace=False))
        self.X_tr, self.y_tr = X_train.iloc[idx], np.asarray(y_train)[idx]
        self.X_val, self.y_val = X_val, np.asarray(y_val)
        self.seed, self.esr = seed, early_stopping_rounds
        self.cache: dict[tuple, float] = {}
        self.calls = 0
        self.real_evaluations = 0
        self.train_time_s = 0.0

    def key(self, params: dict) -> tuple:
        return tuple(params[n] for n in self.space.names)

    def __call__(self, u: np.ndarray) -> float:
        self.calls += 1
        params = self.space.decode(u)
        k = self.key(params)
        if k not in self.cache:
            t0 = time.perf_counter()
            model = fit_xgb(params, self.X_tr, self.y_tr, self.X_val, self.y_val,
                            self.seed, self.esr)
            self.cache[k] = rmse(self.y_val, model.predict(self.X_val))
            self.train_time_s += time.perf_counter() - t0
            self.real_evaluations += 1
        return self.cache[k]

    @property
    def cache_hits(self) -> int:
        return self.calls - self.real_evaluations
