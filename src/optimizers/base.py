"""Common interface for every optimiser (SMA, PSO, GA, Random Search).

All optimisers *minimise* a fitness function over the unit hypercube ``[0, 1]^dim``;
the problem (XGBoost search space or a benchmark function) decodes positions itself.
Each optimiser owns a ``np.random.default_rng(seed)`` and never touches NumPy's global
RNG, so a run is fully determined by its seed.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

FitnessFn = Callable[[np.ndarray], float]


@dataclass
class OptimizationResult:
    best_position: np.ndarray      # in [0,1]^dim
    best_fitness: float
    convergence: list[float]       # best-so-far fitness after each iteration
    evaluations: int
    runtime_s: float
    history: list[dict]            # per-iteration: iteration, best_fitness, mean_fitness, best_position
    extras: dict = field(default_factory=dict)   # optimiser-specific diagnostics (e.g. SMA weights)


class Optimizer(ABC):
    """Base class: holds the budget, the RNG and the bookkeeping shared by all optimisers."""

    name: str = "base"

    def __init__(self, pop_size: int, max_iter: int, seed: int):
        if pop_size < 2 or max_iter < 1:
            raise ValueError("pop_size must be >= 2 and max_iter >= 1")
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def optimize(self, fitness_fn: FitnessFn, dim: int) -> OptimizationResult:
        ...

    # ----- helpers used by the subclasses -------------------------------------------
    def _start(self) -> None:
        """Reset per-run state (a fresh RNG makes repeated ``optimize`` calls identical)."""
        self.rng = np.random.default_rng(self.seed)
        self._evals = 0
        self._t0 = time.perf_counter()
        self._history: list[dict] = []
        self._best_x: np.ndarray | None = None
        self._best_f = np.inf

    def _evaluate(self, fitness_fn: FitnessFn, X: np.ndarray) -> np.ndarray:
        """Clip to [0, 1], evaluate every row and update the best-so-far solution."""
        np.clip(X, 0.0, 1.0, out=X)
        f = np.array([float(fitness_fn(x)) for x in X])
        self._evals += len(X)
        i = int(np.argmin(f))
        if f[i] < self._best_f:
            self._best_f, self._best_x = float(f[i]), X[i].copy()
        return f

    def _record(self, iteration: int, fitness: np.ndarray) -> None:
        self._history.append({
            "iteration": iteration,
            "best_fitness": self._best_f,
            "mean_fitness": float(np.mean(fitness)),
            "best_position": self._best_x.tolist(),
        })

    def _result(self, extras: dict | None = None) -> OptimizationResult:
        return OptimizationResult(
            best_position=self._best_x.copy(),
            best_fitness=self._best_f,
            convergence=[h["best_fitness"] for h in self._history],
            evaluations=self._evals,
            runtime_s=time.perf_counter() - self._t0,
            history=self._history,
            extras=extras or {},
        )
