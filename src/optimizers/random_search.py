"""Random Search — Bergstra & Bengio (2012). Sanity-check baseline (no biology).

Draws ``pop_size x max_iter`` points uniformly in [0,1]^D. Each block of ``pop_size``
samples counts as one "iteration", so its convergence curve (best-so-far after each
block) is directly comparable with the population-based optimisers.
"""
from __future__ import annotations

from src.optimizers.base import FitnessFn, OptimizationResult, Optimizer


class RandomSearch(Optimizer):
    name = "Random"

    def optimize(self, fitness_fn: FitnessFn, dim: int) -> OptimizationResult:
        self._start()
        for t in range(1, self.max_iter + 1):
            X = self.rng.uniform(0, 1, (self.pop_size, dim))
            f = self._evaluate(fitness_fn, X)
            self._record(t, f)
        return self._result()
