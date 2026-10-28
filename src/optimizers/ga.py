"""Real-coded Genetic Algorithm (GA) — Holland (1975); SBX: Deb & Agrawal (1995).

Biology
-------
Natural selection: fitter individuals are more likely to reproduce; offspring inherit a
mix of their parents' genes (crossover) plus occasional random changes (mutation). Over
generations the population adapts to its environment.

Operators (one generation = one iteration)
------------------------------------------
* **Tournament selection** (size k = 3): pick k random individuals; the fittest becomes a
  parent.
* **Simulated Binary Crossover** (SBX, p_cx = 0.9, eta_c = 15), per gene, bounded to
  [0, 1]: children ``c1,2 = 0.5·[(y1 + y2) ∓ beta_q·(y2 - y1)]``, where beta_q is drawn so
  that the spread of children around the parents mimics single-point crossover on binary
  strings. Larger eta_c -> children closer to the parents.
* **Polynomial mutation** (p_m = 1/D per gene, eta_m = 20), bounded to [0, 1]: a gene is
  perturbed by ``delta_q`` drawn from a polynomial distribution concentrated near 0.
* **Elitism**: the best 2 individuals survive unchanged into the next generation (their
  fitness is carried over, not re-evaluated).
"""
from __future__ import annotations

import numpy as np

from src.optimizers.base import FitnessFn, OptimizationResult, Optimizer


class GA(Optimizer):
    name = "GA"

    def __init__(self, pop_size: int, max_iter: int, seed: int, tournament: int = 3,
                 p_cx: float = 0.9, eta_c: float = 15, eta_m: float = 20, elites: int = 2):
        super().__init__(pop_size, max_iter, seed)
        self.k, self.p_cx, self.eta_c, self.eta_m = tournament, p_cx, eta_c, eta_m
        self.elites = min(elites, pop_size - 1)

    # ----- operators -----------------------------------------------------------------
    def tournament_select(self, f: np.ndarray) -> int:
        """Index of the fittest of ``k`` randomly drawn individuals."""
        idx = self.rng.integers(0, len(f), self.k)
        return int(idx[np.argmin(f[idx])])

    def sbx(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bounded simulated binary crossover on [0, 1] (Deb & Agrawal 1995)."""
        c1, c2 = p1.copy(), p2.copy()
        if self.rng.random() > self.p_cx:
            return c1, c2
        eta = self.eta_c
        for d in range(len(p1)):
            if self.rng.random() > 0.5 or abs(p1[d] - p2[d]) < 1e-14:
                continue                                   # gene inherited unchanged
            y1, y2 = min(p1[d], p2[d]), max(p1[d], p2[d])
            u = self.rng.random()
            spreads = []
            # beta for the lower-bound side and the upper-bound side of the gene
            for beta in (1 + 2 * y1 / (y2 - y1), 1 + 2 * (1 - y2) / (y2 - y1)):
                alpha = 2 - beta ** -(eta + 1)
                if u <= 1 / alpha:
                    bq = (u * alpha) ** (1 / (eta + 1))
                else:
                    bq = (1 / (2 - u * alpha)) ** (1 / (eta + 1))
                spreads.append(bq)
            ch1 = np.clip(0.5 * ((y1 + y2) - spreads[0] * (y2 - y1)), 0, 1)
            ch2 = np.clip(0.5 * ((y1 + y2) + spreads[1] * (y2 - y1)), 0, 1)
            if self.rng.random() < 0.5:
                ch1, ch2 = ch2, ch1
            c1[d], c2[d] = ch1, ch2
        return c1, c2

    def polynomial_mutation(self, x: np.ndarray) -> np.ndarray:
        """Bounded polynomial mutation on [0, 1], each gene with probability 1/D."""
        x = x.copy()
        D, eta = len(x), self.eta_m
        mut_pow = 1 / (eta + 1)
        for d in range(D):
            if self.rng.random() >= 1 / D:
                continue
            u = self.rng.random()
            d1, d2 = x[d], 1 - x[d]                        # distance to lower / upper bound
            if u < 0.5:
                val = 2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)
                dq = val ** mut_pow - 1
            else:
                val = 2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta + 1)
                dq = 1 - val ** mut_pow
            x[d] = np.clip(x[d] + dq, 0, 1)
        return x

    # ----- main loop -------------------------------------------------------------------
    def optimize(self, fitness_fn: FitnessFn, dim: int) -> OptimizationResult:
        self._start()
        N, T = self.pop_size, self.max_iter
        pop = self.rng.uniform(0, 1, (N, dim))
        f = self._evaluate(fitness_fn, pop)
        self._record(1, f)

        for t in range(2, T + 1):
            elite_idx = np.argsort(f, kind="stable")[: self.elites]
            children = []
            while len(children) < N - self.elites:
                p1 = pop[self.tournament_select(f)]
                p2 = pop[self.tournament_select(f)]
                for c in self.sbx(p1, p2):
                    children.append(self.polynomial_mutation(c))
            children = np.array(children[: N - self.elites])
            child_f = self._evaluate(fitness_fn, children)
            pop = np.vstack([pop[elite_idx], children])
            f = np.concatenate([f[elite_idx], child_f])
            self._record(t, f)

        return self._result()
