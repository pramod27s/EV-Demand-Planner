"""Slime Mould Algorithm (SMA) — Li, Chen, Wang, Heidari & Mirjalili (2020).

Biology
-------
The slime mould *Physarum polycephalum* forages by spreading a network of protoplasmic
tubes ("veins"). Cytoplasm oscillates through the network; tubes that lead to rich food
receive more flow and thicken (positive feedback), while tubes leading to poor food thin
out and disappear. The organism therefore concentrates on the best food sources while
still sending exploratory tendrils elsewhere. Tero et al. (*Science*, 2010) showed that a
slime mould placed on oat flakes arranged like the cities around Tokyo grew a network
strikingly similar to the Tokyo rail system.

SMA models this with three behaviours:

* **Weights (vein thickness).** Cells are ranked by fitness. For the better half
  ``W = 1 + r·q`` and for the worse half ``W = 1 - r·q`` with
  ``q = log10((bF - S_i) / (bF - wF) + 1) ∈ [0, log10 2]``,
  where ``bF``/``wF`` are this iteration's best/worst fitness and ``r ~ U(0,1)``.
  Good cells get thick veins (W > 1), poor cells thin veins (W < 1).
* **Approach food** (with probability ``p = tanh|S_i - DF|``, DF = best fitness so far):
  ``X_i,d = Xb_d + vb_d · (W_i,d · X_A,d - X_B,d)``, ``vb ~ U(-a, a)``,
  ``a = arctanh(1 - t/T)``. Cells move around the best food found (Xb), guided by the
  weighted difference of two random cells A, B. ``a`` shrinks to 0, narrowing the search.
* **Wrap / contract** (otherwise): ``X_i,d = vc_d · X_i,d``, ``vc ~ U(-b, b)``,
  ``b = 1 - t/T``. Models the oscillating contraction of the vein network. (Note: in
  the normalised space this contraction pulls towards 0, i.e. the lower bounds; this is a
  property of the original algorithm, kept here for faithfulness.)
* **Random relocation** (probability ``z = 0.03``): the cell is re-seeded uniformly
  anywhere in the space — exploration that protects against premature convergence.

Scale note: ``p = tanh(|S_i - DF|)`` uses raw fitness differences. For XGBoost tuning the
fitness is validation RMSE in kWh, where differences of 0.01-2 kWh give a sensible range
of p (0.01 -> 0.96), so no rescaling is applied.
"""
from __future__ import annotations

import numpy as np

from src.optimizers.base import FitnessFn, OptimizationResult, Optimizer

EPS = 1e-12


class SMA(Optimizer):
    name = "SMA"

    def __init__(self, pop_size: int, max_iter: int, seed: int, z: float = 0.03):
        super().__init__(pop_size, max_iter, seed)
        self.z = z

    def compute_weights(self, S: np.ndarray, order: np.ndarray, dim: int) -> np.ndarray:
        """Vein-thickness weights W (N x D) from the fitness ranking ``order`` (ascending)."""
        N = len(S)
        bF, wF = S[order[0]], S[order[-1]]
        rng_ = bF - wF - EPS                      # <= 0, as in the reference code
        W = np.empty((N, dim))
        for rank, i in enumerate(order):
            q = np.log10((bF - S[i]) / rng_ + 1)  # 0 for the best cell, log10(2) for the worst
            r = self.rng.random(dim)
            W[i] = 1 + r * q if rank < N / 2 else 1 - r * q
        return W

    def optimize(self, fitness_fn: FitnessFn, dim: int) -> OptimizationResult:
        self._start()
        N, T, rng = self.pop_size, self.max_iter, self.rng
        X = rng.uniform(0, 1, (N, dim))
        snapshot_at = max(1, T // 2)              # iteration whose weights are kept for the viva figure
        extras: dict = {}

        for t in range(1, T + 1):
            S = self._evaluate(fitness_fn, X)     # clips X, updates DF / Xb
            DF, Xb = self._best_f, self._best_x
            order = np.argsort(S, kind="stable")
            W = self.compute_weights(S, order, dim)
            if t == snapshot_at:
                extras["weights_snapshot"] = {"iteration": t, "fitness": S.tolist(),
                                              "rank": np.argsort(order).tolist(), "weights": W.tolist()}
            self._record(t, S)

            a = np.arctanh(1 - t / T)             # t >= 1, so the argument is < 1; a -> 0 as t -> T
            b = 1 - t / T

            for i in range(N):
                if rng.random() < self.z:
                    X[i] = rng.uniform(0, 1, dim)                 # random relocation (exploration)
                    continue
                p = np.tanh(abs(S[i] - DF))
                vb = rng.uniform(-a, a, dim)
                vc = rng.uniform(-b, b, dim)
                for d in range(dim):
                    A, B = rng.integers(0, N, 2)
                    if rng.random() < p:
                        X[i, d] = Xb[d] + vb[d] * (W[i, d] * X[A, d] - X[B, d])   # approach food
                    else:
                        X[i, d] = vc[d] * X[i, d]                                  # wrap / contract
            np.clip(X, 0.0, 1.0, out=X)

        return self._result(extras)
