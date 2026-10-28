"""Particle Swarm Optimisation (PSO) — Kennedy & Eberhart (1995), global-best variant.

Biology
-------
Inspired by bird flocking and fish schooling: each individual adjusts its movement using
its own best experience (cognition) and the best position found by the whole flock
(social information). No leader is needed; good regions emerge from local rules.

Update equations (per particle, per iteration)
----------------------------------------------
    v <- w·v + c1·r1·(pbest - x) + c2·r2·(gbest - x)      r1, r2 ~ U(0,1)^D
    v <- clip(v, -v_max, v_max)
    x <- clip(x + v, 0, 1)

* ``w`` (inertia) decreases linearly from 0.9 to 0.4: large early (explore), small late
  (exploit).
* ``c1 = c2 = 2.0``: cognitive and social acceleration.
* ``v_max = 0.2`` limits step size (20 % of each normalised range); velocities start in
  ``U(-0.1, 0.1)``.
"""
from __future__ import annotations

import numpy as np

from src.optimizers.base import FitnessFn, OptimizationResult, Optimizer


class PSO(Optimizer):
    name = "PSO"

    def __init__(self, pop_size: int, max_iter: int, seed: int, w_start: float = 0.9,
                 w_end: float = 0.4, c1: float = 2.0, c2: float = 2.0, v_max: float = 0.2,
                 v_init: float = 0.1):
        super().__init__(pop_size, max_iter, seed)
        self.w_start, self.w_end = w_start, w_end
        self.c1, self.c2, self.v_max, self.v_init = c1, c2, v_max, v_init

    def optimize(self, fitness_fn: FitnessFn, dim: int) -> OptimizationResult:
        self._start()
        N, T, rng = self.pop_size, self.max_iter, self.rng
        x = rng.uniform(0, 1, (N, dim))
        v = rng.uniform(-self.v_init, self.v_init, (N, dim))
        pbest, pbest_f = x.copy(), np.full(N, np.inf)

        for t in range(1, T + 1):
            f = self._evaluate(fitness_fn, x)
            improved = f < pbest_f                      # update personal bests
            pbest[improved], pbest_f[improved] = x[improved], f[improved]
            gbest = self._best_x                        # global best so far
            self._record(t, f)
            if t == T:
                break

            w = self.w_start - (self.w_start - self.w_end) * (t - 1) / max(T - 1, 1)
            r1, r2 = rng.random((N, dim)), rng.random((N, dim))
            v = w * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (gbest - x)
            v = np.clip(v, -self.v_max, self.v_max)
            x = np.clip(x + v, 0.0, 1.0)

        return self._result()
