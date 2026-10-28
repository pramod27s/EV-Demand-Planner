"""Classic benchmark functions used to verify the optimisers before the expensive ML tuning.

Each function takes a position ``u ∈ [0,1]^D`` (the optimisers' search space), decodes it
linearly into the function's standard bounds (as in the SMA paper, Li et al. 2020) and
returns the value to minimise. The global minimum of every function is 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    d = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
                 - np.exp(np.sum(np.cos(2 * np.pi * x)) / d) + 20 + np.e)


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


def griewank(x: np.ndarray) -> float:
    i = np.arange(1, len(x) + 1)
    return float(np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1)


@dataclass(frozen=True)
class Benchmark:
    name: str
    fn: Callable[[np.ndarray], float]
    lower: float
    upper: float

    def decode(self, u: np.ndarray) -> np.ndarray:
        return self.lower + np.asarray(u) * (self.upper - self.lower)

    def __call__(self, u: np.ndarray) -> float:
        return self.fn(self.decode(u))


BENCHMARKS: dict[str, Benchmark] = {
    "sphere": Benchmark("Sphere", sphere, -100.0, 100.0),
    "rastrigin": Benchmark("Rastrigin", rastrigin, -5.12, 5.12),
    "ackley": Benchmark("Ackley", ackley, -32.0, 32.0),
    "rosenbrock": Benchmark("Rosenbrock", rosenbrock, -30.0, 30.0),
    "griewank": Benchmark("Griewank", griewank, -600.0, 600.0),
}
