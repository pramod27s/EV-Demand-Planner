"""From-scratch NumPy implementations of SMA, PSO, GA and Random Search."""
from src.optimizers.base import OptimizationResult, Optimizer
from src.optimizers.ga import GA
from src.optimizers.pso import PSO
from src.optimizers.random_search import RandomSearch
from src.optimizers.sma import SMA

REGISTRY = {"sma": SMA, "pso": PSO, "ga": GA, "random": RandomSearch}
LABELS = {"sma": "SMA", "pso": "PSO", "ga": "GA", "random": "Random"}


def make_optimizer(key: str, pop_size: int, max_iter: int, seed: int, cfg: dict | None = None) -> Optimizer:
    """Build an optimiser from its key, taking operator parameters from ``config.yaml``."""
    params = (cfg or {}).get("optimizers", {}).get(key, {})
    return REGISTRY[key](pop_size=pop_size, max_iter=max_iter, seed=seed, **params)


__all__ = ["GA", "PSO", "SMA", "RandomSearch", "Optimizer", "OptimizationResult",
           "REGISTRY", "LABELS", "make_optimizer"]
