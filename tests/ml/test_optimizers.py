"""Optimiser correctness: convergence on Sphere, reproducibility, bounds."""
import numpy as np
import pytest

from src.optimizers import GA, PSO, SMA, RandomSearch, make_optimizer
from src.optimizers.benchmarks import BENCHMARKS

SPHERE = BENCHMARKS["sphere"]


@pytest.mark.parametrize("cls", [SMA, PSO, GA])
def test_reaches_sphere_optimum(cls):
    res = cls(pop_size=20, max_iter=200, seed=1).optimize(SPHERE, dim=5)
    assert res.best_fitness < 1e-3


@pytest.mark.parametrize("key", ["sma", "pso", "ga", "random"])
def test_same_seed_same_result(key):
    a = make_optimizer(key, 10, 30, seed=7).optimize(SPHERE, dim=4)
    b = make_optimizer(key, 10, 30, seed=7).optimize(SPHERE, dim=4)
    assert a.best_fitness == b.best_fitness
    assert np.array_equal(a.best_position, b.best_position)
    assert a.convergence == b.convergence


@pytest.mark.parametrize("key", ["sma", "pso", "ga", "random"])
def test_positions_stay_in_unit_cube(key):
    seen = []

    def fitness(u):
        seen.append(u.copy())
        return SPHERE(u)

    res = make_optimizer(key, 12, 25, seed=3).optimize(fitness, dim=6)
    seen = np.array(seen)
    assert seen.min() >= 0.0 and seen.max() <= 1.0
    assert ((res.best_position >= 0) & (res.best_position <= 1)).all()
    assert res.evaluations == len(seen)


@pytest.mark.parametrize("key", ["sma", "pso", "ga", "random"])
def test_convergence_is_monotone_and_full_length(key):
    res = make_optimizer(key, 8, 15, seed=2).optimize(SPHERE, dim=3)
    assert len(res.convergence) == 15 == len(res.history)
    assert all(b <= a for a, b in zip(res.convergence, res.convergence[1:]))
    assert res.convergence[-1] == res.best_fitness


def test_does_not_touch_global_rng():
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    SMA(10, 10, seed=1).optimize(SPHERE, dim=3)
    assert np.random.random() == expected


def test_sma_weights_thick_for_good_thin_for_bad():
    sma = SMA(6, 1, seed=0)
    S = np.array([5.0, 1.0, 3.0, 2.0, 6.0, 4.0])
    order = np.argsort(S)
    W = sma.compute_weights(S, order, dim=4)
    good, bad = order[:3], order[3:]
    assert (W[good] >= 1).all() and (W[bad] <= 1).all()
    assert np.allclose(W[order[0]], 1.0)          # q = 0 for the best cell


def test_ga_operators_respect_bounds():
    ga = GA(10, 5, seed=0)
    p1, p2 = np.zeros(8), np.ones(8)
    for _ in range(50):
        c1, c2 = ga.sbx(p1, p2)
        m = ga.polynomial_mutation(c1)
        assert ((c1 >= 0) & (c1 <= 1) & (c2 >= 0) & (c2 <= 1) & (m >= 0) & (m <= 1)).all()


def test_random_search_budget():
    res = RandomSearch(15, 20, seed=1).optimize(SPHERE, dim=9)
    assert res.evaluations == 300
