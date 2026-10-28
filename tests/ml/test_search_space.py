"""Search-space decoding and the cached fitness function."""
import numpy as np
import pytest

from src.models.xgb_space import SearchSpace, XGBFitness

SPACE = SearchSpace()


def test_decode_zero_gives_lower_bounds():
    params = SPACE.decode(np.zeros(SPACE.dim))
    for p in SPACE.PARAMS:
        assert params[p.name] == pytest.approx(p.lo)


def test_decode_one_gives_upper_bounds():
    params = SPACE.decode(np.ones(SPACE.dim))
    for p in SPACE.PARAMS:
        assert params[p.name] == pytest.approx(p.hi)


def test_integer_params_are_ints():
    for u in np.random.default_rng(0).random((20, SPACE.dim)):
        params = SPACE.decode(u)
        assert isinstance(params["n_estimators"], int)
        assert isinstance(params["max_depth"], int)


@pytest.mark.parametrize("name", ["learning_rate", "reg_lambda", "reg_alpha"])
def test_log_params_monotonic_and_log_uniform(name):
    i = SPACE.names.index(name)
    values = []
    for ui in np.linspace(0, 1, 11):
        u = np.full(SPACE.dim, 0.5)
        u[i] = ui
        values.append(SPACE.decode(u)[name])
    assert all(b > a for a, b in zip(values, values[1:]))
    p = SPACE.PARAMS[i]
    assert values[5] == pytest.approx(np.sqrt(p.lo * p.hi), rel=1e-3)   # midpoint = geometric mean


def test_normalise_inverts_decode():
    u = np.random.default_rng(1).random(SPACE.dim)
    back = SPACE.normalise(SPACE.decode(u))
    assert np.allclose(back[2:], u[2:], atol=1e-3)          # continuous params
    assert np.allclose(back[:2], u[:2], atol=0.07)          # integers lose precision


def test_fitness_cache(pipeline):
    from src.experiments.common import Data
    d = Data(pipeline["train"], pipeline["val"], pipeline["test"], pipeline["meta"])
    fit = XGBFitness(d.X("train"), d.y("train"), d.X("val"), d.y("val"),
                     seed=1, sample_size=500, early_stopping_rounds=5)
    u = np.zeros(SPACE.dim)
    a, b = fit(u), fit(u.copy())
    assert a == b > 0
    assert fit.calls == 2 and fit.real_evaluations == 1 and fit.cache_hits == 1
