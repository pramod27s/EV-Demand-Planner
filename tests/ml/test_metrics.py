"""Metrics on hand-computed toy arrays."""
import math

import pytest

from src.evaluation.metrics import all_metrics, mae, r2, rmse, smape

Y = [1.0, 2.0, 3.0, 4.0]
P = [1.0, 3.0, 2.0, 4.0]          # errors: 0, -1, +1, 0


def test_rmse():
    assert rmse(Y, P) == pytest.approx(math.sqrt(0.5))


def test_mae():
    assert mae(Y, P) == pytest.approx(0.5)


def test_r2():
    # SS_res = 2, SS_tot = 5 -> 1 - 2/5
    assert r2(Y, P) == pytest.approx(0.6)


def test_smape():
    # terms: 0, 2*1/5, 2*1/5, 0 -> mean 0.2 -> 20 %
    assert smape(Y, P) == pytest.approx(20.0, abs=1e-6)


def test_perfect_prediction():
    m = all_metrics(Y, Y)
    assert m["rmse"] == 0 and m["mae"] == 0 and m["r2"] == 1 and m["smape"] == pytest.approx(0)
