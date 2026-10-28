"""Shared pytest fixtures for tests/ml and tests/api."""
import copy

import pytest

from src.config import load_config
from src.data.preprocess import run_pipeline
from tests.fixtures.synthetic import make_raw


@pytest.fixture(scope="session")
def cfg():
    c = copy.deepcopy(load_config())
    c["data"]["rare_category_min"] = 20
    return c


@pytest.fixture(scope="session")
def raw():
    return make_raw(n=3000, seed=0)


@pytest.fixture(scope="session")
def pipeline(raw, cfg):
    return run_pipeline(raw, cfg)
