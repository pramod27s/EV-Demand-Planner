"""Project configuration: loads ``config.yaml`` and provides seeding helpers.

Every tunable number in the project lives in ``config.yaml``; modules call
:func:`load_config` instead of hard-coding values.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config (default: ``<project>/config.yaml``)."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path: str | Path) -> Path:
    """Turn a config path (relative to the project root) into an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def processed_dir(cfg: dict) -> Path:
    return resolve(cfg["paths"]["processed_dir"])


def results_dir(cfg: dict) -> Path:
    """Results root; can be overridden with the ``RESULTS_DIR`` environment variable."""
    return resolve(os.environ.get("RESULTS_DIR", cfg["paths"]["results_dir"]))


def set_seeds(seed: int) -> None:
    """Seed Python's and NumPy's global RNGs.

    The optimisers never use the global RNG (they own a ``np.random.default_rng``);
    this only makes incidental library randomness reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
