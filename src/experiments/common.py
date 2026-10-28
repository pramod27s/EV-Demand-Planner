"""Helpers shared by the experiment scripts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import TARGET, load_splits, to_model_frame


@dataclass
class Data:
    """Model-ready matrices for the three chronological splits."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    meta: dict

    def X(self, split: str) -> pd.DataFrame:
        return to_model_frame(getattr(self, split), self.meta)

    def y(self, split: str) -> np.ndarray:
        return getattr(self, split)[TARGET].to_numpy()


def load_data(cfg: dict) -> Data:
    return Data(*load_splits(cfg))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_jsonable), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")
