"""Loads the exported XGBoost models once at startup (the API never trains or tunes)."""
from __future__ import annotations

import json
from pathlib import Path

from xgboost import XGBRegressor

# model key -> (file name, display name)
MODELS = {
    "default": ("xgb_default.json", "XGBoost (default)"),
    "sma": ("xgb_sma.json", "XGBoost + SMA"),
    "pso": ("xgb_pso.json", "XGBoost + PSO"),
    "ga": ("xgb_ga.json", "XGBoost + GA"),
}


class ModelStore:
    def __init__(self, models_dir: Path):
        self.models: dict[str, XGBRegressor] = {}
        self.meta: dict | None = None
        self.missing: list[str] = []
        meta_path = models_dir / "feature_meta.json"
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            self.missing.append(str(meta_path))
        for key, (fname, _) in MODELS.items():
            path = models_dir / fname
            if not path.exists():
                self.missing.append(str(path))
                continue
            model = XGBRegressor()
            model.load_model(path)
            self.models[key] = model

    @property
    def ok(self) -> bool:
        return self.meta is not None and len(self.models) == len(MODELS)

    @staticmethod
    def display_name(key: str) -> str:
        return MODELS[key][1]
