"""Loads the exported result JSON files (``results/api``) once at startup."""
from __future__ import annotations

import json
from pathlib import Path

from backend.services.model_store import MODELS, ModelStore
from src.config import load_config, resolve

REQUIRED = ["dataset_summary", "stations", "comparison", "convergence", "hyperparams",
            "test_sample", "feature_importance"]

PIPELINE_MSG = "Run the ML pipeline first: python -m src.experiments.export_artifacts"


class ResultsStore:
    def __init__(self, api_dir: Path):
        self.api_dir = api_dir
        self.data: dict[str, object] = {}
        self.missing: list[str] = []
        for name in REQUIRED:
            path = api_dir / f"{name}.json"
            if path.exists():
                self.data[name] = json.loads(path.read_text(encoding="utf-8"))
            else:
                self.missing.append(str(path))
        self.stations = {s["name"]: s for s in self.data.get("stations", [])}

    @property
    def ok(self) -> bool:
        return not self.missing

    def history(self, optimizer: str, seed: int) -> dict | None:
        path = self.api_dir / f"history_{optimizer}_seed{seed}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def history_seeds(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for p in self.api_dir.glob("history_*_seed*.json"):
            opt, seed = p.stem[len("history_"):].rsplit("_seed", 1)
            out.setdefault(opt, []).append(int(seed))
        return {k: sorted(v) for k, v in out.items()}

    def test_rmse(self) -> dict[str, float]:
        """Test RMSE per served model, keyed by display name (from the comparison table)."""
        rows = {r["model"]: r for r in self.data.get("comparison", {}).get("rows", [])}
        return {name: rows[name]["rmse"] for _, name in MODELS.values() if name in rows}


class Store:
    """Everything the API serves; ``ok`` is False until the ML pipeline has been run."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.models = ModelStore(results_dir / "models")
        self.results = ResultsStore(results_dir / "api")
        cfg = load_config()
        self.placement_cfg = cfg["placement"]
        self.sample_dir = resolve(cfg["paths"]["sample_dir"])

    @property
    def ok(self) -> bool:
        return self.models.ok and self.results.ok

    @property
    def missing(self) -> list[str]:
        return self.models.missing + self.results.missing
