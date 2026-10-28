"""Experiment results: dataset summary, comparison table, convergence, hyperparameters."""
from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException

from backend.deps import get_store
from backend.schemas import FeatureImportance, TestSample
from backend.services.results_store import Store

router = APIRouter(prefix="/api", tags=["results"])


@router.get("/dataset/summary", summary="Row counts per preprocessing step, splits, target stats")
def dataset_summary(store: Store = Depends(get_store)) -> dict[str, Any]:
    return store.results.data["dataset_summary"]


@router.get("/results/comparison", summary="Final comparison table (test set)")
def comparison(store: Store = Depends(get_store)) -> dict[str, Any]:
    return store.results.data["comparison"]


@router.get("/results/convergence", summary="Convergence curves (mean, std and every seed)")
def convergence(optimizer: Optional[str] = None, store: Store = Depends(get_store)) -> dict[str, Any]:
    conv = store.results.data["convergence"]
    if optimizer is None:
        return conv
    if optimizer not in conv:
        raise HTTPException(status_code=404, detail={"msg": f"Unknown optimizer {optimizer!r}",
                                                     "allowed": list(conv)})
    return {optimizer: conv[optimizer]}


@router.get("/results/hyperparams", summary="Best hyperparameters per optimiser")
def hyperparams(store: Store = Depends(get_store)) -> dict[str, Any]:
    return store.results.data["hyperparams"]


@router.get("/results/test-sample", response_model=TestSample,
            summary="Actual vs predicted kWh for 2,000 random test sessions")
def test_sample(model: Literal["default", "sma"] = "sma", store: Store = Depends(get_store)) -> TestSample:
    s = store.results.data["test_sample"]
    return TestSample(model=model, actual=s["actual"], predicted=s[model])


@router.get("/results/feature-importance", response_model=list[FeatureImportance],
            summary="Top-15 gain importances of the SMA-tuned model")
def feature_importance(store: Store = Depends(get_store)) -> list[dict]:
    return store.results.data["feature_importance"]
