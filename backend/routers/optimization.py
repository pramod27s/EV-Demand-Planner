"""Per-iteration optimiser history (drives the replay animation in the UI)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.deps import get_store
from backend.schemas import History
from backend.services.results_store import Store

router = APIRouter(prefix="/api/optimization", tags=["optimization"])


@router.get("/runs", summary="Available optimiser / seed combinations")
def runs(store: Store = Depends(get_store)) -> dict[str, list[int]]:
    return store.results.history_seeds()


@router.get("/history", response_model=History, summary="Per-iteration history of one tuning run")
def history(optimizer: str, seed: int, store: Store = Depends(get_store)) -> dict:
    h = store.results.history(optimizer, seed)
    if h is None:
        raise HTTPException(status_code=404, detail={
            "msg": f"No history for optimizer={optimizer!r} seed={seed}",
            "available": store.results.history_seeds()})
    return h
