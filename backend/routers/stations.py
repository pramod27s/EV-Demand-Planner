"""Station listing and details."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.deps import get_store
from backend.schemas import Station
from backend.services.results_store import Store

router = APIRouter(prefix="/api", tags=["stations"])


@router.get("/stations", response_model=list[Station], summary="All stations with usage statistics")
def stations(store: Store = Depends(get_store)) -> list[dict]:
    return store.results.data["stations"]


@router.get("/stations/{station_name:path}", response_model=Station,
            summary="One station, including its 24-hour mean kWh profile")
def station(station_name: str, store: Store = Depends(get_store)) -> dict:
    s = store.results.stations.get(station_name)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Unknown station {station_name!r}")
    return s
