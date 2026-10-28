"""Analyse an uploaded area dataset: estimates, per-location demand and SMA charger placement.

The CSV is sent as the raw request body (``Content-Type: text/csv``), which keeps the API free
of multipart dependencies.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, Response

from backend.deps import get_store
from backend.schemas import AreaEstimate, AreaEstimateRequest
from backend.services import area, samples
from backend.services.features import build_features
from backend.services.results_store import Store

router = APIRouter(prefix="/api/area", tags=["area analysis"])

MAX_BYTES = 25 * 1024 * 1024
CSV_BODY = {"requestBody": {"required": True, "content": {"text/csv": {"schema": {"type": "string", "format": "binary"}}}}}

TEMPLATE = (
    "Station Name,Latitude,Longitude,Start Date,Port Type,Plug Type,Port Number,User ID,Energy (kWh)\n"
    "MG ROAD #1,12.9756,77.6066,2024-03-04 08:15,Level 2,J1772,1,u001,9.4\n"
    "MG ROAD #1,12.9756,77.6066,2024-03-04 18:40,Level 2,J1772,2,u017,6.1\n"
    "INDIRANAGAR #2,12.9784,77.6408,2024-03-05 12:05,Level 2,J1772,1,u001,\n"
)


async def _body(request: Request) -> bytes:
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=422, detail="Upload a CSV file as the request body.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (limit 25 MB).")
    return raw


@router.post("/analyze", summary="Estimates, location demand and SMA-suggested charger sites", openapi_extra=CSV_BODY)
async def analyze(request: Request, k: int = Query(3, ge=1, le=10, description="number of sites to suggest"),
                  store: Store = Depends(get_store)) -> dict:
    raw = await _body(request)
    try:
        return area.analyse(raw, store.models.models["sma"], store.models.meta, k, store.placement_cfg)
    except area.AreaInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/predictions", summary="Every uploaded row with its estimated kWh (CSV download)",
             response_class=PlainTextResponse, openapi_extra=CSV_BODY)
async def predictions(request: Request, store: Store = Depends(get_store)) -> Response:
    raw = await _body(request)
    try:
        csv = area.predictions_csv(raw, store.models.models["sma"], store.models.meta)
    except area.AreaInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(csv, media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="energy_estimates.csv"'})


@router.get("/template", summary="CSV template with the recognised columns", response_class=PlainTextResponse)
def template() -> Response:
    return Response(TEMPLATE, media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="area_template.csv"'})


@router.get("/sample", summary="Sample dataset (Palo Alto test sessions) to try the analysis",
            response_class=PlainTextResponse)
def sample(store: Store = Depends(get_store)) -> Response:
    path = store.results.api_dir / "sample_upload.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No sample file; re-run python -m src.experiments.export_artifacts")
    return Response(path.read_bytes(), media_type="text/csv")


@router.post("/estimate", response_model=AreaEstimate, summary="Estimate one session at any location")
def estimate(req: AreaEstimateRequest, store: Store = Depends(get_store)) -> AreaEstimate:
    meta = store.models.meta
    station = {"name": req.name, "lat": req.lat, "lon": req.lon,
               "station_prev_mean_kwh": req.location_mean_kwh or meta["global_train_mean_kwh"]}
    X, _ = build_features(station=station, port_type=req.port_type, plug_type=req.plug_type,
                          port_number=req.port_number, start=req.start_datetime,
                          user_prev_sessions=req.user_prev_sessions, user_prev_mean_kwh=req.user_prev_mean_kwh,
                          meta=meta)
    kwh = max(0.0, float(store.models.models["sma"].predict(X)[0]))
    err = store.results.test_rmse().get("XGBoost + SMA", 0.0)
    return AreaEstimate(kwh=round(kwh, 3), low_kwh=round(max(0.0, kwh - err), 3),
                        high_kwh=round(kwh + err, 3), typical_error_kwh=round(err, 3))


@router.get("/samples", summary="Built-in sample datasets (Palo Alto, Hyderabad, Boulder, Cary, Leeds)")
def list_samples(store: Store = Depends(get_store)) -> list[dict]:
    return samples.available(store.results.api_dir, store.sample_dir)


@router.get("/samples/{sample_id}", summary="Download one sample dataset as CSV", response_class=PlainTextResponse)
def get_sample(sample_id: str, store: Store = Depends(get_store)) -> Response:
    path = samples.sample_path(sample_id, store.results.api_dir, store.sample_dir)
    if path is None:
        raise HTTPException(status_code=404, detail={"msg": f"Unknown or missing sample {sample_id!r}",
                                                     "available": [s["id"] for s in samples.available(
                                                         store.results.api_dir, store.sample_dir)]})
    return Response(path.read_bytes(), media_type="text/csv")
