"""POST /api/predict — energy prediction from all four XGBoost models."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.deps import get_store
from backend.schemas import Options, PredictRequest, PredictResponse, Prediction
from backend.services.features import build_features
from backend.services.model_store import MODELS
from backend.services.results_store import Store
from src.data.preprocess import OTHER

router = APIRouter(prefix="/api", tags=["predict"])


def _allowed(store: Store) -> dict[str, list[str]]:
    cats = store.models.meta["categories"]
    return {
        "station": sorted(store.results.stations),
        "port_type": [v for v in cats["port_type"] if v != OTHER],
        "plug_type": [v for v in cats["plug_type"] if v != OTHER],
    }


@router.get("/options", response_model=Options, summary="Dropdown values for the predict form")
def options(store: Store = Depends(get_store)) -> Options:
    a = _allowed(store)
    return Options(stations=a["station"], port_types=a["port_type"], plug_types=a["plug_type"])


@router.post("/predict", response_model=PredictResponse, summary="Predict session energy (kWh)")
def predict(req: PredictRequest, store: Store = Depends(get_store)) -> PredictResponse:
    allowed = _allowed(store)
    for field in ("station", "port_type", "plug_type"):
        if getattr(req, field) not in allowed[field]:
            raise HTTPException(status_code=422, detail={
                "msg": f"Unknown {field}: {getattr(req, field)!r}", "field": field,
                "allowed": allowed[field]})

    X, used = build_features(
        station=store.results.stations[req.station], port_type=req.port_type,
        plug_type=req.plug_type, port_number=req.port_number, start=req.start_datetime,
        user_prev_sessions=req.user_prev_sessions, user_prev_mean_kwh=req.user_prev_mean_kwh,
        meta=store.models.meta)

    preds = []
    for key in MODELS:
        kwh = float(store.models.models[key].predict(X)[0])
        preds.append(Prediction(model=store.models.display_name(key), kwh=round(max(kwh, 0.0), 3)))
    return PredictResponse(predictions=preds, test_rmse=store.results.test_rmse(), features_used=used)
