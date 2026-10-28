"""Pydantic v2 request / response schemas (shown in Swagger at ``/docs``)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Health(BaseModel):
    status: str
    models_loaded: list[str]
    artifacts_ok: bool
    missing: list[str] = []


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "station": "PALO ALTO CA / HAMILTON #1", "port_type": "Level 2", "plug_type": "J1772",
        "port_number": 1, "start_datetime": "2019-06-14T09:30:00",
        "user_prev_sessions": 12, "user_prev_mean_kwh": 8.4}})

    station: str
    port_type: str
    plug_type: str
    port_number: int = Field(1, ge=1, le=10)
    start_datetime: datetime
    user_prev_sessions: int = Field(0, ge=0)
    user_prev_mean_kwh: Optional[float] = Field(None, ge=0, le=100,
                                                description="Omit to use the global training mean")


class Prediction(BaseModel):
    model: str
    kwh: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    test_rmse: dict[str, float]
    features_used: dict[str, Any]


class Options(BaseModel):
    stations: list[str]
    port_types: list[str]
    plug_types: list[str]


class Station(BaseModel):
    name: str
    lat: float
    lon: float
    port_types: list[str]
    plug_types: list[str]
    port_numbers: list[int]
    session_count: int
    first_session: str
    last_session: str
    mean_actual_kwh: Optional[float]
    mean_predicted_kwh: Optional[float]
    test_sessions: int
    station_prev_mean_kwh: float
    hourly_mean_kwh: list[Optional[float]]


class HistoryStep(BaseModel):
    iteration: int
    best_fitness: float
    mean_fitness: float
    best_params: dict[str, float]


class History(BaseModel):
    optimizer: str
    label: str
    seed: int
    history: list[HistoryStep]


class TestSample(BaseModel):
    model: str
    actual: list[float]
    predicted: list[float]


class FeatureImportance(BaseModel):
    feature: str
    gain: float


class AreaEstimateRequest(BaseModel):
    """One plug-in event at any location (built-in or uploaded)."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "name": "MG ROAD #1", "lat": 12.9756, "lon": 77.6066, "location_mean_kwh": 8.1,
        "port_type": "Level 2", "plug_type": "J1772", "port_number": 1,
        "start_datetime": "2024-03-04T08:15:00", "user_prev_sessions": 0}})

    name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    location_mean_kwh: Optional[float] = Field(None, ge=0, le=200, description="average session energy at this location")
    port_type: str = "Level 2"
    plug_type: str = "J1772"
    port_number: int = Field(1, ge=1, le=10)
    start_datetime: datetime
    user_prev_sessions: int = Field(0, ge=0)
    user_prev_mean_kwh: Optional[float] = Field(None, ge=0, le=100)


class AreaEstimate(BaseModel):
    kwh: float
    low_kwh: float
    high_kwh: float
    typical_error_kwh: float
