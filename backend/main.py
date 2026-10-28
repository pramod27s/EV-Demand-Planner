"""FastAPI app serving the SMA-tuned EV demand models and experiment results.

    uvicorn backend.main:app --reload --port 8000      # Swagger UI at http://localhost:8000/docs

Artifacts are read from ``results/`` (override with the ``RESULTS_DIR`` environment
variable). If they are missing the app still starts and every data endpoint returns 503.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import area, optimization, predict, results, stations
from backend.schemas import Health
from backend.services.results_store import Store
from src.config import load_config, results_dir


def create_app(results_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = Store(results_path or results_dir(load_config()))   # load once
        yield

    app = FastAPI(title="SMA-Optimised XGBoost — EV Charging Demand API", version="1.0.0",
                  description="Serves models and results exported by the ML pipeline. "
                              "Never trains or tunes.", lifespan=lifespan)

    origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
    if os.environ.get("FRONTEND_ORIGIN"):
        origins.append(os.environ["FRONTEND_ORIGIN"])
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health", response_model=Health, tags=["health"])
    def health() -> Health:
        store: Store = app.state.store
        return Health(status="ok", models_loaded=sorted(store.models.models),
                      artifacts_ok=store.ok, missing=store.missing)

    for r in (predict.router, stations.router, results.router, optimization.router, area.router):
        app.include_router(r)
    return app


app = create_app()
