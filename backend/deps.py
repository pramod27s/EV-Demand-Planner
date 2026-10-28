"""FastAPI dependencies: access to the loaded artifact store."""
from __future__ import annotations

from fastapi import HTTPException, Request

from backend.services.results_store import PIPELINE_MSG, Store


def get_store(request: Request) -> Store:
    """The artifact store; responds 503 if the ML pipeline has not been run yet."""
    store: Store = request.app.state.store
    if not store.ok:
        raise HTTPException(status_code=503, detail=PIPELINE_MSG)
    return store
