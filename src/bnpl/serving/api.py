"""FastAPI serving layer for BNPL default prediction.

Exposes endpoints:
    - GET  /          : Service information and documentation links.
    - GET  /health    : Health check with model version and threshold.
    - POST /predict   : Accepts a loan application, returns APPROVE/DENY.
    - GET  /ab-summary: A/B test summary (if challenger loaded).

The champion XGBoost model is loaded once at startup via the FastAPI
lifespan context manager. If CHALLENGER_MODEL_PATH env var is set,
an A/B router handles traffic splitting.

Start the server::

    uvicorn src.bnpl.serving.api:app --reload --port 8000
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException

from bnpl.serving.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from bnpl.serving.schemas import (
    HealthResponse,
    LoanApplication,
    PredictionResponse,
)


_MODEL_PATH = os.getenv("MODEL_PATH", "models/champion_xgboost.pkl")
_CONFIG_PATH = os.getenv("CONFIG_PATH", "reports/data_prep_config.json")
_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.45"))
_CHALLENGER_PATH = os.getenv("CHALLENGER_MODEL_PATH", "")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model at startup with graceful degradation.

    If MODEL_PATH does not exist, the service starts in degraded
    mode where /health returns status=degraded and /predict returns 503.
    This allows Render deployments to start even before model files
    are present.
    """
    from bnpl.logger import get_logger

    logger = get_logger("bnpl.serving.api")

    app.state.degraded = False
    app.state.predictor = None
    app.state.ab_router = None

    try:
        from bnpl.models.predictor import Predictor

        predictor = Predictor(
            model_path=_MODEL_PATH,
            config_path=_CONFIG_PATH,
            threshold=_THRESHOLD,
        )
        app.state.predictor = predictor

        from bnpl.serving.ab_router import ABRouter

        router = ABRouter(predictor)

        if _CHALLENGER_PATH and os.path.exists(_CHALLENGER_PATH):
            router.load_challenger(_CHALLENGER_PATH, _CONFIG_PATH, _THRESHOLD)
            logger.info("Challenger loaded for A/B testing: %s", _CHALLENGER_PATH)

        app.state.ab_router = router

    except FileNotFoundError as exc:
        logger.warning("Model not found at startup: %s", exc)
        app.state.degraded = True
    except Exception as exc:
        logger.warning("Model loading failed: %s", exc)
        app.state.degraded = True

    yield


app = FastAPI(
    title="BNPL Default Prediction API",
    description=(
        "Predicts default probability for BNPL loan applications. "
        "Returns an APPROVE or DENY decision based on the champion "
        "XGBoost model and configurable business threshold."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)


@app.get("/", response_model=dict)
def root() -> dict:
    """Return basic service information and documentation links.

    Returns:
        dict: Service name, documentation URL, and endpoint paths.
    """
    return {
        "message": "BNPL Default Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "ab_summary": "/ab-summary",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check service health and report model version and threshold.

    Returns:
        HealthResponse: Current status with model metadata.
    """
    if app.state.degraded:
        return HealthResponse(
            status="degraded",
            model_version="none",
            threshold=_THRESHOLD,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    predictor = app.state.predictor
    return HealthResponse(
        status="healthy",
        model_version=predictor.model_version,
        threshold=predictor.threshold,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication) -> PredictionResponse:
    """Score a loan application and return an APPROVE/DENY decision.

    If a challenger model is loaded via CHALLENGER_MODEL_PATH env var,
    the ABRouter handles traffic splitting. Otherwise, all traffic
    goes to the champion model.

    Args:
        application: Validated loan application with 17 fields.

    Returns:
        PredictionResponse: Decision, probability, and metadata.

    Raises:
        HTTPException: 503 if model is not available (degraded mode).
        HTTPException: 500 if prediction fails unexpectedly.
    """
    if app.state.degraded or app.state.predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Service is degraded. "
                   "Please check that model file exists at deployment.",
        )

    try:
        raw = application.model_dump()
        router = app.state.ab_router

        if router is not None:
            request_id = str(uuid.uuid4())
            result = router.route(raw, request_id)
        else:
            result = app.state.predictor.predict(raw)

        return PredictionResponse(**{
            k: v for k, v in result.items()
            if k in PredictionResponse.model_fields
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/ab-summary", response_model=dict)
def ab_summary() -> dict:
    """Return A/B test summary statistics.

    Returns:
        dict: Champion vs challenger comparison metrics.
              Empty summary if no challenger is loaded or no requests logged.
    """
    router = app.state.ab_router
    if router is not None:
        return router.get_summary()
    return {
        "total_requests": 0, "champion_requests": 0,
        "challenger_requests": 0, "has_challenger": False,
    }
