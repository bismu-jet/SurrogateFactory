"""FastAPI application for serving trained surrogate models over HTTP."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from surrogate_factory.factory import SurrogateFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("SURROGATE_MODEL_PATH", "model.pkl")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_factory: Optional[SurrogateFactory] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; clean up on shutdown."""
    global _factory
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    model_file = Path(MODEL_PATH)
    if model_file.exists():
        logger.info("Loading model from %s", model_file)
        _factory = SurrogateFactory.load_model(model_file)
        logger.info("Model ready (type=%s, QoIs=%s).", _factory.model_type.value, _factory.qoi_names)
    else:
        logger.warning("Model file not found at %s. /predict will return 503.", model_file)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SurrogateFactory API",
    description="REST API for surrogate model predictions (Kriging / RBF / Neural Network).",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    """Input features for prediction."""

    features: List[Dict[str, Any]] = Field(
        ...,
        description="List of feature dictionaries. Each dict maps feature names to values.",
        examples=[[{"Feature_A": 5.0, "Feature_B": 105.0, "Categoria": "A"}]],
    )


class PredictionResponse(BaseModel):
    """Prediction results per QoI."""

    predictions: Dict[str, List[List[float]]] = Field(
        ...,
        description="Mapping from QoI name to a list of predicted time-series vectors.",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    qoi_names: List[str] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def health_check() -> HealthResponse:
    """Kubernetes liveness / readiness probe."""
    return HealthResponse(
        status="ok" if _factory is not None else "no_model",
        model_loaded=_factory is not None,
        model_type=_factory.model_type.value if _factory else None,
        qoi_names=list(_factory.qoi_names) if _factory else [],
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["inference"],
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Run inference on the loaded surrogate model.

    Accepts a list of feature dictionaries and returns per-QoI predictions.
    """
    if _factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Set SURROGATE_MODEL_PATH and restart.",
        )

    try:
        df = pd.DataFrame(request.features)
        raw_preds = _factory.predict(df)
    except (ValueError, KeyError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    response_data: Dict[str, List[List[float]]] = {}
    for qoi_name, arr in raw_preds.items():
        response_data[qoi_name] = arr.tolist()

    return PredictionResponse(predictions=response_data)


@app.get("/model/info", tags=["info"])
async def model_info() -> Dict[str, Any]:
    """Return metadata about the loaded model."""
    if _factory is None:
        raise HTTPException(status_code=503, detail="No model loaded.")
    return {
        "model_type": _factory.model_type.value,
        "qoi_names": _factory.qoi_names,
        "feature_names": _factory.feature_names,
        "metadata": _factory.metadata,
    }
