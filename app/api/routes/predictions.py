# app/api/routes/predictions.py
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import FEATURE_COLUMNS
from app.core.security import get_current_user
from app.schemas.prediction_schema import FeaturesInput, PredictionResponse, ReloadModelResponse
from app.services.model_service import load_trained_model, make_predictions

logger = logging.getLogger(__name__)

router = APIRouter()

# --------------------------------------------------------------------------
# Load model once at startup
# --------------------------------------------------------------------------
model = load_trained_model()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict(
    features: FeaturesInput,
    user: str = Depends(get_current_user),
) -> PredictionResponse:
    """
    Run model prediction (JWT protected).
    """
    if model is None:
        logger.error("Prediction requested but model is not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        df = pd.DataFrame([features.model_dump()]).loc[:, FEATURE_COLUMNS]
        X: np.ndarray[Any, Any] = df.to_numpy()

        preds, probs = make_predictions(model, X)

        return PredictionResponse(
            prediction=int(preds[0]),
            confidence=float(probs[0]),
        )

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@router.post(
    "/reload-model",
    response_model=ReloadModelResponse,
    status_code=status.HTTP_200_OK,
)
def reload_model(user: str = Depends(get_current_user)) -> ReloadModelResponse:
    """
    Reload model from disk (admin-only endpoint).
    """
    global model
    try:
        model = load_trained_model()
        return ReloadModelResponse(status="Model reloaded successfully")
    except Exception:
        logger.exception("Model reload failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload model",
        )
