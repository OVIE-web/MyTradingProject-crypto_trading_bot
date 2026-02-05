# src/routers/predict.py
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.auth import get_current_user
from src.config import FEATURE_COLUMNS
from src.model_manager import load_trained_model, make_predictions

logger = logging.getLogger(__name__)

router = APIRouter()

# --------------------------------------------------------------------------
# Load model once at startup
# --------------------------------------------------------------------------
model = load_trained_model()


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class FeaturesInput(BaseModel):
    rsi: float
    bb_upper: float
    bb_lower: float
    bb_mid: float
    bb_pct_b: float
    sma_20: float
    sma_50: float
    ma_cross: float
    price_momentum: float
    atr: float
    atr_pct: float


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float


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
    user: dict[str, str] = Depends(get_current_user),
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
        df = pd.DataFrame([features.model_dump()])[FEATURE_COLUMNS]
        X: np.ndarray = df.to_numpy()

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


@router.post("/reload-model", status_code=status.HTTP_200_OK)
def reload_model(user: dict[str, str] = Depends(get_current_user)) -> dict[str, str]:
    """
    Reload model from disk (admin-only endpoint).
    """
    global model
    try:
        model = load_trained_model()
        return {"status": "Model reloaded successfully"}
    except Exception:
        logger.exception("Model reload failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload model",
        )
