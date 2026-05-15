from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.utils.validators import normalize_optional_symbol as normalize_prediction_symbol


class FeaturesInput(BaseModel):
    """Feature payload expected by the prediction endpoint."""

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
    """Prediction response returned by the API."""

    prediction: int = Field(..., ge=-1, le=1)
    confidence: float = Field(..., ge=0, le=1)


class PredictionCreate(BaseModel):
    """Schema for persisting a prediction record."""

    prediction: int = Field(..., ge=-1, le=1)
    confidence: Decimal = Field(..., ge=0, le=1)
    features: dict[str, Any]
    model_name: str | None = Field(default=None, max_length=120)
    model_version: str | None = Field(default=None, max_length=80)
    model_path: str | None = Field(default=None, max_length=500)
    symbol: str | None = Field(default=None, max_length=20)
    source: str | None = Field(default=None, max_length=80)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str | None) -> str | None:
        return normalize_prediction_symbol(value)


class PredictionRead(PredictionCreate):
    """Schema for returning stored prediction records."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: datetime


class ReloadModelResponse(BaseModel):
    """Response returned after reloading the model."""

    status: str
