from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.schemas.prediction_schema import (
    FeaturesInput,
    PredictionCreate,
    PredictionRead,
    PredictionResponse,
    ReloadModelResponse,
)

VALID_FEATURES = {
    "rsi": 52.1,
    "bb_upper": 70200.0,
    "bb_lower": 68100.0,
    "bb_mid": 69150.0,
    "bb_pct_b": 0.71,
    "sma_20": 69000.0,
    "sma_50": 68500.0,
    "ma_cross": 1.0,
    "price_momentum": 0.04,
    "atr": 450.0,
    "atr_pct": 0.006,
}


def test_features_input_accepts_complete_feature_payload() -> None:
    features = FeaturesInput(**VALID_FEATURES)

    assert features.model_dump() == VALID_FEATURES


def test_features_input_rejects_missing_required_feature() -> None:
    payload = VALID_FEATURES.copy()
    payload.pop("rsi")

    with pytest.raises(ValidationError):
        FeaturesInput(**payload)


def test_prediction_response_bounds_prediction_and_confidence() -> None:
    response = PredictionResponse(prediction=1, confidence=0.87)

    assert response.prediction == 1
    assert response.confidence == 0.87

    with pytest.raises(ValidationError):
        PredictionResponse(prediction=2, confidence=0.87)

    with pytest.raises(ValidationError):
        PredictionResponse(prediction=1, confidence=1.1)


def test_prediction_create_normalizes_symbol_and_coerces_confidence() -> None:
    prediction = PredictionCreate(
        prediction=-1,
        confidence="0.64",
        features={"rsi": 52.1},
        model_name="xgboost",
        model_version="v1",
        model_path="models/xgboost_model.json",
        symbol=" btc/usdt ",
        source="api",
    )

    assert prediction.confidence == Decimal("0.64")
    assert prediction.symbol == "BTCUSDT"


def test_prediction_create_allows_optional_symbol() -> None:
    prediction = PredictionCreate(
        prediction=0,
        confidence=Decimal("0.50"),
        features={"rsi": 52.1},
        symbol=None,
    )

    assert prediction.symbol is None


def test_prediction_create_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        PredictionCreate(prediction=9, confidence=Decimal("0.5"), features={})

    with pytest.raises(ValidationError):
        PredictionCreate(prediction=1, confidence=Decimal("1.5"), features={})

    with pytest.raises(ValidationError):
        PredictionCreate(
            prediction=1,
            confidence=Decimal("0.5"),
            features={},
            model_name="x" * 121,
        )


def test_prediction_read_validates_from_orm_like_object() -> None:
    timestamp = datetime(2026, 5, 16, tzinfo=UTC)
    db_prediction = SimpleNamespace(
        id=10,
        prediction=1,
        confidence=Decimal("0.91"),
        features={"rsi": 61.0},
        model_name="xgboost",
        model_version="v1",
        model_path="models/xgboost_model.json",
        symbol="eth-usdt",
        source="scheduler",
        timestamp=timestamp,
    )

    prediction = PredictionRead.model_validate(db_prediction)

    assert prediction.id == 10
    assert prediction.symbol == "ETHUSDT"
    assert prediction.timestamp == timestamp


def test_reload_model_response_exposes_status() -> None:
    response = ReloadModelResponse(status="reloaded")

    assert response.status == "reloaded"
