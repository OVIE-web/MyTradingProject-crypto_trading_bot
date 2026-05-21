"""Unit tests for the Prediction model."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.prediction import Prediction


def test_prediction_model_persists_features_and_metadata(model_session: Session) -> None:
    """Prediction rows should retain model output, features, and provenance."""
    prediction = Prediction(
        prediction=1,
        confidence=Decimal("0.876543"),
        features={"rsi": 28.2, "atr": 1.5},
        model_name="xgboost_signal_model",
        model_version="v1",
        model_path="models/xgboost_model.json",
        symbol="BTCUSDT",
        source="api",
    )

    model_session.add(prediction)
    model_session.commit()
    model_session.refresh(prediction)

    loaded = model_session.query(Prediction).filter_by(symbol="BTCUSDT").one()

    assert loaded.id is not None
    assert loaded.prediction == 1
    assert loaded.confidence == Decimal("0.876543")
    assert loaded.features == {"rsi": 28.2, "atr": 1.5}
    assert loaded.model_name == "xgboost_signal_model"
    assert loaded.model_version == "v1"
    assert loaded.model_path == "models/xgboost_model.json"
    assert loaded.source == "api"
    assert isinstance(loaded.timestamp, datetime)


def test_prediction_requires_prediction_confidence_and_features(model_session: Session) -> None:
    """Prediction output, confidence, and features are required columns."""
    prediction = Prediction(prediction=1, confidence=Decimal("0.5"))
    model_session.add(prediction)

    with pytest.raises(IntegrityError):
        model_session.commit()
