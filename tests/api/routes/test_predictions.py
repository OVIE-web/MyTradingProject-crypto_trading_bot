from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.routes import predictions
from app.main import app

client = TestClient(app)

VALID_FEATURES = {
    "rsi": 65.2,
    "bb_upper": 108.2,
    "bb_lower": 100.0,
    "bb_mid": 104.0,
    "bb_pct_b": 0.52,
    "sma_20": 103.0,
    "sma_50": 102.0,
    "ma_cross": 1.0,
    "price_momentum": 0.5,
    "atr": 2.34,
    "atr_pct": 0.02,
}


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def allow_authenticated_user() -> None:
    app.dependency_overrides[predictions.get_current_user] = lambda: "test-user"


class TestPredictionsRouter:
    """Tests for the FastAPI predictions router."""

    def test_predict_requires_authentication(self) -> None:
        response = client.post("/predictions/predict", json=VALID_FEATURES)

        assert response.status_code == 401

    def test_predict_rejects_invalid_auth_header(self) -> None:
        response = client.post(
            "/predictions/predict",
            json=VALID_FEATURES,
            headers={"Authorization": "Bearer invalid-token"},
        )

        assert response.status_code == 401

    def test_predict_validates_required_feature_fields(self) -> None:
        allow_authenticated_user()

        response = client.post("/predictions/predict", json={"rsi": 65.2})

        assert response.status_code == 422

    def test_predict_returns_503_when_model_is_not_loaded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        allow_authenticated_user()
        monkeypatch.setattr(predictions, "model", None)

        response = client.post("/predictions/predict", json=VALID_FEATURES)

        assert response.status_code == 503
        assert response.json()["error"]["message"] == "Model not loaded"

    def test_predict_returns_prediction_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        allow_authenticated_user()
        fake_model = object()
        make_predictions = MagicMock(return_value=(np.array([1]), np.array([0.87])))
        monkeypatch.setattr(predictions, "model", fake_model)
        monkeypatch.setattr(predictions, "make_predictions", make_predictions)

        response = client.post("/predictions/predict", json=VALID_FEATURES)

        assert response.status_code == 200
        assert response.json() == {"prediction": 1, "confidence": 0.87}
        make_predictions.assert_called_once()

    def test_reload_model_requires_authentication(self) -> None:
        response = client.post("/predictions/reload-model")

        assert response.status_code == 401

    def test_reload_model_returns_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        allow_authenticated_user()
        fake_model = object()
        monkeypatch.setattr(predictions, "load_trained_model", lambda: fake_model)

        response = client.post("/predictions/reload-model")

        assert response.status_code == 200
        assert response.json() == {"status": "Model reloaded successfully"}
        assert predictions.model is fake_model
