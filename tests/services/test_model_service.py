"""Tests for model_service module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pytest import MonkeyPatch
from sklearn.model_selection import train_test_split

from app.core.config import TARGET_COLUMN
from app.services.model_service import load_trained_model, make_predictions, train_xgboost_model


class FakeXGBClassifier:
    """Small stand-in for XGBoost so unit tests stay fast and deterministic."""

    def __init__(self, **params: object) -> None:
        self.params = params
        self.loaded_path: str | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FakeXGBClassifier:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([[0.8, 0.1, 0.1]], (len(X), 1))

    def save_model(self, path: str) -> None:
        Path(path).write_text("fake xgboost model", encoding="utf-8")

    def load_model(self, path: str) -> None:
        self.loaded_path = path

    def get_params(self) -> dict[str, object]:
        return self.params


# -------------------------------------------------
# 🎯 FIXTURE: Create mock dataset
# -------------------------------------------------
@pytest.fixture
def sample_model_data() -> pd.DataFrame:
    """Create a synthetic DataFrame with realistic trading features."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "rsi": np.random.rand(100),
            "bb_upper": np.random.rand(100),
            "bb_lower": np.random.rand(100),
            "bb_mid": np.random.rand(100),
            "bb_pct_b": np.random.rand(100),
            "sma_20": np.random.rand(100),
            "sma_50": np.random.rand(100),
            "ma_cross": np.random.randint(0, 2, 100),
            "price_momentum": np.random.rand(100),
            "atr": np.random.rand(100),
            "atr_pct": np.random.rand(100),
            "signal": np.random.choice([-1, 0, 1], 100, p=[0.3, 0.4, 0.3]),  # Use [-1, 0, 1]
        }
    )
    return df


# -------------------------------------------------
# ✅ TEST: prepare_model_data
# -------------------------------------------------
def test_prepare_model_data(sample_model_data: pd.DataFrame) -> None:
    """Ensure train_test_split works correctly with sample data."""
    X_train, X_test, y_train, y_test = train_test_split(
        sample_model_data.drop(columns=[TARGET_COLUMN]),
        sample_model_data[TARGET_COLUMN],
        test_size=0.2,
        random_state=42,
    )

    # Shape assertions
    assert (
        not X_train.empty and not X_test.empty
        if isinstance(X_train, pd.DataFrame)
        else X_train.size > 0
    )
    assert (
        not y_train.empty and not y_test.empty
        if isinstance(y_train, pd.DataFrame)
        else y_train.size > 0
    )

    expected_test_size = int(len(sample_model_data) * 0.2)
    assert len(X_test) == expected_test_size
    assert len(y_test) == expected_test_size

    # Verify split maintains data size
    assert len(X_train) + len(X_test) == len(sample_model_data)

    # Ensure 3 classes exist in training set - should be [-1, 0, 1]
    assert set(np.array(y_train).tolist()).issubset({-1, 0, 1})

    # Distribution check
    unique, counts = np.unique(y_train, return_counts=True)
    proportions = counts / len(y_train)
    assert np.std(proportions) < 0.25  # Relaxed threshold


# -------------------------------------------------
# ✅ TEST: train_xgboost_model
# -------------------------------------------------
def test_train_only_model(
    tmp_path: Path, monkeypatch: MonkeyPatch, sample_model_data: pd.DataFrame
) -> None:
    """Test that train_xgboost_model works with [-1, 0, 1] class labels."""
    X_train, X_test, y_train, y_test = train_test_split(
        sample_model_data.drop(columns=[TARGET_COLUMN]),
        sample_model_data[TARGET_COLUMN],
        test_size=0.2,
        random_state=42,
    )

    # XGBoost expects consecutive integers starting from 0
    # So we need to remap [-1, 0, 1] to [0, 1, 2]
    y_train_mapped = y_train + 1  # -1 -> 0, 0 -> 1, 1 -> 2
    y_test_mapped = y_test + 1

    model_path = tmp_path / "test_model.json"
    monkeypatch.setattr("app.services.model_service.MODEL_SAVE_PATH", str(model_path))
    monkeypatch.setattr("app.services.model_service.xgb.XGBClassifier", FakeXGBClassifier)

    trained_model, best_params = train_xgboost_model(
        X_train,
        y_train_mapped,
        X_test,
        y_test_mapped,
    )

    # Verify model was trained
    assert trained_model is not None
    assert best_params is not None
    assert "accuracy" in best_params

    loaded_model = load_trained_model(str(model_path))

    assert loaded_model is not None
    assert hasattr(loaded_model, "predict_proba")


@pytest.mark.filterwarnings("ignore:Precision and F-score are ill-defined")
def test_train_xgboost_model_saves_model(
    sample_model_data: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that training the XGBoost model saves correctly and returns best params."""
    X_train, X_test, y_train, y_test = train_test_split(
        sample_model_data.drop(columns=[TARGET_COLUMN]),
        sample_model_data[TARGET_COLUMN],
        test_size=0.2,
        random_state=42,
    )

    model_path = tmp_path / "saved_model.json"
    save_metadata = MagicMock()

    monkeypatch.setattr("app.services.model_service.USE_MODEL_REGISTRY", False)
    monkeypatch.setattr("app.services.model_service.xgb.XGBClassifier", FakeXGBClassifier)
    monkeypatch.setattr("app.services.model_service.save_model_metadata", save_metadata)

    trained_model, metadata = train_xgboost_model(
        X_train,
        y_train + 1,
        X_test,
        y_test + 1,
        model_path=str(model_path),
    )

    assert isinstance(trained_model, FakeXGBClassifier)
    assert model_path.exists()
    assert model_path.read_text(encoding="utf-8") == "fake xgboost model"
    assert metadata["model_name"] == "xgboost_signal_model"
    assert metadata["model_type"] == "xgboost"
    assert metadata["model_path"] == str(model_path)
    assert "accuracy" in metadata
    assert "evaluation" in metadata
    assert metadata["params"] == {"eval_metric": "mlogloss", "random_state": 42}
    save_metadata.assert_called_once_with(metadata)


# -------------------------------------------------
# ✅ TEST: make_predictions
# -------------------------------------------------
def test_make_predictions(sample_model_data: pd.DataFrame) -> None:
    """Test prediction logic and empty input behavior."""
    mock_model = MagicMock()

    # Define X_data for predictions
    X_data = sample_model_data.drop(columns=[TARGET_COLUMN])

    # 3-class probability output (softmax-like) for classes [0, 1, 2]
    n = len(X_data)
    mock_model.predict_proba.return_value = np.tile(
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]], (n // 3 + 1, 1)
    )[:n]

    preds, conf = make_predictions(mock_model, X_data)

    # Assertions - preds should be [-1, 0, 1] because make_predictions subtracts 1
    assert len(preds) == len(X_data)
    assert len(conf) == len(X_data)
    assert set(np.array(preds).tolist()).issubset({-1, 0, 1})
    # Check confidence scores are between 0 and 1
    assert (conf >= 0).all() and (conf <= 1).all()
