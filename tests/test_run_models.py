"""Tests for CLI modes and training pipeline functionality."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.config import FEATURE_COLUMNS
from src.model_manager import load_trained_model, train_xgboost_model


@pytest.fixture
def sample_training_data():
    """Create a small but valid dataset for quick training tests."""
    n_samples = 100
    n_features = len(FEATURE_COLUMNS)
    rng = np.random.RandomState(42)

    # Generate synthetic features
    X = pd.DataFrame(rng.randn(n_samples, n_features), columns=FEATURE_COLUMNS)

    # Generate balanced labels in {-1, 0, 1}
    y = pd.Series(np.tile([-1, 0, 1], n_samples // 3 + 1)[:n_samples])

    return X, y


def test_train_only_model(tmp_path, monkeypatch, sample_training_data):
    """Test the --train-only mode creates and saves a valid model."""
    X, y = sample_training_data
    model_path = tmp_path / "test_model.json"

    # Patch MODEL_SAVE_PATH to use tmp_path
    monkeypatch.setattr("src.model_manager.MODEL_SAVE_PATH", str(model_path))

    # Split data (small test for speed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Remap class labels from [-1, 0, 1] to [0, 1, 2] for XGBoost
    # XGBoost expects consecutive integers starting from 0
    y_train_mapped = y_train + 1  # -1 -> 0, 0 -> 1, 1 -> 2
    y_test_mapped = y_test + 1

    # Train model with mapped labels
    model, best_params = train_xgboost_model(X_train, y_train_mapped, X_test, y_test_mapped)

    # Verify model file exists
    assert model_path.exists(), "Model file was not created"

    # Verify we can load and use the model
    loaded_model = load_trained_model(str(model_path))
    assert loaded_model is not None, "Loaded model is None"
    assert hasattr(loaded_model, "predict_proba"), "Loaded model missing predict_proba"

    # Basic prediction shape check
    pred_proba = loaded_model.predict_proba(X_test)
    assert pred_proba.shape == (len(X_test), 3), "Wrong prediction shape"
