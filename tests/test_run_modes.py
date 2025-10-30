"""Tests for CLI modes and training pipeline functionality."""
import os
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from src.model_manager import train_xgboost_model, load_trained_model
from src.config import FEATURE_COLUMNS, MODEL_SAVE_PATH


@pytest.fixture
def sample_training_data():
    """Create a small but valid dataset for quick training tests."""
    n_samples = 100
    n_features = len(FEATURE_COLUMNS)
    rng = np.random.RandomState(42)
    
    # Generate synthetic features
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=FEATURE_COLUMNS
    )
    
    # Generate balanced labels in {-1, 0, 1}
    y = pd.Series(np.tile([-1, 0, 1], n_samples // 3 + 1)[:n_samples])
    
    return X, y


def test_train_only_mode(tmp_path, monkeypatch, sample_training_data):
    """Test the --train-only mode creates and saves a valid model."""
    X, y = sample_training_data
    model_path = tmp_path / "test_model.json"
    
    # Patch MODEL_SAVE_PATH to use tmp_path
    monkeypatch.setattr("src.model_manager.MODEL_SAVE_PATH", str(model_path))
    
    # Split data (small test for speed)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model, best_params = train_xgboost_model(X_train, y_train, X_test, y_test)
    
    # Verify model file exists
    assert model_path.exists(), "Model file was not created"
    
    # Verify we can load and use the model
    loaded_model = load_trained_model(str(model_path))
    assert hasattr(loaded_model, "predict_proba"), "Loaded model missing predict_proba"
    
    # Basic prediction shape check
    pred_proba = loaded_model.predict_proba(X_test)
    assert pred_proba.shape == (len(X_test), 3), "Wrong prediction shape"