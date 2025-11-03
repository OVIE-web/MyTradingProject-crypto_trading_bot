import os
import pytest
import numpy as np
import pandas as pd
import logging
from unittest.mock import patch, MagicMock

from src.model_manager import (
    prepare_model_data,
    train_xgboost_model,
    make_predictions,
)
from src.config import FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_STATE, CONFIDENCE_THRESHOLD


# -------------------------------------------------
# 🎯 FIXTURE: Create mock dataset
# -------------------------------------------------
@pytest.fixture
def sample_model_data():
    """Create a synthetic DataFrame with realistic trading features."""
    np.random.seed(42)
    df = pd.DataFrame({
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
        "signal": np.random.choice([-1, 0, 1], 100, p=[0.3, 0.4, 0.3]),
    })
    return df


# -------------------------------------------------
# ✅ TEST: prepare_model_data
# -------------------------------------------------
def test_prepare_model_data(sample_model_data):
    """Ensure SMOTE balancing and stratified split works."""
    X_train, X_test, y_train, y_test = prepare_model_data(
        sample_model_data,
        FEATURE_COLUMNS,
        TARGET_COLUMN
    )

    # Shape assertions
    assert not X_train.empty and not X_test.empty
    assert not y_train.empty and not y_test.empty

    expected_test_size = int(len(sample_model_data) * 0.2)
    assert len(X_test) == expected_test_size
    assert len(y_test) == expected_test_size

    # SMOTE increases training samples
    assert len(X_train) >= len(sample_model_data) - expected_test_size

    # Ensure 3 classes exist after resampling
    assert set(y_train.unique()) == {-1, 0, 1}

    # Distribution check (soft, because SMOTE balances)
    assert abs(y_train.value_counts(normalize=True).std()) < 0.05


# -------------------------------------------------
# TEST: xgboost training and saving
# -------------------------------------------------
@pytest.mark.filterwarnings("ignore:Precision and F-score are ill-defined")
def test_train_xgboost_model_saves_model(sample_model_data, tmp_path, caplog):
    """Test that training the XGBoost model saves correctly and returns best params."""
    with patch("xgboost.XGBClassifier.save_model") as mock_save_model:
        X_train, X_test, y_train, y_test = prepare_model_data(
            sample_model_data, FEATURE_COLUMNS, TARGET_COLUMN
        )

        # ✅ Patch RandomizedSearchCV inside model_manager
        with patch("src.model_manager.RandomizedSearchCV") as mock_rscv:
            # Mock model (simulate fitted XGBoost model)
            mock_model = MagicMock()
            mock_model._get_tags.return_value = {"pairwise": False}
            # <-- critical fix: return a real array of labels
            mock_model.predict.return_value = np.zeros(len(X_test), dtype=int)
            mock_model.predict_proba.return_value = np.array(
                [[0.1, 0.8, 0.1]] * len(X_test)
            )
            mock_model.feature_importances_ = np.ones(len(FEATURE_COLUMNS))
            mock_model.save_model = MagicMock()

            # Mock RandomizedSearchCV behavior
            mock_rscv_instance = MagicMock()
            mock_rscv_instance.best_estimator_ = mock_model
            mock_rscv_instance.best_params_ = {"mock_param": "mock_value"}
            mock_rscv_instance.fit.return_value = None
            mock_rscv.return_value = mock_rscv_instance

            # Use a fake save path
            model_path = tmp_path / "temp_model.json"

            with caplog.at_level(logging.INFO):
                trained_model, best_params = train_xgboost_model(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    random_state=RANDOM_STATE,
                    model_path=str(model_path),
                )

            # ✅ Assertions
            assert trained_model is mock_model
            assert best_params == {"mock_param": "mock_value"}
            mock_model.save_model.assert_called_once()
            assert "Model saved" in caplog.text



# -------------------------------------------------
# ✅ TEST: make_predictions
# -------------------------------------------------
def test_make_predictions(sample_model_data):
    """Test prediction logic and empty input behavior."""
    mock_model = MagicMock()

    # Define X_data for predictions
    X_data = sample_model_data.drop(columns=[TARGET_COLUMN])

    # 3-class probability output (softmax-like)
    mock_model.predict.return_value = np.array([0, 1, 2] * (len(X_test) // 3))

    predictions, confidence = make_predictions(mock_model, X_data, CONFIDENCE_THRESHOLD)

    # Output validity
    assert not predictions.empty
    assert len(predictions) == len(X_data)
    assert set(predictions.unique()).issubset({-1, 0, 1})
    assert confidence.between(0, 1).all()

    # If threshold < 0.8, all signals are 'hold' (0)
    if CONFIDENCE_THRESHOLD < 0.8:
        assert (predictions == 0).all()

    # Empty input must return empty Series
    empty_df = pd.DataFrame(columns=FEATURE_COLUMNS)
    empty_preds, empty_conf = make_predictions(mock_model, empty_df)
    assert empty_preds.empty
    assert empty_conf.empty
