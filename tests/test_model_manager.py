import pytest
import pandas as pd
import numpy as np
import os
import logging
from unittest.mock import patch, MagicMock
from src.model_manager import (
    prepare_model_data,
    train_xgboost_model,
    make_predictions
)
from src.config import FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_STATE, CONFIDENCE_THRESHOLD

@pytest.fixture
def sample_model_data():
    """Generates sample data for model training/prediction."""
    num_samples = 100
    features = pd.DataFrame(
        np.random.rand(num_samples, len(FEATURE_COLUMNS)),
        columns=FEATURE_COLUMNS
    )
    target_values = np.random.choice([-1, 0, 1], size=num_samples, p=[0.1, 0.8, 0.1])
    features[TARGET_COLUMN] = target_values
    return features

def test_prepare_model_data(sample_model_data):
    X_train, X_test, y_train, y_test = prepare_model_data(
        sample_model_data,
        FEATURE_COLUMNS,
        TARGET_COLUMN
    )

    assert not X_train.empty and not X_test.empty
    assert not y_train.empty and not y_test.empty

    # Test set size should match expected proportion
    expected_test_size = int(len(sample_model_data) * 0.2)  # default TEST_SIZE=0.2
    assert len(X_test) == expected_test_size
    assert len(y_test) == expected_test_size

    # Training set should be >= original split size (because of SMOTE)
    assert len(X_train) >= len(sample_model_data) - expected_test_size

    # Class distributions between train/test should be roughly similar pre-SMOTE
    train_dist = y_train.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()
    np.testing.assert_allclose(train_dist, test_dist, atol=0.05)

def test_train_xgboost_model_saves_model(sample_model_data, tmp_path, caplog):
    with patch('sklearn.model_selection.RandomizedSearchCV') as mock_rscv:
    mock_estimator = MagicMock()
    mock_estimator.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]] * len(X_test))
    mock_estimator.feature_importances_ = np.ones(len(FEATURE_COLUMNS))
    mock_estimator.save_model = MagicMock()  # <-- add this

    mock_rscv_instance = MagicMock()
    mock_rscv_instance.best_estimator_ = mock_estimator
    mock_rscv_instance.best_params_ = {'mock_param': 'mock_value'}
    mock_rscv_instance.fit.return_value = None
    mock_rscv.return_value = mock_rscv_instance

            # Set a fake model save path
            original_path = os.environ.get('MODEL_SAVE_PATH')
            os.environ['MODEL_SAVE_PATH'] = str(tmp_path / "temp_model.json")

            try:
                with caplog.at_level(logging.INFO):
                    trained_model, best_params = train_xgboost_model(
                        X_train, y_train, X_test, y_test, RANDOM_STATE
                    )

                assert trained_model is not None
                assert 'mock_param' in best_params
                mock_save_model.assert_called_once_with(os.environ['MODEL_SAVE_PATH'])

                assert "Test accuracy" in caplog.text
                assert "Classification Report" in caplog.text
            finally:
                if original_path:
                    os.environ['MODEL_SAVE_PATH'] = original_path
                else:
                    del os.environ['MODEL_SAVE_PATH']

def test_make_predictions(mock_xgboost_model, sample_model_data):
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]]* len(X_data))
    predictions, confidence = make_predictions(mock_model, X_data, CONFIDENCE_THRESHOLD)

    assert not predictions.empty
    assert len(predictions) == len(X_data)
    assert predictions.isin([-1, 0, 1]).all()

    assert not confidence.empty
    assert (confidence >= 0).all() and (confidence <= 1).all()

    if CONFIDENCE_THRESHOLD < 0.8:
        assert (predictions == 0).all()  # All 'hold' if 0.8 is max_prob

    # Test empty input
    empty_df = pd.DataFrame(columns=FEATURE_COLUMNS)
    empty_preds, empty_conf = make_predictions(mock_xgboost_model, empty_df)
    assert empty_preds.empty
    assert empty_conf.empty