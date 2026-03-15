"""
Unit tests for model training and evaluation metrics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import model_manager as mm
from src.config import FEATURE_COLUMNS, RANDOM_STATE, TEST_SIZE


def test_train_and_save_load(tmp_path: Path) -> None:
    """Train a tiny XGBoost model, ensure it is saved and can be loaded."""
    model_path = tmp_path / "xgboost_model.json"

    # Build a small but reasonable synthetic dataset
    n_samples = 150
    n_features = len(FEATURE_COLUMNS)
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(n_samples, n_features), columns=FEATURE_COLUMNS)
    # Create balanced labels in {-1,0,1} and map to {0,1,2} for model compatibility
    y_raw = np.tile([-1, 0, 1], int(np.ceil(n_samples / 3)))[:n_samples]
    label_map = {-1: 0, 0: 1, 1: 2}
    y = pd.Series([label_map[val] for val in y_raw])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ✅ Train model with explicit model_path
    model, metadata = mm.train_xgboost_model(
        X_train, y_train, X_test, y_test, model_path=str(model_path)
    )

    # Confirm file saved
    assert model_path.exists(), f"Model file was not created at {model_path}"

    # Confirm it can be loaded
    loaded = mm.load_trained_model(str(model_path))
    assert hasattr(loaded, "predict_proba")

    # ✅ Verify metadata contains evaluation metrics
    assert "accuracy" in metadata
    assert "evaluation" in metadata
    assert metadata["accuracy"] >= 0.0 and metadata["accuracy"] <= 1.0


def test_classification_report(tmp_path: Path) -> None:
    """Test classification report generation."""
    # Build dataset
    n_samples = 150
    n_features = len(FEATURE_COLUMNS)
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(n_samples, n_features), columns=FEATURE_COLUMNS)
    y_raw = np.tile([-1, 0, 1], int(np.ceil(n_samples / 3)))[:n_samples]
    label_map = {-1: 0, 0: 1, 1: 2}
    y = pd.Series([label_map[val] for val in y_raw])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model_path = tmp_path / "xgboost_model.json"
    model, metadata = mm.train_xgboost_model(
        X_train, y_train, X_test, y_test, model_path=str(model_path)
    )

    # Get predictions
    y_pred = model.predict(X_test)

    # ✅ Test classification report (string format)
    report_str = mm.get_classification_report(
        y_test.to_numpy(), y_pred, target_names=["SELL", "HOLD", "BUY"]
    )
    assert isinstance(report_str, str)
    assert "precision" in report_str.lower()
    assert "recall" in report_str.lower()
    assert "f1-score" in report_str.lower()

    # ✅ Test classification report (dict format)
    report_dict = mm.get_classification_report(y_test.to_numpy(), y_pred, as_dict=True)
    assert isinstance(report_dict, dict)
    assert "0" in report_dict or "0" in str(report_dict)
    assert "accuracy" in report_dict


def test_confusion_matrix(tmp_path: Path) -> None:
    """Test confusion matrix generation and analysis."""
    # Build dataset
    n_samples = 150
    n_features = len(FEATURE_COLUMNS)
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(n_samples, n_features), columns=FEATURE_COLUMNS)
    y_raw = np.tile([-1, 0, 1], int(np.ceil(n_samples / 3)))[:n_samples]
    label_map = {-1: 0, 0: 1, 1: 2}
    y = pd.Series([label_map[val] for val in y_raw])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model_path = tmp_path / "xgboost_model.json"
    model, metadata = mm.train_xgboost_model(
        X_train, y_train, X_test, y_test, model_path=str(model_path)
    )

    # Get predictions
    y_pred = model.predict(X_test)

    # ✅ Test confusion matrix report
    cm_report = mm.get_confusion_matrix_report(y_test.to_numpy(), y_pred)

    assert isinstance(cm_report, dict)
    assert "confusion_matrix" in cm_report
    assert "matrix_shape" in cm_report

    # Verify confusion matrix structure
    cm = np.array(cm_report["confusion_matrix"])
    assert cm.shape == (3, 3), f"Expected (3, 3) confusion matrix, got {cm.shape}"

    # Verify per-class metrics exist
    for i in range(3):
        assert f"class_{i}" in cm_report
        class_metrics = cm_report[f"class_{i}"]
        assert "precision" in class_metrics
        assert "recall" in class_metrics
        assert "f1_score" in class_metrics
        assert "true_positives" in class_metrics
        assert "false_positives" in class_metrics


def test_evaluate_model(tmp_path: Path) -> None:
    """Test comprehensive model evaluation."""
    # Build dataset
    n_samples = 150
    n_features = len(FEATURE_COLUMNS)
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(n_samples, n_features), columns=FEATURE_COLUMNS)
    y_raw = np.tile([-1, 0, 1], int(np.ceil(n_samples / 3)))[:n_samples]
    label_map = {-1: 0, 0: 1, 1: 2}
    y = pd.Series([label_map[val] for val in y_raw])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model_path = tmp_path / "xgboost_model.json"
    model, _ = mm.train_xgboost_model(X_train, y_train, X_test, y_test, model_path=str(model_path))

    # ✅ Test comprehensive evaluation
    evaluation = mm.evaluate_model(model, X_test.to_numpy(), y_test.to_numpy())

    assert isinstance(evaluation, dict)
    assert "accuracy" in evaluation
    assert "classification_report" in evaluation
    assert "confusion_matrix_report" in evaluation
    assert "predictions" in evaluation
    assert "probabilities" in evaluation

    # Verify evaluation metrics are valid
    assert 0.0 <= evaluation["accuracy"] <= 1.0
    assert len(evaluation["predictions"]) == len(X_test)
    assert len(evaluation["probabilities"]) == len(X_test)
