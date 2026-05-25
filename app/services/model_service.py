"""Model training, evaluation, and prediction services for the trading signal application."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from app.core.config import (
    CONFIDENCE_THRESHOLD,
    MODEL_METADATA_PATH,
    MODEL_SAVE_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from app.services.mlflow_tracking import log_training_run
from app.services.model_registry import create_registry

LOG = logging.getLogger(__name__)

TrainTestSplit = tuple[
    NDArray[Any],
    NDArray[Any],
    NDArray[Any],
    NDArray[Any],
]

PredictionResult = tuple[NDArray[Any], NDArray[Any]]

USE_MODEL_REGISTRY: bool = os.getenv("USE_MODEL_REGISTRY", "false").lower() == "true"


# -----------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------
def prepare_model_data(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> TrainTestSplit:
    """
    Split a DataFrame into train/test arrays for model training.
    """
    if df.empty:
        raise ValueError("Cannot prepare model data from an empty DataFrame.")

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    features = feature_cols or [col for col in df.columns if col != target_col]

    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")

    X = df[features].to_numpy()
    y = df[target_col].to_numpy()

    return cast(
        TrainTestSplit,
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        ),
    )


# -----------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------
def get_classification_report(
    y_true: NDArray[Any],
    y_pred: NDArray[Any],
    target_names: list[str] | None = None,
    as_dict: bool = False,
) -> str | dict[str, Any]:
    """
    Generate a classification report.
    """
    if target_names is None:
        target_names = [str(label) for label in sorted(np.unique(y_true))]

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=as_dict,
        zero_division=0,
    )

    LOG.info("Classification report generated.")
    return cast(str | dict[str, Any], report)


def get_confusion_matrix_report(
    y_true: NDArray[Any],
    y_pred: NDArray[Any],
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """
    Generate a confusion matrix and per-class statistics.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report: dict[str, Any] = {
        "confusion_matrix": cm.tolist(),
        "matrix_shape": cm.shape,
    }

    n_classes = cm.shape[0]

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        report[f"class_{i}"] = {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

    LOG.info("Confusion matrix report generated.")
    return report


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: NDArray[Any],
    y_test: NDArray[Any],
) -> dict[str, Any]:
    """
    Evaluate a trained XGBoost classifier.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    evaluation = {
        "accuracy": float(accuracy),
        "classification_report": get_classification_report(
            y_test,
            y_pred,
            as_dict=True,
        ),
        "confusion_matrix_report": get_confusion_matrix_report(y_test, y_pred),
        "predictions": y_pred.tolist(),
        "probabilities": y_pred_proba.tolist(),
    }

    LOG.info("Model evaluation complete. accuracy=%.4f", accuracy)
    return evaluation


# -----------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------
def save_model_metadata(metadata: dict[str, Any], metadata_path: str = MODEL_METADATA_PATH) -> None:
    """
    Save model metadata locally as JSON.
    """
    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, default=str)

    LOG.info("Model metadata saved to %s", path)


def load_model_metadata(metadata_path: str = MODEL_METADATA_PATH) -> dict[str, Any] | None:
    """
    Load model metadata from disk if available.
    """
    path = Path(metadata_path)

    if not path.exists():
        LOG.warning("Model metadata file not found: %s", path)
        return None

    with path.open("r", encoding="utf-8") as file:
        return cast(dict[str, Any], json.load(file))


# -----------------------------------------------------------------
# Training
# -----------------------------------------------------------------
def train_xgboost_model(
    X_train: NDArray[Any],
    y_train: NDArray[Any],
    X_test: NDArray[Any],
    y_test: NDArray[Any],
    model_path: str | None = None,
) -> tuple[xgb.XGBClassifier, dict[str, Any]]:
    """
    Train an XGBoost classifier, save it, and store metadata.
    """
    save_path = model_path or MODEL_SAVE_PATH
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    model = xgb.XGBClassifier(
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    evaluation = evaluate_model(model, X_test, y_test)

    model.save_model(str(save_path_obj))
    LOG.info("Model saved to %s", save_path_obj)

    metadata: dict[str, Any] = {
        "model_name": "xgboost_signal_model",
        "model_type": "xgboost",
        "model_path": str(save_path_obj),
        "accuracy": float(accuracy),
        "trained_at": datetime.now(UTC).isoformat(),
        "evaluation": evaluation,
        "params": model.get_params(),
    }
    metadata.update(
        log_training_run(
            model=model,
            metadata=metadata,
            model_path=save_path_obj,
        )
    )

    try:
        if USE_MODEL_REGISTRY:
            with create_registry() as registry:
                registry.register_model(
                    model_name=metadata["model_name"],
                    model_path=metadata["model_path"],
                    accuracy=float(accuracy),
                    params=model.get_params(),
                )
        else:
            save_model_metadata(metadata)

    except Exception as exc:
        LOG.exception("Failed to store model metadata: %s", exc)

    return model, metadata


# -----------------------------------------------------------------
# Loading
# -----------------------------------------------------------------
def load_trained_model(
    model_path: str | None = None,
) -> xgb.XGBClassifier | None:
    """
    Load a trained XGBoost model from disk or registry.
    """
    path = model_path or MODEL_SAVE_PATH

    try:
        if USE_MODEL_REGISTRY:
            with create_registry() as registry:
                latest = registry.get_model("xgboost_signal_model")
                if latest is not None:
                    path = latest["model_path"]
                    LOG.info("Loading model from registry: %s", path)

        path_obj = Path(path)

        if not path_obj.exists():
            LOG.error("Model file not found: %s", path_obj)
            return None

        model = xgb.XGBClassifier()
        model.load_model(str(path_obj))

        LOG.info("Model loaded from %s", path_obj)
        return model

    except Exception as exc:
        LOG.exception("Failed to load model: %s", exc)
        return None


# -----------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------
def make_predictions(
    model: xgb.XGBClassifier,
    X: pd.DataFrame | NDArray[Any],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> PredictionResult:
    """
    Generate class predictions and confidence scores.

    Prediction convention:
        0 from XGBoost -> -1 SELL
        1 from XGBoost ->  0 HOLD
        2 from XGBoost ->  1 BUY
    """
    if model is None:
        raise ValueError("Model is not loaded.")

    try:
        if isinstance(X, pd.DataFrame):
            X_numeric = X.select_dtypes(include=["int64", "int32", "float64", "float32", "bool"])

            if X_numeric.empty:
                raise ValueError("No numeric features available for prediction.")

            X_array = X_numeric.to_numpy()
        else:
            X_array = X

        proba = model.predict_proba(X_array)

        confidence = np.max(proba, axis=1)
        preds = np.argmax(proba, axis=1) - 1

        # Optional safety threshold:
        # Low-confidence predictions become HOLD.
        preds = np.where(confidence >= threshold, preds, 0)

        return preds, confidence

    except ValueError:
        raise
    except Exception as exc:
        LOG.exception("Prediction failed: %s", exc)
        raise ValueError(f"Prediction failed: {exc}") from exc
