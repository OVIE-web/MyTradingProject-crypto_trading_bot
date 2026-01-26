from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from numpy.typing import NDArray
from pytz import UTC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

load_dotenv()

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------
# Types
# -----------------------------------------------------------------
TrainTestSplit = Tuple[
    NDArray,  # X_train
    NDArray,  # X_test
    NDArray,  # y_train
    NDArray,  # y_test
]

PredictionResult = Tuple[NDArray, NDArray]

# -----------------------------------------------------------------
# Config
# -----------------------------------------------------------------
MODEL_SAVE_PATH: str = os.getenv("MODEL_SAVE_PATH", "src/models/xgboost_model.json")
USE_MODEL_REGISTRY: bool = os.getenv("USE_MODEL_REGISTRY", "false").lower() == "true"


# -----------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------
def prepare_model_data(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    target_col: str = "target",
) -> TrainTestSplit:
    """
    Split DataFrame into NumPy train/test arrays.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found")

    features = feature_cols or [c for c in df.columns if c != target_col]

    X = df[features].to_numpy()
    y = df[target_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------
# Optional Model Registry
# -----------------------------------------------------------------
if USE_MODEL_REGISTRY:
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        logger.warning("psycopg2 not installed — disabling model registry")
        USE_MODEL_REGISTRY = False

if USE_MODEL_REGISTRY:

    class ModelRegistry:
        def __init__(self) -> None:
            self.conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                dbname=os.getenv("POSTGRES_DB", "trading"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            )
            self._create_table()

        def _create_table(self) -> None:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS model_registry (
                        id SERIAL PRIMARY KEY,
                        model_name TEXT NOT NULL,
                        model_path TEXT NOT NULL,
                        accuracy FLOAT,
                        params JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                    """
                )
                self.conn.commit()

        def register_model(
            self,
            model_name: str,
            model_path: str,
            accuracy: float,
            params: dict[str, Any],
        ) -> None:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_registry (model_name, model_path, accuracy, params)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (model_name, model_path, accuracy, json.dumps(params)),
                )
                self.conn.commit()

            logger.info("Registered model '%s' (accuracy=%.4f)", model_name, accuracy)

        def latest(self) -> Optional[dict[str, Any]]:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM model_registry ORDER BY id DESC LIMIT 1;")
                return cur.fetchone()


# -----------------------------------------------------------------
# Model Training
# -----------------------------------------------------------------
def train_xgboost_model(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    model_path: Optional[str] = None,
) -> Tuple[xgb.XGBClassifier, dict[str, Any]]:
    """
    Train an XGBoost classifier and persist it.
    """
    save_path = model_path or MODEL_SAVE_PATH

    model = xgb.XGBClassifier(
        eval_metric="mlogloss",
        random_state=42,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    logger.info("Model trained — accuracy=%.4f", accuracy)

    model.save_model(save_path)
    logger.info("Model saved to %s", save_path)

    metadata: dict[str, Any] = {
        "model_name": "xgboost_signal_model",
        "model_path": save_path,
        "accuracy": accuracy,
        "trained_at": datetime.now(UTC).isoformat(),
    }

    try:
        if USE_MODEL_REGISTRY:
            registry = ModelRegistry()
            registry.register_model(
                model_name=metadata["model_name"],
                model_path=metadata["model_path"],
                accuracy=float(accuracy),
                params=model.get_params(),
            )
        else:
            meta_path = save_path.replace(".json", "_metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            logger.info("Metadata saved locally to %s", meta_path)

    except Exception as exc:
        logger.exception("Failed to store model metadata: %s", exc)

    return model, metadata


# -----------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------
def load_trained_model(model_path: Optional[str] = None) -> Optional[xgb.XGBClassifier]:
    """
    Load trained model from disk or registry.
    """
    path = model_path or MODEL_SAVE_PATH

    try:
        if USE_MODEL_REGISTRY:
            registry = ModelRegistry()
            latest = registry.latest()
            if latest is not None:
                path = latest["model_path"]
                logger.info("Loading model from registry: %s", path)

        model = xgb.XGBClassifier()
        model.load_model(path)

        logger.info("Model loaded from %s", path)
        return model

    except Exception as exc:
        logger.exception("Failed to load model: %s", exc)
        return None


# -----------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------
def make_predictions(
    model: xgb.XGBClassifier,
    X: pd.DataFrame | NDArray,
    threshold: float = 0.5,
) -> PredictionResult:
    """
    Generate class predictions and confidence scores.

    Handles both DataFrame and NumPy array inputs. Automatically filters
    out datetime and object columns that XGBoost cannot process.

    Args:
        model: Trained XGBoost classifier
        X: Feature data (DataFrame or NDArray)
        threshold: Confidence threshold (reserved for future filtering)

    Returns:
        PredictionResult: Tuple of (predictions, confidence_scores)
            - predictions: Array of class labels [-1, 0, 1]
            - confidence_scores: Array of max probabilities per sample

    Raises:
        ValueError: If no numeric features or prediction fails
    """
    try:
        # Convert DataFrame to numeric-only array if needed
        if isinstance(X, pd.DataFrame):
            X_numeric = X.select_dtypes(include=["int64", "int32", "float64", "float32", "bool"])
            if X_numeric.empty:
                raise ValueError("No numeric features available for prediction")
            X_array = X_numeric.to_numpy()
        else:
            X_array = X

        # Get probability predictions
        proba = model.predict_proba(X_array)

        # Extract confidence (max probability across classes)
        confidence = np.max(proba, axis=1)

        # Get class predictions: argmax -> map to [-1, 0, 1] convention
        preds = np.argmax(proba, axis=1) - 1

        return preds, confidence

    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise ValueError(f"Prediction failed: {str(exc)}") from exc
