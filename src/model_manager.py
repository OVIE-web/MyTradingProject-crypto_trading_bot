import os
import json
import logging
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import psycopg2
from psycopg2.extras import Json

from src.config import (
    FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, CONFIDENCE_THRESHOLD,
    MODEL_SAVE_PATH, DATABASE_URL
)

logger = logging.getLogger(__name__)

__all__ = [
    "prepare_model_data",
    "train_xgboost_model",
    "load_trained_model",
    "make_predictions",
    "ModelRegistry",
]

# ---------------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------------

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}


def _ensure_dir(path: str) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


def _save_model_with_metadata(
    model: xgb.XGBClassifier,
    model_path: str,
    best_params: Dict[str, Any],
    accuracy: float,
    feature_set_id: str,
    feature_names: list,
) -> Tuple[str, Dict[str, Any]]:
    """
    Save model to `model_path` and metadata to `model_path + .meta.json`.
    Returns (meta_path, metadata dict).
    """
    _ensure_dir(model_path)
    model.save_model(model_path)

    metadata = {
        "feature_names": list(feature_names),
        "label_map": LABEL_MAP,
        "best_params": best_params,
        "accuracy": accuracy,
        "feature_set_id": feature_set_id,
        "train_date": datetime.utcnow().isoformat(),
        "model_filename": os.path.basename(model_path),
    }
    meta_path = model_path + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model and metadata saved: {model_path}, {meta_path}")
    return meta_path, metadata


def _next_model_version(model_dir: str, base_name: str = "xgboost_model") -> Tuple[str, int]:
    """
    Filesystem-based versioning: returns next available path and version.
    e.g., model_dir/xgboost_model_v3.json, 3
    """
    os.makedirs(model_dir, exist_ok=True)
    existing = [
        f for f in os.listdir(model_dir)
        if f.startswith(base_name) and f.endswith(".json") and "_v" in f
    ]
    versions = []
    for f in existing:
        try:
            v = int(f.split("_v")[-1].split(".")[0])
            versions.append(v)
        except ValueError:
            continue
    next_v = max(versions) + 1 if versions else 1
    return os.path.join(model_dir, f"{base_name}_v{next_v}.json"), next_v


# ---------------------------------------------------------------------------------
# Model Registry (Postgres)
# ---------------------------------------------------------------------------------

class ModelRegistry:
    """
    Postgres-backed model registry for versioning and provenance.

    Schema suggestion:
        CREATE TABLE model_registry (
            id SERIAL PRIMARY KEY,
            version INT NOT NULL UNIQUE,
            model_path TEXT NOT NULL,
            meta_path TEXT NOT NULL,
            accuracy FLOAT,
            feature_set_id TEXT,
            train_date TIMESTAMP DEFAULT NOW(),
            best_params JSONB,
            notes TEXT
        );
    """

    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url

    def _connect(self):
        return psycopg2.connect(self.database_url)

    def register(
        self,
        version: int,
        model_path: str,
        meta_path: str,
        accuracy: float,
        feature_set_id: str,
        best_params: Dict[str, Any],
        notes: Optional[str] = None,
    ) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_registry (version, model_path, meta_path, accuracy, feature_set_id, best_params, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (version) DO UPDATE
                    SET accuracy = EXCLUDED.accuracy,
                        model_path = EXCLUDED.model_path,
                        meta_path = EXCLUDED.meta_path,
                        feature_set_id = EXCLUDED.feature_set_id,
                        best_params = EXCLUDED.best_params,
                        notes = EXCLUDED.notes
                    """,
                    (version, model_path, meta_path, accuracy, feature_set_id, Json(best_params), notes)
                )
            conn.commit()
            logger.info(f"Registered model v{version} in DB: {model_path}")
        finally:
            conn.close()

    def latest(self) -> Optional[Tuple[int, str, str]]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT version, model_path, meta_path FROM model_registry ORDER BY version DESC LIMIT 1;"
                )
                row = cur.fetchone()
                return (row[0], row[1], row[2]) if row else None
        finally:
            conn.close()

    def get(self, version: int) -> Optional[Tuple[int, str, str]]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT version, model_path, meta_path FROM model_registry WHERE version = %s LIMIT 1;",
                    (version,)
                )
                row = cur.fetchone()
                return (row[0], row[1], row[2]) if row else None
        finally:
            conn.close()

    def list_all(self) -> list:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT version, model_path, meta_path, accuracy, train_date FROM model_registry ORDER BY version;")
                return cur.fetchall()
        finally:
            conn.close()


# ---------------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------------

def prepare_model_data(
    df: pd.DataFrame,
    feature_cols: list = FEATURE_COLUMNS,
    target_col: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train/test and balance classes with SMOTE.
    """
    try:
        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        logger.info(f"Data split successfully. Training: {X_train.shape}, Testing: {X_test.shape}")
        logger.info("Class distribution before SMOTE (Train):\n%s", y_train.value_counts())

        smote_k = max(1, min(5, y_train.value_counts().min() - 1))
        smote = SMOTE(k_neighbors=smote_k, random_state=RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        logger.info("Class distribution after SMOTE (Train):\n%s", y_train_balanced.value_counts())
        return X_train_balanced, X_test, y_train_balanced, y_test
    except Exception as e:
        logger.error("Error preparing model data: %s", e, exc_info=True)
        raise


# ---------------------------------------------------------------------------------
# Training (with filesystem versioning + DB registration)
# ---------------------------------------------------------------------------------

def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = RANDOM_STATE,
    model_path: Optional[str] = None,
    feature_set_id: str = "default_v1",
    use_registry: bool = True,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Train an XGBoost model, save versioned artifacts and optionally register in DB.
    """
    try:
        # Encode labels
        y_train_encoded = y_train.replace(LABEL_MAP)
        y_test_encoded = y_test.replace(LABEL_MAP)

        base_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            random_state=random_state,
            eval_metric="mlogloss",
            n_jobs=-1,
        )

        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
        }

        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=3,
            cv=2,
            scoring="accuracy",
            random_state=random_state,
            verbose=0,
            n_jobs=-1,
        )

        search.fit(X_train, y_train_encoded)
        best_model = search.best_estimator_
        best_params = search.best_params_

        # Eval
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test_encoded, y_pred)
        report = classification_report(y_test_encoded, y_pred)
        logger.info(f"✅ Test accuracy: {acc:.4f}")
        logger.info("Classification Report:\n%s", report)

        # Persist feature names for alignment
        best_model.feature_names_in_ = X_train.columns.tolist()

        # Versioned path
        resolved_path = model_path or MODEL_SAVE_PATH
        model_dir = os.path.dirname(resolved_path) or os.path.dirname(MODEL_SAVE_PATH)
        base_name = "xgboost_model"
        versioned_model_path, version = _next_model_version(model_dir, base_name=base_name)

        # Save model + metadata
        meta_path, metadata = _save_model_with_metadata(
            model=best_model,
            model_path=versioned_model_path,
            best_params=best_params,
            accuracy=acc,
            feature_set_id=feature_set_id,
            feature_names=best_model.feature_names_in_,
        )

        # Optional DB registration
        if use_registry:
            registry = ModelRegistry()
            registry.register(
                version=version,
                model_path=versioned_model_path,
                meta_path=meta_path,
                accuracy=acc,
                feature_set_id=feature_set_id,
                best_params=best_params,
                notes=f"Trained on {len(X_train)} samples; eval on {len(X_test)} samples."
            )

        return best_model, best_params

    except Exception as e:
        logger.error("Error training XGBoost model: %s", e, exc_info=True)
        raise


# ---------------------------------------------------------------------------------
# Loading (latest or specific version)
# ---------------------------------------------------------------------------------

def load_trained_model(
    version: Optional[int] = None,
    model_path: Optional[str] = None,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Load a trained model and its metadata.
    - If `version` provided, load that version from registry (if available) or filesystem.
    - Else if `model_path` provided, load from that path.
    - Else load latest from registry, falling back to MODEL_SAVE_PATH (non-versioned).
    """
    try:
        registry = ModelRegistry()
        resolved_model_path = None

        # Priority: explicit model_path > version from registry > latest from registry > fallback MODEL_SAVE_PATH
        if model_path:
            resolved_model_path = model_path
        elif version is not None:
            record = registry.get(version)
            if record:
                _, resolved_model_path, _ = record
        else:
            record = registry.latest()
            if record:
                _, resolved_model_path, _ = record

        if not resolved_model_path:
            resolved_model_path = MODEL_SAVE_PATH  # fallback (non-versioned)

        model = xgb.XGBClassifier()
        if not os.path.exists(resolved_model_path):
            logger.warning(f"No trained model found at {resolved_model_path}, returning untrained model")
            return model, {}

        model.load_model(resolved_model_path)

        meta_path = resolved_model_path + ".meta.json"
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        # If metadata contains feature_names, attach for alignment
        if "feature_names" in metadata:
            model.feature_names_in_ = metadata["feature_names"]

        logger.info(f"Loaded model: {resolved_model_path}")
        return model, metadata

    except Exception as e:
        logger.error("Error loading trained model: %s", e, exc_info=True)
        return xgb.XGBClassifier(), {}


# ---------------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------------

def make_predictions(
    model: xgb.XGBClassifier,
    X_data: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate predictions with confidence gating and feature alignment.
    Returns (decoded_preds: Series in {-1,0,1}, confidence: Series in [0,1]).
    """
    try:
        if X_data.empty:
            logger.warning("Empty input received for prediction.")
            return pd.Series(dtype=int), pd.Series(dtype=float)

        # Align features either via model.feature_names_in_ or provided metadata
        feature_names = None
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        elif metadata and "feature_names" in metadata:
            feature_names = list(metadata["feature_names"])

        if feature_names:
            missing = set(feature_names) - set(X_data.columns)
            if missing:
                raise ValueError(f"Missing required features for prediction: {missing}")
            X_data = X_data[feature_names]

        proba = model.predict_proba(X_data)
        max_prob = np.max(proba, axis=1)
        preds = np.argmax(proba, axis=1)

        decoded_preds = pd.Series([INV_LABEL_MAP[p] for p in preds], index=X_data.index, name="signal")

        # Confidence gating -> hold (0) if below threshold
        decoded_preds[max_prob < confidence_threshold] = 0
        confidence = pd.Series(max_prob, index=X_data.index, name="confidence")

        logger.info(f"Predictions generated with confidence threshold {confidence_threshold}.")
        logger.info("Prediction distribution:\n%s", decoded_preds.value_counts())

        return decoded_preds, confidence

    except Exception as e:
        logger.error("Error making predictions: %s", e, exc_info=True)
        return pd.Series(dtype=int), pd.Series(dtype=float)