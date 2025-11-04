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
logger.setLevel(logging.INFO)

__all__ = [
    "prepare_model_data",
    "train_xgboost_model",
    "load_trained_model",
    "make_predictions",
    "ModelRegistry",
]

# ---------------------------------------------------------------------------------
# Constants and Utilities
# ---------------------------------------------------------------------------------

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _next_model_version(model_dir: str, base_name: str = "xgboost_model") -> Tuple[str, int]:
    """Generate a filesystem-based next versioned model path."""
    os.makedirs(model_dir, exist_ok=True)
    versions = []
    for f in os.listdir(model_dir):
        if f.startswith(base_name) and "_v" in f and f.endswith(".json"):
            try:
                v = int(f.split("_v")[-1].split(".")[0])
                versions.append(v)
            except ValueError:
                continue
    next_v = max(versions) + 1 if versions else 1
    return os.path.join(model_dir, f"{base_name}_v{next_v}.json"), next_v


def _save_model_with_metadata(
    model: xgb.XGBClassifier,
    model_path: str,
    best_params: Dict[str, Any],
    accuracy: float,
    feature_set_id: str,
    feature_names: list,
) -> Tuple[str, Dict[str, Any]]:
    """Save the model file and accompanying metadata JSON."""
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

    meta_path = f"{model_path}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Model saved: {model_path}")
    logger.info(f"✅ Metadata saved: {meta_path}")
    return meta_path, metadata


# ---------------------------------------------------------------------------------
# Model Registry (Postgres)
# ---------------------------------------------------------------------------------

class ModelRegistry:
    """Postgres-backed model registry for version tracking and reproducibility."""

    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url

    def _connect(self):
        return psycopg2.connect(self.database_url)

    def next_version(self) -> int:
        """Return the next model version number."""
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COALESCE(MAX(version), 0) FROM model_registry;")
                return (cur.fetchone()[0] or 0) + 1
        finally:
            conn.close()

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
        """Insert or update a model record in the registry."""
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
                        notes = EXCLUDED.notes;
                    """,
                    (version, model_path, meta_path, accuracy, feature_set_id, Json(best_params), notes)
                )
            conn.commit()
            logger.info(f"📘 Registered model v{version} in model_registry.")
        finally:
            conn.close()

    def latest(self) -> Optional[Tuple[int, str, str]]:
        """Fetch the latest model entry."""
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT version, model_path, meta_path FROM model_registry ORDER BY version DESC LIMIT 1;")
                row = cur.fetchone()
                return (row[0], row[1], row[2]) if row else None
        finally:
            conn.close()

    def get(self, version: int) -> Optional[Tuple[int, str, str]]:
        """Fetch a specific version record."""
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
        """List all versions stored in the registry."""
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT version, model_path, meta_path, accuracy, train_date FROM model_registry ORDER BY version;"
                )
                return cur.fetchall()
        finally:
            conn.close()


# ---------------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------------

def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split, balance, and prepare training/test datasets."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    smote_k = max(1, min(5, y_train.value_counts().min() - 1))
    smote = SMOTE(k_neighbors=smote_k, random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    logger.info(f"Data prepared. Train shape: {X_train_res.shape}, Test shape: {X_test.shape}")
    return X_train_res, X_test, y_train_res, y_test


# ---------------------------------------------------------------------------------
# Training and Registration
# ---------------------------------------------------------------------------------

def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_set_id: str = "default_v1",
    use_registry: bool = True,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """Train an XGBoost model, version it, save, and register in Postgres."""
    try:
        y_train_enc = y_train.replace(LABEL_MAP)
        y_test_enc = y_test.replace(LABEL_MAP)

        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            n_jobs=-1,
        )

        search = RandomizedSearchCV(
            model,
            param_distributions={
                "n_estimators": [50, 100, 150],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
            },
            n_iter=3,
            cv=2,
            scoring="accuracy",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train_enc)

        best_model = search.best_estimator_
        best_params = search.best_params_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test_enc, y_pred)

        logger.info(f"✅ XGBoost trained | Accuracy: {acc:.4f}")
        logger.debug("Classification Report:\n%s", classification_report(y_test_enc, y_pred))

        # Save and register
        model_dir = os.path.dirname(MODEL_SAVE_PATH) or "models"
        versioned_path, _ = _next_model_version(model_dir)
        meta_path, metadata = _save_model_with_metadata(best_model, versioned_path, best_params, acc, feature_set_id, X_train.columns.tolist())

        if use_registry:
            registry = ModelRegistry()
            version = registry.next_version()
            registry.register(
                version, versioned_path, meta_path, acc, feature_set_id, best_params,
                notes=f"Trained on {len(X_train)} samples, tested on {len(X_test)}."
            )

        return best_model, best_params
    except Exception as e:
        logger.exception("❌ Error training XGBoost model: %s", e)
        raise


# ---------------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------------

def load_trained_model(version: Optional[int] = None, model_path: Optional[str] = None) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """Load model and metadata (latest or specific version)."""
    registry = ModelRegistry()
    resolved_path = model_path

    try:
        if not resolved_path:
            record = registry.get(version) if version else registry.latest()
            if record:
                _, resolved_path, _ = record
            else:
                resolved_path = MODEL_SAVE_PATH

        model = xgb.XGBClassifier()
        if not os.path.exists(resolved_path):
            logger.warning(f"No model found at {resolved_path}. Returning untrained model.")
            return model, {}

        model.load_model(resolved_path)

        meta = {}
        meta_path = f"{resolved_path}.meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)

        if "feature_names" in meta:
            model.feature_names_in_ = meta["feature_names"]

        logger.info(f"Loaded model from {resolved_path}")
        return model, meta
    except Exception as e:
        logger.exception("❌ Error loading model: %s", e)
        return xgb.XGBClassifier(), {}


# ---------------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------------

def make_predictions(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Tuple[pd.Series, pd.Series]:
    """Make predictions with confidence gating."""
    try:
        if X.empty:
            logger.warning("Prediction skipped: Empty DataFrame.")
            return pd.Series(dtype=int), pd.Series(dtype=float)

        features = getattr(model, "feature_names_in_", metadata.get("feature_names", []))
        X = X[features]

        proba = model.predict_proba(X)
        max_prob = np.max(proba, axis=1)
        preds = np.argmax(proba, axis=1)

        decoded = pd.Series([INV_LABEL_MAP[p] for p in preds], index=X.index, name="signal")
        decoded[max_prob < confidence_threshold] = 0  # gate low-confidence signals

        conf = pd.Series(max_prob, index=X.index, name="confidence")
        logger.info(f"Predictions done ({len(decoded)} rows). Threshold={confidence_threshold}")
        return decoded, conf
    except Exception as e:
        logger.exception("❌ Error making predictions: %s", e)
        return pd.Series(dtype=int), pd.Series(dtype=float)
