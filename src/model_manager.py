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
    MODEL_SAVE_PATH, DATABASE_URL, MODEL_METADATA_PATH

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

def prepare_model_data(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split, balance, and prepare training/test datasets safely.

    - Automatically detects feature and target columns if not provided.
    - Handles rare or single-class targets gracefully.
    - Dynamically adjusts SMOTE neighbors to avoid small-sample errors.
    - Provides detailed logging for debugging and traceability.
    """
    try:
        # --- Step 1: Determine features and target ---
        feature_cols = feature_cols or FEATURE_COLUMNS
        target_col = target_col or TARGET_COLUMN

        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
        if not all(col in df.columns for col in feature_cols):
            missing = [c for c in feature_cols if c not in df.columns]
            raise KeyError(f"Missing feature columns in DataFrame: {missing}")

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        if y.nunique() < 2:
            logger.warning(
                f"Target column '{target_col}' has only one unique class ({y.unique()}). "
                f"SMOTE and stratified split will be skipped."
            )
            return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        # --- Step 2: Split train/test ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        logger.info(f"✅ Data split successfully: Train {X_train.shape}, Test {X_test.shape}")
        logger.info("Class distribution before SMOTE (train):\n%s", y_train.value_counts())

        # --- Step 3: Handle rare classes for SMOTE ---
        min_class_count = y_train.value_counts().min()
        if min_class_count < 2:
            logger.warning("Too few samples for SMOTE (min class count=%d). Skipping resampling.", min_class_count)
            return X_train, X_test, y_train, y_test

        smote_k = max(1, min(5, min_class_count - 1))
        smote = SMOTE(k_neighbors=smote_k, random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        logger.info("✅ Class distribution after SMOTE:\n%s", y_train_res.value_counts())
        logger.info("Final train/test sizes: %s / %s", X_train_res.shape, X_test.shape)

        return X_train_res, X_test, y_train_res, y_test

    except Exception as e:
        logger.error("❌ Error preparing model data: %s", e, exc_info=True)
        raise

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
    model_path: Optional[str] = None,
    random_state: int = RANDOM_STATE,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Train an XGBoost model, version it, save artifacts, and optionally register in Postgres.
    Compatible with tests using old arguments.
    """
    try:
        # Encode labels
        y_train_enc = y_train.replace(LABEL_MAP)
        y_test_enc = y_test.replace(LABEL_MAP)

        model = xgb.XGBClassifier(
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
            model,
            param_distributions=param_grid,
            n_iter=3,
            cv=2,
            scoring="accuracy",
            random_state=random_state,
            n_jobs=-1,
        )
        search.fit(X_train, y_train_enc)

        best_model = search.best_estimator_
        best_params = search.best_params_

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test_enc, y_pred)

        logger.info(f"✅ XGBoost trained | Accuracy: {acc:.4f}")
        logger.debug("Classification Report:\n%s", classification_report(y_test_enc, y_pred))

        # Version control + saving
        model_dir = os.path.dirname(model_path or MODEL_SAVE_PATH) or "models"
        os.makedirs(model_dir, exist_ok=True)

        if use_registry:
            registry = ModelRegistry()
            version = registry.next_version()
            versioned_path = os.path.join(model_dir, f"xgboost_model_v{version}.json")
        else:
            version = 1
            versioned_path = model_path or MODEL_SAVE_PATH

        meta_path, metadata = _save_model_with_metadata(
            model=best_model,
            model_path=versioned_path,
            best_params=best_params,
            accuracy=acc,
            feature_set_id=feature_set_id,
            feature_names=X_train.columns.tolist(),
        )

        metadata["version"] = version
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if use_registry:
            registry.register(
                version=version,
                model_path=versioned_path,
                meta_path=meta_path,
                accuracy=acc,
                feature_set_id=feature_set_id,
                best_params=best_params,
                notes=f"Trained on {len(X_train)} samples, tested on {len(X_test)}."
            )

        logger.info(f"✅ Model v{version} saved at {versioned_path}")
        return best_model, metadata

    except Exception as e:
        logger.exception("❌ Error training XGBoost model: %s", e)
        raise




# ---------------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------------

def load_trained_model(
    version: Optional[int] = None,
    model_path: Optional[str] = None,
    return_metadata: bool = False,
) -> Tuple[xgb.XGBClassifier, Optional[Dict[str, Any]]]:
    """
    Load a trained model and its metadata safely.
    Supports:
      - Loading specific version from registry
      - Loading latest version if no version is given
      - Loading from a direct file path (manual override)

    Args:
        version: Optional model version number (from registry).
        model_path: Optional direct filesystem path override.
        return_metadata: If True, also returns metadata dict (default: False).

    Returns:
        model: Loaded XGBoost model (untrained if not found)
        metadata: (Optional) metadata dict if return_metadata=True
    """
    registry = ModelRegistry()
    resolved_path = model_path
    metadata: Dict[str, Any] = {}

    try:
        # 1️⃣ Resolve model path
        if not resolved_path:
            record = registry.get(version) if version else registry.latest()
            if record:
                _, resolved_path, _ = record
                logger.info(f"📦 Found model record in registry (version {version or 'latest'}).")
            else:
                logger.warning("⚠️ No record found in registry; using fallback MODEL_SAVE_PATH.")
                resolved_path = MODEL_SAVE_PATH

        # 2️⃣ Initialize model
        model = xgb.XGBClassifier()

        if not resolved_path or not os.path.exists(resolved_path):
            logger.warning(f"⚠️ No model found at {resolved_path}. Returning untrained model.")
            return (model, metadata) if return_metadata else (model,)

        # 3️⃣ Load model file
        model.load_model(resolved_path)

        # 4️⃣ Load metadata
        meta_path = f"{resolved_path}.meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            logger.info(f"🧾 Loaded model metadata from {meta_path}")
        else:
            logger.warning(f"⚠️ Metadata file not found for model: {meta_path}")

        # 5️⃣ Attach feature names if present
        if "feature_names" in metadata:
            model.feature_names_in_ = metadata["feature_names"]

        logger.info(f"✅ Loaded model successfully from: {resolved_path}")
        if return_metadata:
            return model, metadata
        else:
            return model, {}

    except Exception as e:
        logger.exception("❌ Error loading trained model: %s", e)
        return (xgb.XGBClassifier(), {}) if return_metadata else (xgb.XGBClassifier(),)



# ---------------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------------

def make_predictions(model, df, metadata=None):
    """
    Make predictions using a trained model and metadata.
    Handles missing metadata gracefully.
    """
    try:
        if model is None:
            raise ValueError("Model is None; cannot make predictions.")

        # Ensure metadata is a dict, even if None
        metadata = metadata or {}

        # Try to extract feature names safely
        feature_names = metadata.get("feature_names")
        if not feature_names:
            feature_names = getattr(model, "feature_names_in_", df.columns.tolist())

        features = df[feature_names].copy()
        preds = model.predict(features)
        decoded_preds = pd.Series(np.sign(preds), index=df.index)
        confidence = pd.Series(np.abs(preds), index=df.index)

        logger.info("✅ Predictions generated successfully.")
        return decoded_preds, confidence

    except Exception as e:
        logger.error(f"❌ Error making predictions: {e}", exc_info=True)
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
def get_latest_model_metadata() -> Dict[str, Any]:
    """Backward-compatible helper for tests expecting this function."""
    registry = ModelRegistry()
    record = registry.latest()
    if not record:
        return {}
    _, model_path, _ = record
    meta_path = f"{model_path}.meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}
