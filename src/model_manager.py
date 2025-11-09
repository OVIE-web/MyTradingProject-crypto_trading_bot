import os
import json
import logging
from pytz import timezone
from datetime import datetime
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from dotenv import load_dotenv
from typing import List, Optional
import pandas as pd
from typing import Tuple


load_dotenv()

logger = logging.getLogger(__name__)

def prepare_model_data(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    target_col: Optional[str] = "target",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compatibility helper for tests — split DataFrame into train/test sets.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")

    X = df[feature_cols or [c for c in df.columns if c != target_col]]
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)
# -----------------------------------------------------------------
# ✅ Config
# -----------------------------------------------------------------
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "src/models/xgboost_model.json")
USE_MODEL_REGISTRY = os.getenv("USE_MODEL_REGISTRY", "False").lower() == "true"

# -----------------------------------------------------------------
# ✅ Optional Model Registry (activated by env variable)
# -----------------------------------------------------------------
if USE_MODEL_REGISTRY:
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        logger.warning("⚠️ psycopg2 not installed — ModelRegistry will be disabled.")
        USE_MODEL_REGISTRY = False

if USE_MODEL_REGISTRY:
    class ModelRegistry:
        def __init__(self):
            self.conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                dbname=os.getenv("POSTGRES_DB", "trading"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            )
            self.create_table()

        def create_table(self):
            with self.conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_registry (
                        id SERIAL PRIMARY KEY,
                        model_name TEXT NOT NULL,
                        model_path TEXT NOT NULL,
                        accuracy FLOAT,
                        params JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                self.conn.commit()

        def register_model(self, model_name, model_path, accuracy, params):
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO model_registry (model_name, model_path, accuracy, params)
                    VALUES (%s, %s, %s, %s)
                """, (model_name, model_path, accuracy, json.dumps(params)))
                self.conn.commit()
            logger.info(f"🧾 Registered model '{model_name}' with accuracy={accuracy:.4f}")

        def latest(self):
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM model_registry ORDER BY id DESC LIMIT 1;")
                record = cur.fetchone()
                return record if record else None


# -----------------------------------------------------------------
# ✅ Core Model Functions
# -----------------------------------------------------------------
def train_xgboost_model(X_train, y_train, X_test, y_test, model_path: Optional[str] = None):
    """
    Train an XGBoost model and save it (with registry or JSON metadata).
    """
    model_path = model_path or MODEL_SAVE_PATH

    model = xgb.XGBClassifier(
        eval_metric="mlogloss",
        random_state=42,
        use_label_encoder=False
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    logger.info(f"✅ Model trained — Accuracy: {accuracy:.4f}")

    model.save_model(model_path)
    logger.info(f"📁 Model saved to {model_path}")

    # ---------------------------------------------------------
    # Save metadata (Registry or local JSON)
    # ---------------------------------------------------------
    metadata = {
        "model_name": "xgboost_signal_model",
        "model_path": model_path,
        "accuracy": accuracy,
        "trained_at": datetime.datetime.now(timezone('UTC')).isoformat(),
    }

    try:
        if USE_MODEL_REGISTRY:
            registry = ModelRegistry()
            registry.register_model(
                model_name=metadata["model_name"],
                model_path=metadata["model_path"],
                accuracy=metadata["accuracy"],
                params=model.get_params(),
            )
        else:
            meta_path = model_path.replace(".json", "_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"📝 Model metadata saved locally at {meta_path}")

    except Exception as e:
        logger.error(f"⚠️ Failed to record model metadata: {e}", exc_info=True)

    return model, metadata


def load_trained_model(model_path: Optional[str] = None):
    """
    Load a previously trained XGBoost model from file or latest registry entry.
    """
    model_path = model_path or MODEL_SAVE_PATH

    try:
        if USE_MODEL_REGISTRY:
            registry = ModelRegistry()
            latest = registry.latest()
            if latest:
                model_path = latest["model_path"]
                logger.info(f"📦 Loading latest model from registry: {model_path}")

        model = xgb.XGBClassifier()
        model.load_model(model_path)
        logger.info(f"✅ Model loaded successfully from {model_path}")
        return model

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}", exc_info=True)
        return None


def make_predictions(model, X, threshold: float = 0.5):
    """
    Generate predictions and confidence from trained model.
    """
    try:
        proba = model.predict_proba(X)
        confidence = np.max(proba, axis=1)
        preds = np.argmax(proba, axis=1) - 1  # assuming labels [-1, 0, 1]
        return preds, confidence
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}", exc_info=True)
        return np.zeros(len(X)), np.zeros(len(X))