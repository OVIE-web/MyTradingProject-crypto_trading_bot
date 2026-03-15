# src/train_model.py
"""Standalone script to train the XGBoost model."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_COLUMNS,
    MODEL_SAVE_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.model_manager import train_xgboost_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 500) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic training data for testing."""
    logger.info("📊 Generating synthetic training data...")
    X = pd.DataFrame(
        np.random.randn(n_samples, len(FEATURE_COLUMNS)),
        columns=FEATURE_COLUMNS,
    )
    y = pd.Series(np.random.choice([0, 1, 2], n_samples))
    logger.info("Dataset shape: %s", X.shape)
    logger.info("Class distribution:\n%s", y.value_counts())
    return X, y


def load_real_data(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate a CSV data file for training."""
    logger.info("📂 Loading real data from %s...", data_path)
    df = pd.read_csv(data_path)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in data file: {missing}")

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found. Available columns: {list(df.columns)}"
        )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    logger.info("✅ Real data loaded | Shape: %s", X.shape)
    logger.info("Class distribution:\n%s", y.value_counts())
    return X, y


def main(use_real_data: bool = False, data_path: str | None = None) -> None:
    """Train XGBoost model on real or synthetic data."""
    logger.info("=" * 70)
    logger.info("🚀 XGBoost Model Training Pipeline")
    logger.info("=" * 70)

    # Load or generate data
    if use_real_data and data_path:
        X, y = load_real_data(data_path)
    else:
        if use_real_data:
            logger.warning("--real flag set but no --data path provided; using synthetic data.")
        X, y = generate_synthetic_data()

    # Train/test split
    logger.info("\n📋 Splitting data (test_size=%s, random_state=%s)...", TEST_SIZE, RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Training set: %d samples", X_train.shape[0])
    logger.info("Test set:     %d samples", X_test.shape[0])

    # Ensure model directory exists
    model_dir = os.path.dirname(os.path.abspath(MODEL_SAVE_PATH))
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    logger.info("📁 Model directory ready: %s", model_dir)

    # Train
    logger.info("\n🎯 Training XGBoost model...")
    logger.info("Model will be saved to: %s", MODEL_SAVE_PATH)
    _, metadata = train_xgboost_model(
        X_train.to_numpy(),
        y_train.to_numpy(),
        X_test.to_numpy(),
        y_test.to_numpy(),
        model_path=MODEL_SAVE_PATH,
    )

    # Results
    logger.info("\n%s", "=" * 70)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info("Accuracy:    %.4f", metadata.get("accuracy", 0.0))
    logger.info("Model saved: %s", metadata.get("model_path"))
    logger.info("Trained at:  %s", metadata.get("trained_at"))
    logger.info("=" * 70)


if __name__ == "__main__":
    use_real = "--real" in sys.argv
    _data_path: str | None = None

    if "--data" in sys.argv:
        idx = sys.argv.index("--data")
        if idx + 1 < len(sys.argv):
            _data_path = sys.argv[idx + 1]
        else:
            logger.error("--data flag provided but no path given.")
            sys.exit(1)

    main(use_real_data=use_real, data_path=_data_path)
