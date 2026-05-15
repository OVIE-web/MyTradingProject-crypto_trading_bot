# app/services/training_service.py
"""Standalone training service for the XGBoost trading model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.config import (
    DATA_FILE_PATH,
    FEATURE_COLUMNS,
    MODEL_SAVE_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from app.services.model_service import train_xgboost_model

LOG = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 500) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic training data for local testing.

    Labels use the model convention:
        0 = SELL
        1 = HOLD
        2 = BUY
    """
    LOG.info("Generating synthetic training data...")

    X = pd.DataFrame(
        np.random.randn(n_samples, len(FEATURE_COLUMNS)),
        columns=FEATURE_COLUMNS,
    )

    y = pd.Series(
        np.random.choice([0, 1, 2], n_samples),
        name=TARGET_COLUMN,
    )

    LOG.info("Synthetic dataset shape: %s", X.shape)
    LOG.info("Class distribution:\n%s", y.value_counts())

    return X, y


def load_real_data(data_path: str | Path = DATA_FILE_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and validate a CSV data file for model training.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    LOG.info("Loading real training data from %s...", data_path)

    df = pd.read_csv(data_path)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns in data file: {missing_features}")

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found. Available columns: {list(df.columns)}"
        )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    LOG.info("Real data loaded. Shape=%s", X.shape)
    LOG.info("Class distribution:\n%s", y.value_counts())

    return X, y


def run_training_pipeline(
    use_real_data: bool = False,
    data_path: str | Path | None = None,
    model_path: str | Path = MODEL_SAVE_PATH,
) -> dict:
    """
    Train the XGBoost model using real data.

    Args:
        use_real_data: If True, load data from CSV.
        data_path: Optional path to real training data.
        model_path: Output path for the trained model.

    Returns:
        Training metadata dictionary.
    """
    LOG.info("=" * 70)
    LOG.info("XGBoost Model Training Pipeline")
    LOG.info("=" * 70)

    if use_real_data:
        X, y = load_real_data(data_path or DATA_FILE_PATH)
    else:
        X, y = generate_synthetic_data()

    stratify_target = y if len(y.unique()) > 1 else None

    LOG.info(
        "Splitting data. test_size=%s random_state=%s",
        TEST_SIZE,
        RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_target,
    )

    LOG.info("Training set: %d samples", X_train.shape[0])
    LOG.info("Test set: %d samples", X_test.shape[0])

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    LOG.info("Model will be saved to: %s", model_path)

    _, metadata = train_xgboost_model(
        X_train.to_numpy(),
        y_train.to_numpy(),
        X_test.to_numpy(),
        y_test.to_numpy(),
        model_path=str(model_path),
    )

    LOG.info("=" * 70)
    LOG.info("TRAINING COMPLETE")
    LOG.info("Accuracy:    %.4f", metadata.get("accuracy", 0.0))
    LOG.info("Model saved: %s", metadata.get("model_path"))
    LOG.info("Trained at:  %s", metadata.get("trained_at"))
    LOG.info("=" * 70)

    return metadata


def main() -> None:
    """
    CLI entrypoint for model training.
    """
    parser = argparse.ArgumentParser(description="Train the XGBoost trading model.")

    parser.add_argument(
        "--real",
        action="store_true",
        help="Train using real CSV data instead of synthetic data.",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to real training data CSV.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_SAVE_PATH,
        help="Path where the trained model should be saved.",
    )

    args = parser.parse_args()

    run_training_pipeline(
        use_real_data=args.real,
        data_path=args.data,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    main()
