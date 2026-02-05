from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import model_manager as mm
from src.config import FEATURE_COLUMNS


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
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ✅ Train model with explicit model_path
    _, _ = mm.train_xgboost_model(X_train, y_train, X_test, y_test, model_path=str(model_path))

    # Confirm file saved
    assert model_path.exists(), f"Model file was not created at {model_path}"

    # Confirm it can be loaded
    loaded = mm.load_trained_model(str(model_path))
    assert hasattr(loaded, "predict_proba")
