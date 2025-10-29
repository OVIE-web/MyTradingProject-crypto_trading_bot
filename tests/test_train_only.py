import os
import numpy as np
import pandas as pd

from src import model_manager as mm
from src.config import FEATURE_COLUMNS


def test_train_and_save_load(tmp_path, monkeypatch):
    """Train a tiny XGBoost model, ensure it is saved and can be loaded."""
    model_path = tmp_path / "xgboost_model.json"

    # Patch the module-level MODEL_SAVE_PATH to write into tmp_path
    monkeypatch.setattr(mm, "MODEL_SAVE_PATH", str(model_path))

    # Replace RandomizedSearchCV with a lightweight dummy that simply fits the estimator
    class DummySearch:
        def __init__(self, estimator, **kwargs):
            self.best_estimator_ = estimator
            self.best_params_ = {}

        def fit(self, X, y):
            # Fit the underlying estimator directly
            self.best_estimator_.fit(X, y)
            return self

    monkeypatch.setattr(mm, "RandomizedSearchCV", DummySearch)

    # Build a small but reasonable synthetic dataset
    n_samples = 150
    n_features = len(FEATURE_COLUMNS)
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(n_samples, n_features), columns=FEATURE_COLUMNS)
    # Create balanced labels in {-1,0,1}
    y = pd.Series(np.tile([-1, 0, 1], int(np.ceil(n_samples / 3)))[:n_samples])

    # Train/test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trained_model, best_params = mm.train_xgboost_model(X_train, y_train, X_test, y_test)

    # Confirm file saved
    assert model_path.exists(), f"Model file was not created at {model_path}"

    # Confirm it can be loaded
    loaded = mm.load_trained_model(str(model_path))
    assert hasattr(loaded, "predict_proba")
