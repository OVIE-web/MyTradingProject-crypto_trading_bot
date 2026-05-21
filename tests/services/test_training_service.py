"""
Tests for app.services.training_service.
Tests model training functionality.
"""

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.training_service import (
    generate_synthetic_data,
    load_real_data,
    run_training_pipeline,
)


class TestGenerateSyntheticData:
    """Test cases for generate_synthetic_data function."""

    def test_generate_synthetic_data_shape(self) -> None:
        """Test that synthetic data has correct shape."""
        X, y = generate_synthetic_data(n_samples=100)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape == (100, 11)  # 11 features as defined in config
        assert y.shape == (100,)

    def test_generate_synthetic_data_values(self) -> None:
        """Test that synthetic data contains expected values."""
        X, y = generate_synthetic_data(n_samples=50)

        # Check that all feature columns exist
        from app.core.config import FEATURE_COLUMNS

        for col in FEATURE_COLUMNS:
            assert col in X.columns

        # Check that target values are in expected range (0, 1, 2)
        assert y.min() >= 0
        assert y.max() <= 2
        assert y.dtype == int


class TestLoadRealData:
    """Test cases for load_real_data function."""

    def test_load_real_data_success(self) -> None:
        """Test successful loading of real data."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Create sample data with required columns
            from app.core.config import FEATURE_COLUMNS, TARGET_COLUMN

            data: dict[str, Any] = {col: np.random.randn(50) for col in FEATURE_COLUMNS}
            data[TARGET_COLUMN] = np.random.randint(0, 3, 50)
            df = pd.DataFrame(data)
            df.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            X, y = load_real_data(temp_file)

            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)
            assert X.shape[0] == 50
            assert y.shape[0] == 50
            assert len(X.columns) == len(FEATURE_COLUMNS)
        finally:
            os.unlink(temp_file)

    def test_load_real_data_missing_feature_columns(self) -> None:
        """Test error when required feature columns are missing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Create data missing some feature columns
            df = pd.DataFrame(
                {
                    "close": [100, 101, 102],
                    "volume": [1000, 1100, 1200],
                    "signal": [0, 1, 2],  # Missing most feature columns
                }
            )
            df.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            with pytest.raises(KeyError, match="Missing feature columns"):
                load_real_data(temp_file)
        finally:
            os.unlink(temp_file)

    def test_load_real_data_missing_target_column(self) -> None:
        """Test error when target column is missing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            from app.core.config import FEATURE_COLUMNS

            # Create data with features but no target
            data = {col: np.random.randn(10) for col in FEATURE_COLUMNS}
            df = pd.DataFrame(data)
            df.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            with pytest.raises(KeyError, match="Target column.*not found"):
                load_real_data(temp_file)
        finally:
            os.unlink(temp_file)

    def test_load_real_data_file_not_found(self) -> None:
        """Test error when data file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_real_data("/nonexistent/file.csv")


class TestRunTrainingPipeline:
    """Test cases for the training pipeline."""

    @patch("app.services.training_service.train_xgboost_model")
    @patch("app.services.training_service.generate_synthetic_data")
    def test_run_training_pipeline_with_synthetic_data(
        self,
        mock_generate_data: MagicMock,
        mock_train_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test the training pipeline with synthetic data."""
        # Mock data generation
        mock_X = pd.DataFrame(np.random.randn(100, 11))
        mock_y = pd.Series(np.tile([0, 1, 2], 34)[:100])
        mock_generate_data.return_value = (mock_X, mock_y)

        # Mock model training
        mock_train_model.return_value = (None, {"accuracy": 0.85, "model_path": "/tmp/model.json"})

        metadata = run_training_pipeline(use_real_data=False, model_path=tmp_path / "model.json")

        # Verify calls
        mock_generate_data.assert_called_once()
        mock_train_model.assert_called_once()
        assert metadata["accuracy"] == 0.85

    @patch("app.services.training_service.train_xgboost_model")
    @patch("app.services.training_service.load_real_data")
    def test_run_training_pipeline_with_real_data(
        self,
        mock_load_data: MagicMock,
        mock_train_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test the training pipeline with real data."""
        # Mock data loading
        mock_X = pd.DataFrame(np.random.randn(100, 11))
        mock_y = pd.Series(np.tile([0, 1, 2], 34)[:100])
        mock_load_data.return_value = (mock_X, mock_y)

        # Mock model training
        mock_train_model.return_value = (None, {"accuracy": 0.90, "model_path": "/tmp/model.json"})

        metadata = run_training_pipeline(
            use_real_data=True,
            data_path="/tmp/data.csv",
            model_path=tmp_path / "model.json",
        )

        # Verify calls
        mock_load_data.assert_called_once_with("/tmp/data.csv")
        mock_train_model.assert_called_once()
        assert metadata["accuracy"] == 0.90
