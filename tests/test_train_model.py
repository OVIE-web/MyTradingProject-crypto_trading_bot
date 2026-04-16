"""
Tests for src/train_model.py module.
Tests model training functionality.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.train_model import generate_synthetic_data, load_real_data, main


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
        from src.config import FEATURE_COLUMNS

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
            from src.config import FEATURE_COLUMNS, TARGET_COLUMN

            data = {col: np.random.randn(50) for col in FEATURE_COLUMNS}
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
            from src.config import FEATURE_COLUMNS

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


class TestMainFunction:
    """Test cases for main function."""

    @patch("src.train_model.train_xgboost_model")
    @patch("src.train_model.generate_synthetic_data")
    @patch("src.train_model.logging")
    def test_main_with_synthetic_data(
        self,
        mock_logging: MagicMock,
        mock_generate_data: MagicMock,
        mock_train_model: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test main function with synthetic data."""
        # Mock data generation
        mock_X = pd.DataFrame(np.random.randn(100, 11))
        mock_y = pd.Series(np.random.randint(0, 3, 100))
        mock_generate_data.return_value = (mock_X, mock_y)

        # Mock model training
        mock_train_model.return_value = (None, {"accuracy": 0.85, "model_path": "/tmp/model.json"})

        # Mock file operations
        with patch("builtins.open", create=True), patch("pathlib.Path.mkdir"):
            main(use_real_data=False)

        # Verify calls
        mock_generate_data.assert_called_once()
        mock_train_model.assert_called_once()
        assert mock_logging.info.call_count >= 5  # Multiple log messages expected

    @patch("src.train_model.train_xgboost_model")
    @patch("src.train_model.load_real_data")
    @patch("src.train_model.logging")
    def test_main_with_real_data(
        self,
        mock_logging: MagicMock,
        mock_load_data: MagicMock,
        mock_train_model: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test main function with real data."""
        # Mock data loading
        mock_X = pd.DataFrame(np.random.randn(100, 11))
        mock_y = pd.Series(np.random.randint(0, 3, 100))
        mock_load_data.return_value = (mock_X, mock_y)

        # Mock model training
        mock_train_model.return_value = (None, {"accuracy": 0.90, "model_path": "/tmp/model.json"})

        # Mock file operations
        with patch("builtins.open", create=True), patch("pathlib.Path.mkdir"):
            main(use_real_data=True, data_path="/tmp/data.csv")

        # Verify calls
        mock_load_data.assert_called_once_with("/tmp/data.csv")
        mock_train_model.assert_called_once()
        assert mock_logging.info.call_count >= 5
