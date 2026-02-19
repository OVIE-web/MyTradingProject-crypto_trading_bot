"""Tests for data_loader module."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_and_preprocess_data


@pytest.fixture
def dummy_csv_file(tmp_path: Path) -> Path:
    """Creates a temporary CSV file for testing load_and_preprocess_data."""
    file_path = tmp_path / "dummy_data.csv"
    df_data = {
        "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="D"),
        "open": [100, 101, 102, 103, 104],
        "high": [105, 106, 107, 108, 109],
        "low": [95, 96, 97, 98, 99],
        "close": [102, 103, 104, 105, 106],
        "volume": [1000, 1010, 1020, 1030, 1040],
        "extra_col": [1, 2, np.nan, 4, 5],
    }
    pd.DataFrame(df_data).to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def dummy_csv_with_nans(tmp_path: Path) -> Path:
    """Creates a temporary CSV file with NaNs for testing imputation."""
    file_path = tmp_path / "dummy_nan_data.csv"
    df_data = {
        "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="D"),
        "open": [100, 101, np.nan, 103, 104],
        "high": [105, np.nan, 107, 108, 109],
        "low": [95, 96, 97, 98, 99],
        "close": [102, 103, 104, 105, 106],
        "volume": [1000, 1010, 1020, 1030, 1040],
    }
    pd.DataFrame(df_data).to_csv(file_path, index=False)
    return file_path


def test_load_and_preprocess_data_success(dummy_csv_file: Path) -> None:
    """Test that load_and_preprocess_data successfully loads and processes valid CSV."""
    df = load_and_preprocess_data(str(dummy_csv_file))
    assert not df.empty
    assert "close" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing


def test_load_and_preprocess_data_nan_handling(dummy_csv_with_nans: Path) -> None:
    """Test that load_and_preprocess_data handles NaN values correctly."""
    df = load_and_preprocess_data(str(dummy_csv_with_nans))
    assert not df.isnull().any().any()  # Ensure no NaNs remain

    # Validate forward fill worked
    assert df["open"].iloc[2] == 101.0
    assert df["high"].iloc[1] == 105.0


def test_load_and_preprocess_data_file_not_found() -> None:
    """Test that FileNotFoundError is raised for non-existent files."""
    with pytest.raises(FileNotFoundError):
        load_and_preprocess_data("non_existent_file.csv")


def test_load_and_preprocess_data_missing_timestamp(tmp_path: Path) -> None:
    """Test that missing required columns raises ValueError."""
    file_path = tmp_path / "no_ts_data.csv"
    # Create CSV with missing required columns
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df.to_csv(file_path, index=False)

    # The function should raise ValueError because required columns are missing
    with pytest.raises(ValueError, match="Invalid data schema. Missing required columns"):
        load_and_preprocess_data(str(file_path))


@pytest.fixture
def dummy_csv_no_nans(tmp_path: Path) -> Path:
    """Creates a temporary CSV file with no NaNs."""
    file_path = tmp_path / "dummy_no_nan_data.csv"
    df_data = {
        "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="D"),
        "open": [100, 101, 102, 103, 104],
        "high": [105, 106, 107, 108, 109],
        "low": [95, 96, 97, 98, 99],
        "close": [102, 103, 104, 105, 106],
        "volume": [1000, 1010, 1020, 1030, 1040],
    }
    pd.DataFrame(df_data).to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def dummy_csv_all_invalid_ts(tmp_path: Path) -> Path:
    """Creates a temporary CSV file where all timestamps are invalid."""
    file_path = tmp_path / "invalid_ts_data.csv"
    df_data = {
        "timestamp": ["invalid1", "invalid2", "invalid3"],
        "open": [100, 101, 102],
        "high": [105, 106, 107],
        "low": [95, 96, 97],
        "close": [102, 103, 104],
        "volume": [1000, 1010, 1020],
    }
    pd.DataFrame(df_data).to_csv(file_path, index=False)
    return file_path


def test_load_and_preprocess_data_no_nan_handling(dummy_csv_no_nans: Path) -> None:
    """Test that load_and_preprocess_data works correctly with data containing no NaNs."""
    df = load_and_preprocess_data(str(dummy_csv_no_nans))
    assert not df.isnull().any().any()
    assert df.shape == (5, 5)


def test_load_and_preprocess_data_all_invalid_timestamps(
    dummy_csv_all_invalid_ts: Path,
) -> None:
    """Test that a ValueError is raised if all timestamps are invalid."""
    with pytest.raises(
        ValueError, match="All rows have invalid or missing timestamps. No data to process."
    ):
        load_and_preprocess_data(str(dummy_csv_all_invalid_ts))


def test_load_and_preprocess_data_generic_exception() -> None:
    """Test that a generic Exception during pd.read_csv is caught and re-raised."""
    with patch("pandas.read_csv", side_effect=Exception("mocked error")):
        with pytest.raises(Exception, match="mocked error"):
            load_and_preprocess_data("any_file.csv")
