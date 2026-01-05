
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_and_preprocess_data


@pytest.fixture
def dummy_csv_file(tmp_path):
    """Creates a temporary CSV file for testing load_and_preprocess_data."""
    file_path = tmp_path / "dummy_data.csv"
    df_data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000, 1010, 1020, 1030, 1040],
        'extra_col': [1, 2, np.nan, 4, 5]
    }
    pd.DataFrame(df_data).to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def dummy_csv_with_nans(tmp_path):
    """Creates a temporary CSV file with NaNs for testing imputation."""
    file_path = tmp_path / "dummy_nan_data.csv"
    df_data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        'open': [100, 101, np.nan, 103, 104],
        'high': [105, np.nan, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000, 1010, 1020, 1030, 1040]
    }
    pd.DataFrame(df_data).to_csv(file_path, index=False)
    return file_path


def test_load_and_preprocess_data_success(dummy_csv_file):
    df = load_and_preprocess_data(dummy_csv_file)
    assert not df.empty
    assert 'close' in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    
    
def test_load_and_preprocess_data_nan_handling(dummy_csv_with_nans):
    df = load_and_preprocess_data(dummy_csv_with_nans)
    assert not df.isnull().any().any()  # Ensure no NaNs remain

    # Validate forward fill worked
    assert df['open'].iloc[2] == 101.0
    assert df['high'].iloc[1] == 105.0

def test_load_and_preprocess_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_and_preprocess_data("non_existent_file.csv")


def test_load_and_preprocess_data_missing_timestamp(tmp_path):
    file_path = tmp_path / "no_ts_data.csv"
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    df.to_csv(file_path, index=False)

    df_loaded = load_and_preprocess_data(file_path)
    assert not isinstance(df_loaded.index, pd.DatetimeIndex)
    assert 'timestamp' not in df_loaded.columns

