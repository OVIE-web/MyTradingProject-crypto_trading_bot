# tests/test_feature_engineer.py

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.feature_engineer import (
    apply_rsi_labels,
    calculate_technical_indicators,
    get_rsi_quantile_thresholds,
    normalize_features,
)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame(
        {
            "open": np.random.uniform(100, 200, 100),
            "high": np.random.uniform(100, 200, 100),
            "low": np.random.uniform(90, 190, 100),
            "close": np.random.uniform(100, 200, 100),
            "volume": np.random.uniform(1000, 2000, 100),
        }
    )


@pytest.fixture
def sample_df_with_indicators(sample_ohlcv_data: pd.DataFrame) -> pd.DataFrame:
    df = calculate_technical_indicators(sample_ohlcv_data.copy())
    return df


def test_calculate_technical_indicators(sample_ohlcv_data: pd.DataFrame) -> None:
    df = calculate_technical_indicators(sample_ohlcv_data.copy())

    expected_cols = [
        "rsi",
        "bb_upper",
        "bb_lower",
        "bb_mid",
        "bb_pct_b",
        "sma_20",
        "sma_50",
        "ma_cross",
        "price_momentum",
        "atr",
        "atr_pct",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing indicator column: {col}"

    assert not df.isnull().any().any(), "NaNs found after indicator calculation"
    assert (df["rsi"] >= 0).all() and (df["rsi"] <= 100).all()
    assert df["ma_cross"].isin([0, 1]).all()


def test_get_rsi_quantile_thresholds() -> None:
    rsi_series = pd.Series(np.random.rand(100) * 100)
    lower, upper = get_rsi_quantile_thresholds(rsi_series)
    assert 0 <= lower < upper <= 100

    lower_custom, upper_custom = get_rsi_quantile_thresholds(rsi_series, 0.1, 0.9)
    assert lower_custom < lower and upper_custom > upper


def test_apply_rsi_labels(sample_df_with_indicators: pd.DataFrame) -> None:
    df = apply_rsi_labels(sample_df_with_indicators.copy(), lower_threshold=30, upper_threshold=70)
    assert "signal" in df.columns
    assert df["signal"].isin([-1, 0, 1]).all()
    assert (df.loc[df["rsi"] <= 30, "signal"] == 1).all()
    assert (df.loc[df["rsi"] >= 70, "signal"] == -1).all()


def test_normalize_features(sample_df_with_indicators: pd.DataFrame) -> None:
    df_norm = normalize_features(sample_df_with_indicators.copy())

    for col in FEATURE_COLUMNS:
        if col in df_norm.columns and col not in ["ma_cross", "signal"]:
            mean = df_norm[col].mean()
            std = df_norm[col].std()
            assert abs(mean) < 0.1, f"{col} mean not centered"
            assert abs(std - 1) < 0.1, f"{col} std not normalized"
            assert abs(std - 1) < 0.1, f"{col} std not normalized"
