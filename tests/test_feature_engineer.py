import pytest
import pandas as pd
import numpy as np
from src.feature_engineer import (
    calculate_technical_indicators,
    get_rsi_quantile_thresholds,
    apply_rsi_labels,
    normalize_features
)
from src.config import FEATURE_COLUMNS, RSI_LOWER_QUANTILE, RSI_UPPER_QUANTILE


def test_calculate_technical_indicators(sample_ohlcv_data):
    df = calculate_technical_indicators(sample_ohlcv_data.copy())

    expected_indicator_cols = [
        'rsi', 'bb_upper', 'bb_lower', 'bb_mid', 'bb_pct_b',
        'sma_20', 'sma_50', 'ma_cross', 'price_momentum',
        'atr', 'atr_pct', 'volume_pct_change'
    ]
    for col in expected_indicator_cols:
        assert col in df.columns, f"Missing indicator column: {col}"

    assert not df.isnull().any().any(), "NaNs found after indicator calculation and dropna"
    assert len(df) < len(sample_ohlcv_data), "Rows not dropped after indicator calculation"

    assert (df['rsi'] >= 0).all() and (df['rsi'] <= 100).all()
    assert not np.array_equal(df['sma_20'].values, df['close'].values)
    assert df['ma_cross'].isin([0, 1]).all()

def test_get_rsi_quantile_thresholds():
    rsi_series = pd.Series(np.random.rand(100) * 100)
    lower, upper = get_rsi_quantile_thresholds(rsi_series)

    assert 0 <= lower <= 100
    assert 0 <= upper <= 100
    assert lower < upper

    lower_custom, upper_custom = get_rsi_quantile_thresholds(rsi_series, 0.1, 0.9)
    assert lower_custom < lower
    assert upper_custom > upper

def test_apply_rsi_labels(sample_df_with_indicators):
    df = apply_rsi_labels(sample_df_with_indicators.copy(), lower_threshold=30, upper_threshold=70)

    assert 'signal' in df.columns
    assert df['signal'].isin([-1, 0, 1]).all()

    assert (df.loc[df['rsi'] <= 30, 'signal'] == 1).all()
    assert (df.loc[df['rsi'] >= 70, 'signal'] == -1).all()
    assert (df.loc[(df['rsi'] > 30) & (df['rsi'] < 70), 'signal'] == 0).all()

def test_normalize_features(sample_df_with_indicators):
    df_normalized = normalize_features(sample_df_with_indicators.copy())

    # 'signal' may be target and ignored — just make sure not scaled if present
    if 'ma_cross' in df_normalized.columns:
        assert df_normalized['ma_cross'].isin([0, 1]).all()
        assert not np.isclose(df_normalized['ma_cross'].mean(), 0)

    for col in FEATURE_COLUMNS:
        if col != 'ma_cross' and col in df_normalized.columns:
            mean = df_normalized[col].mean()
            std = df_normalized[col].std()
            assert abs(mean) < 0.1, f"{col} mean not centered around 0"
            assert abs(std - 1) < 0.1, f"{col} std not normalized to 1"
