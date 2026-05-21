from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.tools.technical_indicators import (
    add_atr,
    add_bollinger_bands,
    add_momentum,
    add_moving_averages,
    add_rsi,
    add_technical_indicators,
    validate_ohlcv_columns,
)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    close = pd.Series(np.linspace(100.0, 180.0, 80))
    return pd.DataFrame(
        {
            "open": close - 1.0,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": np.linspace(1_000.0, 2_000.0, 80),
        }
    )


def test_validate_ohlcv_columns_accepts_required_columns(sample_ohlcv_data: pd.DataFrame) -> None:
    validate_ohlcv_columns(sample_ohlcv_data)


def test_validate_ohlcv_columns_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        validate_ohlcv_columns("not a dataframe")  # type: ignore[arg-type]


def test_validate_ohlcv_columns_rejects_missing_required_column(
    sample_ohlcv_data: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_ohlcv_columns(sample_ohlcv_data.drop(columns=["volume"]))


def test_add_rsi_does_not_mutate_input(sample_ohlcv_data: pd.DataFrame) -> None:
    original = sample_ohlcv_data.copy()

    result = add_rsi(sample_ohlcv_data)

    assert "rsi" in result.columns
    assert "rsi" not in sample_ohlcv_data.columns
    pd.testing.assert_frame_equal(sample_ohlcv_data, original)


def test_add_bollinger_bands_adds_expected_columns(sample_ohlcv_data: pd.DataFrame) -> None:
    result = add_bollinger_bands(sample_ohlcv_data)

    assert {"bb_upper", "bb_lower", "bb_mid", "bb_pct_b"}.issubset(result.columns)
    assert result["bb_pct_b"].dropna().between(0, 1).all()


def test_add_moving_averages_adds_cross_signal(sample_ohlcv_data: pd.DataFrame) -> None:
    result = add_moving_averages(sample_ohlcv_data)

    assert {"sma_20", "sma_50", "ma_cross"}.issubset(result.columns)
    assert set(result["ma_cross"].unique()).issubset({0, 1})


def test_add_momentum_adds_price_momentum(sample_ohlcv_data: pd.DataFrame) -> None:
    result = add_momentum(sample_ohlcv_data, periods=2)

    assert "price_momentum" in result.columns
    assert result["price_momentum"].iloc[:2].isna().all()
    assert result["price_momentum"].iloc[2:].notna().all()


def test_add_atr_sets_nan_when_not_enough_rows(sample_ohlcv_data: pd.DataFrame) -> None:
    result = add_atr(sample_ohlcv_data.head(5), window=14)

    assert result["atr"].isna().all()
    assert result["atr_pct"].isna().all()


def test_add_technical_indicators_adds_all_raw_indicator_columns(
    sample_ohlcv_data: pd.DataFrame,
) -> None:
    result = add_technical_indicators(sample_ohlcv_data)

    assert {
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
    }.issubset(result.columns)
    assert len(result) == len(sample_ohlcv_data)
