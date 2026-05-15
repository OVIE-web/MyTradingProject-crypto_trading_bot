"""Raw technical indicator calculations for OHLCV market data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pandas import DataFrame
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from app.core.config import (
    ATR_WINDOW,
    BB_WINDOW,
    BB_WINDOW_DEV,
    RSI_WINDOW,
    SMA_LONG_WINDOW,
    SMA_SHORT_WINDOW,
)
from app.utils.validators import ensure_required_columns

LOG = logging.getLogger(__name__)

REQUIRED_OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}


def validate_ohlcv_columns(df: DataFrame) -> None:
    """Ensure the input DataFrame contains the required OHLCV columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    ensure_required_columns(df, REQUIRED_OHLCV_COLUMNS)


def add_rsi(df: DataFrame, window: int = RSI_WINDOW) -> DataFrame:
    """Add Relative Strength Index values."""
    df_indicators = df.copy()
    df_indicators["rsi"] = RSIIndicator(close=df_indicators["close"], window=window).rsi()
    return df_indicators


def add_bollinger_bands(
    df: DataFrame,
    window: int = BB_WINDOW,
    window_dev: int = BB_WINDOW_DEV,
) -> DataFrame:
    """Add Bollinger Band upper, lower, middle, and percent-b values."""
    df_indicators = df.copy()
    bands = BollingerBands(
        close=df_indicators["close"],
        window=window,
        window_dev=window_dev,
    )

    df_indicators["bb_upper"] = bands.bollinger_hband()
    df_indicators["bb_lower"] = bands.bollinger_lband()
    df_indicators["bb_mid"] = bands.bollinger_mavg()

    denominator = df_indicators["bb_upper"] - df_indicators["bb_lower"]
    df_indicators["bb_pct_b"] = (
        (df_indicators["close"] - df_indicators["bb_lower"]) / denominator
    ).replace([np.inf, -np.inf], np.nan)

    return df_indicators


def add_moving_averages(
    df: DataFrame,
    short_window: int = SMA_SHORT_WINDOW,
    long_window: int = SMA_LONG_WINDOW,
) -> DataFrame:
    """Add short/long simple moving averages and crossover flag."""
    df_indicators = df.copy()
    df_indicators["sma_20"] = SMAIndicator(
        close=df_indicators["close"],
        window=short_window,
    ).sma_indicator()
    df_indicators["sma_50"] = SMAIndicator(
        close=df_indicators["close"],
        window=long_window,
    ).sma_indicator()
    df_indicators["ma_cross"] = (df_indicators["sma_20"] > df_indicators["sma_50"]).astype(int)
    return df_indicators


def add_momentum(df: DataFrame, periods: int = 5) -> DataFrame:
    """Add close-price momentum over the configured number of periods."""
    df_indicators = df.copy()
    df_indicators["price_momentum"] = df_indicators["close"].pct_change(periods=periods)
    return df_indicators


def add_atr(df: DataFrame, window: int = ATR_WINDOW) -> DataFrame:
    """Add Average True Range and ATR as a percent of close price."""
    df_indicators = df.copy()

    if len(df_indicators) < window:
        LOG.warning(
            "Insufficient rows (%d) for ATR window=%d; ATR features set to NaN",
            len(df_indicators),
            window,
        )
        df_indicators["atr"] = np.nan
        df_indicators["atr_pct"] = np.nan
        return df_indicators

    atr = AverageTrueRange(
        high=df_indicators["high"],
        low=df_indicators["low"],
        close=df_indicators["close"],
        window=window,
    )
    df_indicators["atr"] = atr.average_true_range()
    df_indicators["atr_pct"] = df_indicators["atr"] / df_indicators["close"]
    return df_indicators


def add_technical_indicators(df: DataFrame) -> DataFrame:
    """Add all raw technical indicators used by the trading model."""
    validate_ohlcv_columns(df)

    df_indicators = df.copy()
    df_indicators = add_rsi(df_indicators)
    df_indicators = add_bollinger_bands(df_indicators)
    df_indicators = add_moving_averages(df_indicators)
    df_indicators = add_momentum(df_indicators)
    df_indicators = add_atr(df_indicators)

    return df_indicators
