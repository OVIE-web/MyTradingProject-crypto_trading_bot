# src/feature_engineer.py
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from src.config import (
    ATR_WINDOW,
    BB_WINDOW,
    BB_WINDOW_DEV,
    RSI_LOWER_QUANTILE,
    RSI_UPPER_QUANTILE,
    RSI_WINDOW,
    SMA_LONG_WINDOW,
    SMA_SHORT_WINDOW,
)

LOG = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Feature creation
# ------------------------------------------------------------------
def calculate_technical_indicators(df: DataFrame) -> DataFrame:
    """
    Calculate technical indicators and return a cleaned DataFrame.

    Contract:
        - Input DataFrame is not mutated
        - Output contains only valid numeric features
        - No NaNs remain
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df_feat = df.copy()

    # RSI
    df_feat["rsi"] = RSIIndicator(
        close=df_feat["close"],
        window=RSI_WINDOW,
    ).rsi()

    # Bollinger Bands
    bb = BollingerBands(
        close=df_feat["close"],
        window=BB_WINDOW,
        window_dev=BB_WINDOW_DEV,
    )
    df_feat["bb_upper"] = bb.bollinger_hband()
    df_feat["bb_lower"] = bb.bollinger_lband()
    df_feat["bb_mid"] = bb.bollinger_mavg()

    denom = df_feat["bb_upper"] - df_feat["bb_lower"]
    df_feat["bb_pct_b"] = ((df_feat["close"] - df_feat["bb_lower"]) / denom).replace(
        [np.inf, -np.inf], np.nan
    )

    # Moving averages
    df_feat["sma_short"] = SMAIndicator(
        close=df_feat["close"],
        window=SMA_SHORT_WINDOW,
    ).sma_indicator()

    df_feat["sma_long"] = SMAIndicator(
        close=df_feat["close"],
        window=SMA_LONG_WINDOW,
    ).sma_indicator()

    df_feat["ma_cross"] = (df_feat["sma_short"] > df_feat["sma_long"]).astype(int)

    # Momentum
    df_feat["price_momentum"] = df_feat["close"].pct_change(periods=5)

    # ATR
    if len(df_feat) < ATR_WINDOW:
        LOG.warning(
            "Insufficient rows (%d) for ATR window=%d — ATR features set to NaN",
            len(df_feat),
            ATR_WINDOW,
        )
        df_feat["atr"] = np.nan
        df_feat["atr_pct"] = np.nan
    else:
        atr = AverageTrueRange(
            high=df_feat["high"],
            low=df_feat["low"],
            close=df_feat["close"],
            window=ATR_WINDOW,
        )
        df_feat["atr"] = atr.average_true_range()
        df_feat["atr_pct"] = df_feat["atr"] / df_feat["close"]

    # Volume change
    df_feat["volume_pct_change"] = df_feat["volume"].pct_change()

    # Drop NaNs from rolling indicators
    before = len(df_feat)
    df_feat.dropna(inplace=True)
    dropped = before - len(df_feat)

    if dropped:
        LOG.info("Dropped %d rows due to rolling indicator NaNs", dropped)

    return df_feat


# ------------------------------------------------------------------
# RSI thresholds & labeling
# ------------------------------------------------------------------
def get_rsi_quantile_thresholds(
    rsi_series: Series,
    lower_quantile: float = RSI_LOWER_QUANTILE,
    upper_quantile: float = RSI_UPPER_QUANTILE,
) -> Tuple[float, float]:
    """
    Compute dynamic RSI thresholds based on quantiles.
    """
    if not isinstance(rsi_series, pd.Series):
        raise TypeError("rsi_series must be a pandas Series")

    if not (0.0 < lower_quantile < upper_quantile < 1.0):
        raise ValueError("Quantiles must satisfy 0 < lower < upper < 1")

    rsi_clean = rsi_series.dropna()
    if rsi_clean.empty:
        LOG.warning("Empty RSI series — falling back to default thresholds")
        return 30.0, 70.0

    lower = float(rsi_clean.quantile(lower_quantile))
    upper = float(rsi_clean.quantile(upper_quantile))

    return max(0.0, lower), min(100.0, upper)


def apply_rsi_labels(
    df: DataFrame,
    rsi_col: str = "rsi",
    lower_threshold: float = 30.0,
    upper_threshold: float = 70.0,
) -> DataFrame:
    """
    Generate trading signals from RSI values.

    Signals:
        1  → Buy
        0  → Hold
       -1  → Sell
    """
    if rsi_col not in df.columns:
        raise ValueError(f"Missing RSI column: {rsi_col}")

    df_labeled = df.copy()
    df_labeled["signal"] = 0

    df_labeled.loc[df_labeled[rsi_col] <= lower_threshold, "signal"] = 1
    df_labeled.loc[df_labeled[rsi_col] >= upper_threshold, "signal"] = -1

    df_labeled["signal"] = df_labeled["signal"].astype(int)
    return df_labeled


# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------
def normalize_features(df: DataFrame) -> DataFrame:
    """
    Normalize numeric features using StandardScaler.

    Binary columns (e.g. signals) are preserved and column order is stable.
    """
    df_norm = df.copy()

    binary_cols = [c for c in ("ma_cross", "signal") if c in df_norm.columns]
    numeric_cols: List[str] = (
        df_norm.select_dtypes(include=np.number).columns.difference(binary_cols).tolist()
    )

    if not numeric_cols:
        LOG.warning("No numeric columns found for normalization")
        return df_norm

    scaler = StandardScaler()
    df_norm[numeric_cols] = scaler.fit_transform(df_norm[numeric_cols])

    # Preserve column order
    ordered_cols = numeric_cols + binary_cols
    remaining = [c for c in df_norm.columns if c not in ordered_cols]

    return df_norm[ordered_cols + remaining]
