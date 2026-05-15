"""Model-ready feature engineering helpers built on raw indicators."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler

from app.core.config import RSI_LOWER_QUANTILE, RSI_UPPER_QUANTILE
from app.tools.technical_indicators import add_technical_indicators

LOG = logging.getLogger(__name__)


def calculate_technical_indicators(df: DataFrame) -> DataFrame:
    """
    Calculate technical indicators and return a cleaned feature DataFrame.

    Contract:
        - Input DataFrame is not mutated
        - Output contains model-ready indicator columns
        - Rows with rolling indicator NaNs are removed
    """
    df_features = add_technical_indicators(df)

    before = len(df_features)
    df_features.dropna(inplace=True)
    dropped = before - len(df_features)

    if dropped:
        LOG.info("Dropped %d rows due to rolling indicator NaNs", dropped)

    return df_features


def get_rsi_quantile_thresholds(
    rsi_series: Series,
    lower_quantile: float = RSI_LOWER_QUANTILE,
    upper_quantile: float = RSI_UPPER_QUANTILE,
) -> tuple[float, float]:
    """Compute dynamic RSI thresholds based on quantiles."""
    if not isinstance(rsi_series, pd.Series):
        raise TypeError("rsi_series must be a pandas Series")

    if not (0.0 < lower_quantile < upper_quantile < 1.0):
        raise ValueError("Quantiles must satisfy 0 < lower < upper < 1")

    rsi_clean = rsi_series.dropna()
    if rsi_clean.empty:
        LOG.warning("Empty RSI series; falling back to default thresholds")
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
        1 means buy
        0 means hold
       -1 means sell
    """
    if rsi_col not in df.columns:
        raise ValueError(f"Missing RSI column: {rsi_col}")

    df_labeled = df.copy()
    df_labeled["signal"] = 0

    df_labeled.loc[df_labeled[rsi_col] <= lower_threshold, "signal"] = 1
    df_labeled.loc[df_labeled[rsi_col] >= upper_threshold, "signal"] = -1

    df_labeled["signal"] = df_labeled["signal"].astype(int)
    return df_labeled


def normalize_features(df: DataFrame) -> DataFrame:
    """
    Normalize numeric model features using StandardScaler.

    Binary columns are preserved and column order is stable.
    """
    df_norm = df.copy()

    binary_cols = [column for column in ("ma_cross", "signal") if column in df_norm.columns]
    exclude_cols = binary_cols + ["open", "high", "low", "close", "volume"]
    numeric_cols = df_norm.select_dtypes(include=np.number).columns.difference(exclude_cols).tolist()

    if not numeric_cols:
        LOG.warning("No numeric columns found for normalization")
        return df_norm

    scaler = StandardScaler()
    df_norm[numeric_cols] = scaler.fit_transform(df_norm[numeric_cols])

    ordered_cols = numeric_cols + binary_cols
    remaining = [column for column in df_norm.columns if column not in ordered_cols]

    return df_norm[ordered_cols + remaining]
