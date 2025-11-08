# src/feature_engineer.py

import pandas as pd
import numpy as np
import logging
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import SMAIndicator
from sklearn.preprocessing import StandardScaler

from src.config import (
    RSI_WINDOW, BB_WINDOW, BB_WINDOW_DEV, SMA_SHORT_WINDOW, SMA_LONG_WINDOW,
    ATR_WINDOW, RSI_LOWER_QUANTILE, RSI_UPPER_QUANTILE
)


def calculate_technical_indicators(df):
    """
    Calculates essential technical indicators and adds them to the DataFrame.
    Ensures ATR and related features exist and drops rows with NaN.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input for technical indicators must be a pandas DataFrame')
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logging.error(f"Missing required columns for indicators: {missing}")
        raise ValueError(f"DataFrame must contain {required_cols} columns")

    try:
        # RSI
        df['rsi'] = RSIIndicator(close=df['close'], window=RSI_WINDOW).rsi()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=BB_WINDOW, window_dev=BB_WINDOW_DEV)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()

        df['bb_pct_b'] = ((df['close'] - df['bb_lower']) /
                          (df['bb_upper'] - df['bb_lower'])).replace([np.inf, -np.inf], np.nan)
        
        # Moving Averages and crossover
        df['sma_20'] = SMAIndicator(close=df['close'], window=SMA_SHORT_WINDOW).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=SMA_LONG_WINDOW).sma_indicator()
        df['ma_cross'] = (df['sma_20'] > df['sma_50']).astype(int)

        # Price momentum
        df['price_momentum'] = df['close'].pct_change(periods=5)
        
        # ATR (Average True Range)
        if df.empty or len(df) < ATR_WINDOW:
            logging.warning(f"Insufficient data ({len(df)}) for ATR window {ATR_WINDOW}. Filling NaN.")
            df['atr'] = np.nan
            df['atr_pct'] = np.nan
        else:
            atr = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=ATR_WINDOW
            )
            df['atr'] = atr.average_true_range()
            df['atr_pct'] = df['atr'] / df['close']

        # Volume metrics
        df['volume_pct_change'] = df['volume'].pct_change()
        
        # Drop possible legacy columns
        drop_cols = ['BB_hband', 'BB_lband', 'BB_mavg', 'RSI', 'SMA_long', 'SMA_short']
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')
        
        # Drop NaNs from rolling windows
        original_len = len(df)
        df.dropna(inplace=True)
        logging.info(f"Calculated indicators. Dropped {original_len - len(df)} rows due to NaN.")
        
        return df

    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        raise


def get_rsi_quantile_thresholds(rsi_series, lower_quantile=RSI_LOWER_QUANTILE, upper_quantile=RSI_UPPER_QUANTILE):
    """Compute dynamic RSI thresholds."""
    if not isinstance(rsi_series, pd.Series):
        raise TypeError('rsi_series must be a pandas Series')

    rsi_clean = rsi_series.dropna()
    if rsi_clean.empty:
        logging.warning("Empty RSI series, using default thresholds 30/70.")
        return 30, 70

    if not (0 < lower_quantile < upper_quantile < 1):
        raise ValueError('Quantiles must be between 0 and 1 and lower < upper')

    lower_th = rsi_clean.quantile(lower_quantile)
    upper_th = rsi_clean.quantile(upper_quantile)
    return max(0, lower_th), min(100, upper_th)


def apply_rsi_labels(df, rsi_col='rsi', lower_threshold=30, upper_threshold=70):
    """Generate trading signal labels based on RSI."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')
    if rsi_col not in df.columns:
        raise ValueError(f'Missing RSI column: {rsi_col}')

    df = df.copy()
    df['signal'] = 0
    df.loc[df[rsi_col] <= lower_threshold, 'signal'] = 1
    df.loc[df[rsi_col] >= upper_threshold, 'signal'] = -1
    df['signal'] = df['signal'].astype(int)
    return df


def normalize_features(df):
    """Normalize numeric columns (excluding binary) using StandardScaler."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    binary_cols = [c for c in ['ma_cross', 'signal'] if c in df.columns]
    num_cols = df.select_dtypes(include=np.number).columns.difference(binary_cols)

    if num_cols.empty:
        logging.warning("No numeric columns to normalize.")
        return df.copy()

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[num_cols]),
        columns=num_cols,
        index=df.index
    )
    if binary_cols:
        df_scaled = pd.concat([df_scaled, df[binary_cols]], axis=1)

    return df_scaled
