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
    Drops original 'BB_hband', 'BB_lband', 'BB_mavg', 'RSI', 'SMA_long', 'SMA_short' columns
    if they exist from previous processing.
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
        rsi = RSIIndicator(close=df['close'], window=RSI_WINDOW)
        df['rsi'] = rsi.rsi()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=BB_WINDOW, window_dev=BB_WINDOW_DEV)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        
        # Handle division by zero if bb_upper == bb_lower (e.g., flat line)
        bb_pct_b = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_pct_b'] = bb_pct_b.replace([np.inf, -np.inf], np.nan)

        # Moving Averages
        df['sma_20'] = SMAIndicator(close=df['close'], window=SMA_SHORT_WINDOW).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=SMA_LONG_WINDOW).sma_indicator()
        df['ma_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        # Price momentum
        df['price_momentum'] = df['close'].pct_change(5)  # Example, could be configured
        
        
        # Average True Range
        if df is None or df.empty:
            logging.error("Empty DataFrame passed to feature engineer.")
            return df

        if len(df) < ATR_WINDOW:
           logging.warning(f"Not enough data ({len(df)} rows) for ATR window {ATR_WINDOW}. Skipping ATR.")
           df["atr"] = np.nan
           df["atr_pct"] = np.nan
           return df

           atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=ATR_WINDOW)
           df['atr'] = atr.average_true_range()
           df['atr_pct'] = df['atr'] / df['close']
           ...
           return df

            
        # Volume metrics
        df['volume_pct_change'] = df['volume'].pct_change()
        
        # Remove old duplicate columns from previous processing if they exist
        columns_to_drop_from_input = ['BB_hband', 'BB_lband', 'BB_mavg', 'RSI', 'SMA_long', 'SMA_short']
        df = df.drop(columns=[col for col in columns_to_drop_from_input if col in df.columns])
        
        # Drop rows with NaN values resulting from indicator calculations (e.g., initial rows)
        original_rows = len(df)
        df.dropna(inplace=True)
        dropped_rows = original_rows - len(df)
        if dropped_rows > 0:
            pct_dropped = (dropped_rows / original_rows) * 100
            logging.info(
                f"Calculated technical indicators. Dropped {dropped_rows} rows "
                f"({pct_dropped:.2f}%) due to NaN values from rolling indicators."
            )
        else:
            logging.info("Calculated technical indicators. No rows dropped.")
        
        return df
    except Exception as e:
        logging.error(f'Error calculating technical indicators: {str(e)}')
        raise

def get_rsi_quantile_thresholds(rsi_series, lower_quantile=RSI_LOWER_QUANTILE, upper_quantile=RSI_UPPER_QUANTILE):
    """
    Calculate RSI thresholds based on historical distribution.
    
    Args:
        rsi_series (pd.Series): Series containing RSI values
        lower_quantile (float): Quantile for oversold threshold
        upper_quantile (float): Quantile for overbought threshold
    
    Returns:
        tuple: (lower_threshold, upper_threshold)
    """
    if not isinstance(rsi_series, pd.Series):
        raise TypeError('rsi_series must be a pandas Series')
    
    rsi_clean = rsi_series.dropna()
    if rsi_clean.empty:
        logging.warning('RSI series is empty or contains only NaN values. Using default thresholds.')
        return 30, 70 # Fallback to default
    
    if not (0 < lower_quantile < upper_quantile < 1):
        raise ValueError('Quantiles must be between 0 and 1, and lower must be less than upper')
    
    try:
        lower_threshold = rsi_clean.quantile(lower_quantile)
        upper_threshold = rsi_clean.quantile(upper_quantile)
        
        # Ensure thresholds are within valid RSI range (0-100)
        lower_threshold = max(0, min(100, lower_threshold))
        upper_threshold = max(0, min(100, upper_threshold))
        
        return lower_threshold, upper_threshold
    except Exception as e:
        logging.error(f'Error calculating dynamic RSI thresholds: {str(e)}')
        raise

def apply_rsi_labels(df, rsi_col='rsi', lower_threshold=30, upper_threshold=70):
    """
    Apply trading labels based on RSI values.
    
    Args:
        df (pd.DataFrame): DataFrame containing RSI values
        rsi_col (str): Name of the RSI column
        lower_threshold (float): RSI oversold threshold
        upper_threshold (float): RSI overbought threshold
    
    Returns:
        pd.DataFrame: DataFrame with new 'signal' column (1:buy, 0:hold, -1:sell)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input for applying RSI labels must be a pandas DataFrame')
    if rsi_col not in df.columns:
        raise ValueError(f'Column {rsi_col} not found in DataFrame for RSI labeling')
    
    result = df.copy() # Work on a copy
    
    try:
        result['signal'] = 0 # Initialize signals column with 0 (hold)
        result.loc[result[rsi_col] <= lower_threshold, 'signal'] = 1  # Buy signal
        result.loc[result[rsi_col] >= upper_threshold, 'signal'] = -1 # Sell signal
        result['signal'] = result['signal'].astype(int)
        
        # Verify signal values
        unique_signals = set(result['signal'].unique())
        expected_signals = {-1, 0, 1}
        if not unique_signals.issubset(expected_signals):
            logging.warning(f'Unexpected signal values generated: {unique_signals}. Expected: {expected_signals}')
        
        logging.info(f"Trading signals generated based on RSI thresholds ({lower_threshold:.2f}, {upper_threshold:.2f}).")
        return result
    except Exception as e:
        logging.error(f'Error applying RSI labels: {str(e)}')
        raise

def normalize_features(df):
    """
    Normalize features while preserving binary columns.
    Uses StandardScaler.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input for feature normalization must be a pandas DataFrame')
        
    try:
        # Identify binary columns that should NOT be scaled
        binary_cols = ['ma_cross', 'signal']  # 'signal' is the target, should not be scaled
        binary_cols = [col for col in binary_cols if col in df.columns]
        
        # Store binary values if they exist
        binary_data = df[binary_cols] if binary_cols else None
        
        # Get numeric columns to be normalized, excluding binary and object types
        numeric_cols_to_normalize = df.select_dtypes(include=np.number).columns.difference(binary_cols)
        
        if numeric_cols_to_normalize.empty:
            logging.warning("No numeric columns to normalize found after excluding binary/target columns.")
            return df.copy()  # Return copy if nothing to normalize

        scaler = StandardScaler()
        df_normalized = pd.DataFrame(
            scaler.fit_transform(df[numeric_cols_to_normalize]),
            columns=numeric_cols_to_normalize,
            index=df.index
        )
        
        # Restore binary columns
        if binary_data is not None:
            df_normalized = pd.concat([df_normalized, binary_data], axis=1)
            
        logging.info(
            f"Features normalized successfully. "
            f"Normalized: {list(numeric_cols_to_normalize)}; "
            f"Preserved: {binary_cols if binary_cols else 'None'}"
        )
        return df_normalized
        
    except Exception as e:
        logging.error(f'Error normalizing features: {str(e)}')
        raise