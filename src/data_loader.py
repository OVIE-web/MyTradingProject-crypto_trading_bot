# src/data_loader.py

import pandas as pd
import logging
import numpy as np


def load_and_preprocess_data(file_path):  # file_path is now passed as argument
    """
    Loads historical cryptocurrency price data, converts timestamp,
    sets index, sorts, and handles missing values.
    """
    try:
        df = pd.read_csv(file_path)

        # Convert timestamp to datetime and set as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        else:
            expected = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            logging.warning(
                f"No 'timestamp' column found. Ensure data is time-indexed. "
                f"Expected schema includes: {expected}"
            )

        # Sort by index to ensure chronological order
        df = df.sort_index()

        # Handle missing values
        missing_count = df.isnull().sum()
        if missing_count.any():
            logging.warning(
                f"Found missing values:\n{missing_count[missing_count > 0]}\n"
                "→ Filling forward, then imputing numeric columns with median."
            )
            df = df.ffill()  # modern replacement for fillna(method='ffill')
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    except FileNotFoundError:
        logging.error(
            f"Data file not found at {file_path}. "
            "Ensure the file exists or update DATA_FILE_PATH in config.py"
        )
        raise
    except Exception as e:
        logging.error(f"Error loading or preprocessing data: {str(e)}")
        raise