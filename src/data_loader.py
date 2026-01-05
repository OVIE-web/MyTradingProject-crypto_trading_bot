# src/data_loader.py

import logging
import os
from typing import Final

import numpy as np
import pandas as pd
from pandas import DataFrame

LOG = logging.getLogger(__name__)

DATA_FILE_PATH: Final[str] = os.path.join("data", "test_df_features.csv")

REQUIRED_COLUMNS: Final[set[str]] = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def load_and_preprocess_data(file_path: str) -> DataFrame:
    """
    Load and preprocess historical cryptocurrency price data.

    Contract:
        - Always returns a valid pandas DataFrame
        - Raises if file is missing or schema is invalid

    Processing steps:
        1. Load CSV
        2. Validate schema
        3. Convert timestamp → datetime index
        4. Sort chronologically
        5. Forward-fill missing values
        6. Median-impute numeric columns
    """
    try:
        df: DataFrame = pd.read_csv(file_path, sep=",")
    except FileNotFoundError:
        LOG.error(
            "Data file not found at %s. Ensure the file exists or update DATA_FILE_PATH.",
            file_path,
        )
        raise
    except Exception as exc:
        LOG.exception("Failed to read CSV file: %s", exc)
        raise

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Invalid data schema. Missing required columns: {sorted(missing_cols)}")

    # Convert timestamp and set index
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.set_index("timestamp", inplace=True)

    # Drop rows with invalid timestamps
    df = df[~df.index.isna()]

    # If all rows were dropped, raise an error
    if df.empty:
        raise ValueError("All rows have invalid or missing timestamps. No data to process.")

    # Sort chronologically
    df.sort_index(inplace=True)

    # Handle missing values
    missing = df.isna().sum()
    if missing.any():
        LOG.warning(
            "Missing values detected:\n%s\nApplying forward-fill and median imputation.",
            missing[missing > 0],
        )

        df.ffill(inplace=True)

        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    LOG.info("Data loaded and preprocessed successfully. Shape=%s", df.shape)
    return df
