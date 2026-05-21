"""Module: Data Loader to preprocessing historical cryptocurrency OHLCV data for ML model"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd
from pandas import DataFrame

from app.core.config import DATA_FILE_PATH

LOG = logging.getLogger(__name__)

REQUIRED_COLUMNS: Final[set[str]] = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def load_and_preprocess_data(file_path: str | Path = DATA_FILE_PATH) -> DataFrame:
    """
    Load and preprocess historical cryptocurrency OHLCV data.

    Processing steps:
    1. Load CSV
    2. Validate required columns
    3. Convert timestamp to datetime index
    4. Sort chronologically
    5. Forward-fill missing values
    6. Median-impute remaining numeric missing values
    """
    file_path = Path(file_path)

    if not file_path.exists():
        LOG.error(
            "Data file not found at %s. Ensure the file exists or update DATA_FILE_PATH.",
            file_path,
        )
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        df: DataFrame = pd.read_csv(file_path)
    except Exception as exc:
        LOG.exception("Failed to read CSV file: %s", exc)
        raise

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Invalid data schema. Missing required columns: {sorted(missing_cols)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.set_index("timestamp")

    df = cast(DataFrame, df[df.index.notna()])

    if df.empty:
        raise ValueError("All rows have invalid or missing timestamps. No data to process.")

    df.sort_index(inplace=True)

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
