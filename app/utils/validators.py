"""Reusable validation and normalization helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from decimal import Decimal

import pandas as pd

from app.utils.helpers import to_decimal

VALID_TRADE_SIDES = {"BUY", "SELL"}
VALID_BINANCE_INTERVALS = {
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
}
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{3,20}$")


def normalize_symbol(symbol: str) -> str:
    """Normalize exchange symbols for consistent comparisons."""
    normalized = symbol.strip().upper().replace("/", "").replace("-", "")
    if not normalized:
        raise ValueError("symbol must not be empty")
    if not SYMBOL_PATTERN.fullmatch(normalized):
        raise ValueError("symbol must contain 3-20 uppercase letters or numbers")
    return normalized


def normalize_optional_symbol(symbol: str | None) -> str | None:
    """Normalize an optional symbol, preserving None."""
    if symbol is None:
        return None
    return normalize_symbol(symbol)


def normalize_trade_side(side: str) -> str:
    """Normalize and validate a trade side."""
    normalized = side.strip().upper()
    if normalized not in VALID_TRADE_SIDES:
        raise ValueError("side must be BUY or SELL")
    return normalized


def validate_positive_decimal(value: object, *, field_name: str = "value") -> Decimal:
    """Validate that a value is a positive Decimal."""
    decimal_value = to_decimal(value, field_name=field_name)
    if decimal_value <= 0:
        raise ValueError(f"{field_name} must be greater than 0")
    return decimal_value


def validate_probability(value: float, *, field_name: str = "value") -> float:
    """Validate that a value is between 0 and 1."""
    numeric_value = float(value)
    if not 0 <= numeric_value <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return numeric_value


def validate_binance_interval(interval: str) -> str:
    """Validate a Binance candlestick interval."""
    normalized = interval.strip()
    if normalized not in VALID_BINANCE_INTERVALS:
        raise ValueError(f"interval must be one of: {', '.join(sorted(VALID_BINANCE_INTERVALS))}")
    return normalized


def ensure_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Ensure a DataFrame contains all required columns."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
