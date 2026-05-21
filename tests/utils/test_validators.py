from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from app.utils.validators import (
    ensure_required_columns,
    normalize_optional_symbol,
    normalize_symbol,
    normalize_trade_side,
    validate_binance_interval,
    validate_positive_decimal,
    validate_probability,
)


def test_normalize_symbol_strips_separators_and_uppercases() -> None:
    assert normalize_symbol(" btc/usdt ") == "BTCUSDT"
    assert normalize_symbol("eth-usdt") == "ETHUSDT"


def test_normalize_symbol_rejects_blank_or_invalid_symbols() -> None:
    with pytest.raises(ValueError, match="symbol must not be empty"):
        normalize_symbol("   ")

    with pytest.raises(ValueError, match="symbol must contain 3-20 uppercase letters or numbers"):
        normalize_symbol("bt")

    with pytest.raises(ValueError, match="symbol must contain 3-20 uppercase letters or numbers"):
        normalize_symbol("BTC_USDT")


def test_normalize_optional_symbol_preserves_none() -> None:
    assert normalize_optional_symbol(None) is None
    assert normalize_optional_symbol(" sol/usdt ") == "SOLUSDT"


def test_normalize_trade_side_uppercases_valid_sides() -> None:
    assert normalize_trade_side(" buy ") == "BUY"
    assert normalize_trade_side("SELL") == "SELL"


def test_normalize_trade_side_rejects_unsupported_side() -> None:
    with pytest.raises(ValueError, match="side must be BUY or SELL"):
        normalize_trade_side("hold")


def test_validate_positive_decimal_returns_decimal() -> None:
    assert validate_positive_decimal("10.25", field_name="quantity") == Decimal("10.25")


def test_validate_positive_decimal_rejects_zero_negative_and_invalid_values() -> None:
    with pytest.raises(ValueError, match="quantity must be greater than 0"):
        validate_positive_decimal("0", field_name="quantity")

    with pytest.raises(ValueError, match="quantity must be greater than 0"):
        validate_positive_decimal("-1", field_name="quantity")

    with pytest.raises(ValueError, match="quantity must be a valid decimal number"):
        validate_positive_decimal("bad", field_name="quantity")


def test_validate_probability_accepts_values_between_zero_and_one() -> None:
    assert validate_probability(0.0, field_name="confidence") == 0.0
    assert validate_probability(0.42, field_name="confidence") == 0.42
    assert validate_probability(1.0, field_name="confidence") == 1.0


def test_validate_probability_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        validate_probability(-0.1, field_name="confidence")

    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        validate_probability(1.1, field_name="confidence")


def test_validate_binance_interval_accepts_supported_intervals() -> None:
    assert validate_binance_interval(" 1h ") == "1h"
    assert validate_binance_interval("1M") == "1M"


def test_validate_binance_interval_rejects_unsupported_interval() -> None:
    with pytest.raises(ValueError, match="interval must be one of"):
        validate_binance_interval("2y")


def test_ensure_required_columns_accepts_complete_dataframe() -> None:
    df = pd.DataFrame({"open": [1], "high": [2], "low": [0], "close": [1], "volume": [100]})

    ensure_required_columns(df, ["open", "high", "low", "close", "volume"])


def test_ensure_required_columns_reports_missing_columns() -> None:
    df = pd.DataFrame({"open": [1], "close": [1]})

    with pytest.raises(ValueError, match="Missing required columns"):
        ensure_required_columns(df, ["open", "high", "low", "close"])
