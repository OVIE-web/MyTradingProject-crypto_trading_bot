from __future__ import annotations

import app.utils as utils
from app.utils.helpers import chunked, clamp, ensure_utc, quantize_decimal, to_decimal, utc_now
from app.utils.validators import (
    ensure_required_columns,
    normalize_optional_symbol,
    normalize_symbol,
    normalize_trade_side,
    validate_binance_interval,
    validate_positive_decimal,
    validate_probability,
)


def test_utils_package_exports_public_api() -> None:
    assert utils.__all__ == [
        "chunked",
        "clamp",
        "ensure_required_columns",
        "ensure_utc",
        "normalize_optional_symbol",
        "normalize_symbol",
        "normalize_trade_side",
        "quantize_decimal",
        "to_decimal",
        "utc_now",
        "validate_binance_interval",
        "validate_positive_decimal",
        "validate_probability",
    ]


def test_utils_package_exports_match_source_modules() -> None:
    assert utils.chunked is chunked
    assert utils.clamp is clamp
    assert utils.ensure_required_columns is ensure_required_columns
    assert utils.ensure_utc is ensure_utc
    assert utils.normalize_optional_symbol is normalize_optional_symbol
    assert utils.normalize_symbol is normalize_symbol
    assert utils.normalize_trade_side is normalize_trade_side
    assert utils.quantize_decimal is quantize_decimal
    assert utils.to_decimal is to_decimal
    assert utils.utc_now is utc_now
    assert utils.validate_binance_interval is validate_binance_interval
    assert utils.validate_positive_decimal is validate_positive_decimal
    assert utils.validate_probability is validate_probability
