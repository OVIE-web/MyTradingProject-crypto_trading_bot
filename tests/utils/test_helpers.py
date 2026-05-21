from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.utils.helpers import chunked, clamp, ensure_utc, quantize_decimal, to_decimal, utc_now


def test_utc_now_returns_timezone_aware_utc_datetime() -> None:
    current_time = utc_now()

    assert current_time.tzinfo is UTC


def test_ensure_utc_adds_utc_to_naive_datetime() -> None:
    naive = datetime(2026, 5, 16, 12, 30)

    result = ensure_utc(naive)

    assert result == datetime(2026, 5, 16, 12, 30, tzinfo=UTC)


def test_ensure_utc_converts_aware_datetime_to_utc() -> None:
    lagos_time = datetime(2026, 5, 16, 13, 30, tzinfo=timezone(timedelta(hours=1)))

    result = ensure_utc(lagos_time)

    assert result == datetime(2026, 5, 16, 12, 30, tzinfo=UTC)


def test_to_decimal_converts_supported_values() -> None:
    assert to_decimal("1.25") == Decimal("1.25")
    assert to_decimal(10) == Decimal("10")
    assert to_decimal(Decimal("0.5")) == Decimal("0.5")


def test_to_decimal_raises_clear_validation_error() -> None:
    with pytest.raises(ValueError, match="price must be a valid decimal number"):
        to_decimal("not-a-number", field_name="price")


def test_quantize_decimal_uses_half_up_rounding() -> None:
    assert quantize_decimal("1.234567895") == Decimal("1.23456790")
    assert quantize_decimal("1.235", places="0.01") == Decimal("1.24")


def test_clamp_limits_values_to_range() -> None:
    assert clamp(0.5, 0.0, 1.0) == 0.5
    assert clamp(-1.0, 0.0, 1.0) == 0.0
    assert clamp(2.0, 0.0, 1.0) == 1.0


def test_clamp_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="minimum must be less than or equal to maximum"):
        clamp(1.0, 2.0, 1.0)


def test_chunked_yields_fixed_size_chunks() -> None:
    chunks = list(chunked([1, 2, 3, 4, 5], 2))

    assert chunks == [[1, 2], [3, 4], [5]]


def test_chunked_accepts_generators() -> None:
    chunks = list(chunked((value for value in range(4)), 3))

    assert chunks == [[0, 1, 2], [3]]


def test_chunked_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError, match="size must be at least 1"):
        list(chunked([1, 2, 3], 0))
