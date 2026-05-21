"""Tests for the safety checks used in preflight validations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from app.guardrails.safety_checks import (
    SafetyConfig,
    SafetyStatus,
    check_exchange_available,
    check_market_data_freshness,
    check_market_data_shape,
    check_market_data_values,
    check_model_loaded,
    check_order_inputs,
    run_preflight_safety_checks,
)


def make_market_data(
    *,
    rows: int = 3,
    latest: datetime | None = None,
    volume: float = 10.0,
) -> pd.DataFrame:
    """Create a small valid OHLCV frame for safety tests."""
    latest_timestamp = latest or datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)
    index = pd.date_range(end=latest_timestamp, periods=rows, freq="h")
    return pd.DataFrame(
        {
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.5] * rows,
            "volume": [volume] * rows,
        },
        index=index,
    )


def test_safety_config_rejects_invalid_values() -> None:
    """Safety configuration should reject unusable thresholds."""
    with pytest.raises(ValueError, match="max_market_data_age_seconds"):
        SafetyConfig(max_market_data_age_seconds=0)

    with pytest.raises(ValueError, match="min_market_data_rows"):
        SafetyConfig(min_market_data_rows=0)

    with pytest.raises(ValueError, match="required_ohlcv_columns"):
        SafetyConfig(required_ohlcv_columns=())


def test_check_model_loaded_passes_warns_or_fails() -> None:
    """Model availability should fail only when the model is required."""
    model = object()
    required_missing = check_model_loaded(None, SafetyConfig(require_model_loaded=True))
    optional_missing = check_model_loaded(None, SafetyConfig(require_model_loaded=False))
    available = check_model_loaded(model)

    assert required_missing.status == SafetyStatus.FAILED
    assert required_missing.reason == "model_not_loaded"
    assert optional_missing.status == SafetyStatus.WARNING
    assert optional_missing.passed is True
    assert available.status == SafetyStatus.PASSED
    assert available.metadata["model_type"] == "object"


def test_check_exchange_available_handles_missing_offline_and_online() -> None:
    """Exchange checks should distinguish missing clients from offline mode."""
    missing = check_exchange_available(None)
    offline_allowed = check_exchange_available(SimpleNamespace(offline_mode=True))
    offline_blocked = check_exchange_available(
        SimpleNamespace(offline_mode=True),
        SafetyConfig(allow_offline_exchange=False),
    )
    online = check_exchange_available(SimpleNamespace(offline_mode=False))

    assert missing.status == SafetyStatus.FAILED
    assert missing.reason == "exchange_client_missing"
    assert offline_allowed.status == SafetyStatus.WARNING
    assert offline_allowed.reason == "exchange_in_offline_mode"
    assert offline_blocked.status == SafetyStatus.FAILED
    assert online.status == SafetyStatus.PASSED


def test_check_market_data_shape_validates_dataframe_columns_and_rows() -> None:
    """Market data shape should require a DataFrame, OHLCV columns, and enough rows."""
    config = SafetyConfig(min_market_data_rows=3)
    valid = make_market_data(rows=3)
    not_frame = check_market_data_shape("not-a-frame", config)
    empty = check_market_data_shape(pd.DataFrame(), config)
    missing = check_market_data_shape(valid.drop(columns=["volume"]), config)
    too_short = check_market_data_shape(make_market_data(rows=2), config)
    passed = check_market_data_shape(valid, config)

    assert not_frame.reason == "market_data_must_be_dataframe"
    assert empty.reason == "market_data_empty"
    assert missing.reason == "market_data_missing_required_columns"
    assert missing.metadata["missing_columns"] == ["volume"]
    assert too_short.reason == "market_data_has_insufficient_rows"
    assert passed.status == SafetyStatus.PASSED
    assert passed.metadata["row_count"] == 3


def test_check_market_data_values_rejects_bad_values() -> None:
    """OHLCV values should be numeric, finite, and positive where required."""
    config = SafetyConfig(min_market_data_rows=1)

    non_numeric = make_market_data(rows=1)
    non_numeric.loc[non_numeric.index[0], "close"] = "bad"

    non_finite = make_market_data(rows=1)
    non_finite.loc[non_finite.index[0], "close"] = float("inf")

    non_positive_price = make_market_data(rows=1)
    non_positive_price.loc[non_positive_price.index[0], "low"] = 0

    negative_volume = make_market_data(rows=1, volume=-1)

    assert check_market_data_values(non_numeric, config).reason == "market_data_contains_nan_values"
    assert (
        check_market_data_values(non_finite, config).reason
        == "market_data_contains_non_finite_values"
    )
    assert (
        check_market_data_values(non_positive_price, config).reason
        == "market_data_contains_non_positive_prices"
    )
    assert (
        check_market_data_values(negative_volume, config).reason
        == "market_data_contains_negative_volume"
    )
    assert check_market_data_values(make_market_data(rows=1), config).status == SafetyStatus.PASSED


def test_check_market_data_freshness_handles_datetime_edge_cases() -> None:
    """Freshness should pass fresh data, fail stale data, and warn for odd indexes."""
    now = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)
    config = SafetyConfig(max_market_data_age_seconds=60, min_market_data_rows=1)

    fresh = check_market_data_freshness(
        make_market_data(rows=1, latest=now), now=now, config=config
    )
    stale = check_market_data_freshness(
        make_market_data(rows=1, latest=now - timedelta(seconds=120)),
        now=now,
        config=config,
    )
    future = check_market_data_freshness(
        make_market_data(rows=1, latest=now + timedelta(seconds=1)),
        now=now,
        config=config,
    )
    non_datetime = make_market_data(rows=1)
    non_datetime.index = [1]
    non_datetime_result = check_market_data_freshness(non_datetime, now=now, config=config)

    assert fresh.status == SafetyStatus.PASSED
    assert fresh.reason == "market_data_fresh"
    assert stale.status == SafetyStatus.FAILED
    assert stale.reason == "market_data_stale"
    assert future.status == SafetyStatus.WARNING
    assert future.reason == "market_data_timestamp_in_future"
    assert non_datetime_result.status == SafetyStatus.WARNING
    assert non_datetime_result.reason == "market_data_index_not_datetime"


@pytest.mark.parametrize(
    ("symbol", "quantity", "price", "reason"),
    [
        ("", 1, 100, "symbol_empty"),
        ("BTCUSDT", 0, 100, "quantity_invalid"),
        ("BTCUSDT", float("nan"), 100, "quantity_invalid"),
        ("BTCUSDT", 1, 0, "price_invalid"),
        ("BTCUSDT", 1, float("inf"), "price_invalid"),
    ],
)
def test_check_order_inputs_rejects_invalid_inputs(
    symbol: str,
    quantity: float,
    price: float,
    reason: str,
) -> None:
    """Order input checks should block empty symbols and invalid numeric values."""
    result = check_order_inputs(symbol=symbol, quantity=quantity, price=price)

    assert result.status == SafetyStatus.FAILED
    assert result.reason == reason


def test_check_order_inputs_passes_valid_inputs() -> None:
    """Valid order inputs should be normalized in metadata."""
    result = check_order_inputs(symbol=" btcusdt ", quantity=1.5, price=100)

    assert result.status == SafetyStatus.PASSED
    assert result.metadata == {"symbol": "BTCUSDT", "quantity": 1.5, "price": 100}


def test_run_preflight_safety_checks_passes_with_warnings() -> None:
    """Warnings should be captured without failing the whole preflight report."""
    now = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)
    report = run_preflight_safety_checks(
        model=object(),
        exchange=SimpleNamespace(offline_mode=True),
        market_data=make_market_data(rows=3, latest=now),
        config=SafetyConfig(min_market_data_rows=3, allow_offline_exchange=True),
        now=now,
    )

    assert report.passed is True
    assert report.failures == ()
    assert len(report.warnings) == 1
    assert report.warnings[0].reason == "exchange_in_offline_mode"


def test_run_preflight_safety_checks_fails_when_any_check_fails() -> None:
    """Failed preflight checks should make the aggregate report fail."""
    report = run_preflight_safety_checks(
        model=None,
        exchange=None,
        market_data=pd.DataFrame(),
        config=SafetyConfig(min_market_data_rows=1, require_model_loaded=True),
        now=datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC),
    )

    assert report.passed is False
    assert {failure.reason for failure in report.failures} >= {
        "model_not_loaded",
        "exchange_client_missing",
        "market_data_empty",
    }
