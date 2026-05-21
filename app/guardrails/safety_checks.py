from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Any

import pandas as pd


class SafetyStatus(StrEnum):
    """Safety check outcome."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    """Operational safety controls checked before trading."""

    max_market_data_age_seconds: int = 60 * 60 * 6
    min_market_data_rows: int = 50
    required_ohlcv_columns: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    allow_offline_exchange: bool = True
    require_model_loaded: bool = True

    def __post_init__(self) -> None:
        if self.max_market_data_age_seconds < 1:
            raise ValueError("max_market_data_age_seconds must be at least 1.")
        if self.min_market_data_rows < 1:
            raise ValueError("min_market_data_rows must be at least 1.")
        if not self.required_ohlcv_columns:
            raise ValueError("required_ohlcv_columns must not be empty.")


@dataclass(frozen=True, slots=True)
class SafetyCheckResult:
    """Result from an individual safety check."""

    name: str
    status: SafetyStatus
    passed: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SafetyReport:
    """Aggregated safety report for a trading cycle."""

    passed: bool
    results: tuple[SafetyCheckResult, ...]

    @property
    def failures(self) -> tuple[SafetyCheckResult, ...]:
        """Failed checks that should block trading."""
        return tuple(result for result in self.results if result.status == SafetyStatus.FAILED)

    @property
    def warnings(self) -> tuple[SafetyCheckResult, ...]:
        """Non-blocking warnings."""
        return tuple(result for result in self.results if result.status == SafetyStatus.WARNING)


def _passed(name: str, reason: str, metadata: dict[str, Any] | None = None) -> SafetyCheckResult:
    return SafetyCheckResult(
        name=name,
        status=SafetyStatus.PASSED,
        passed=True,
        reason=reason,
        metadata=metadata or {},
    )


def _failed(name: str, reason: str, metadata: dict[str, Any] | None = None) -> SafetyCheckResult:
    return SafetyCheckResult(
        name=name,
        status=SafetyStatus.FAILED,
        passed=False,
        reason=reason,
        metadata=metadata or {},
    )


def _warning(name: str, reason: str, metadata: dict[str, Any] | None = None) -> SafetyCheckResult:
    return SafetyCheckResult(
        name=name,
        status=SafetyStatus.WARNING,
        passed=True,
        reason=reason,
        metadata=metadata or {},
    )


def check_model_loaded(model: Any, config: SafetyConfig | None = None) -> SafetyCheckResult:
    """Ensure a required model object is available."""
    safety_config = config or SafetyConfig()

    if model is None and safety_config.require_model_loaded:
        return _failed("model_loaded", "model_not_loaded")

    if model is None:
        return _warning("model_loaded", "model_not_loaded_but_not_required")

    return _passed("model_loaded", "model_available", {"model_type": type(model).__name__})


def check_exchange_available(
    exchange: Any,
    config: SafetyConfig | None = None,
) -> SafetyCheckResult:
    """Check whether the exchange service is available or safely in offline mode."""
    safety_config = config or SafetyConfig()

    if exchange is None:
        return _failed("exchange_available", "exchange_client_missing")

    offline_mode = bool(getattr(exchange, "offline_mode", False))
    if offline_mode and not safety_config.allow_offline_exchange:
        return _failed("exchange_available", "exchange_in_offline_mode")

    if offline_mode:
        return _warning("exchange_available", "exchange_in_offline_mode")

    return _passed("exchange_available", "exchange_available")


def check_market_data_shape(
    market_data: pd.DataFrame,
    config: SafetyConfig | None = None,
) -> SafetyCheckResult:
    """Validate market data has enough rows and required columns."""
    safety_config = config or SafetyConfig()

    if not isinstance(market_data, pd.DataFrame):
        return _failed("market_data_shape", "market_data_must_be_dataframe")

    if market_data.empty:
        return _failed("market_data_shape", "market_data_empty")

    missing_columns = [
        column
        for column in safety_config.required_ohlcv_columns
        if column not in market_data.columns
    ]
    if missing_columns:
        return _failed(
            "market_data_shape",
            "market_data_missing_required_columns",
            {"missing_columns": missing_columns},
        )

    if len(market_data) < safety_config.min_market_data_rows:
        return _failed(
            "market_data_shape",
            "market_data_has_insufficient_rows",
            {
                "row_count": len(market_data),
                "min_market_data_rows": safety_config.min_market_data_rows,
            },
        )

    return _passed("market_data_shape", "market_data_shape_valid", {"row_count": len(market_data)})


def check_market_data_values(
    market_data: pd.DataFrame,
    config: SafetyConfig | None = None,
) -> SafetyCheckResult:
    """Validate OHLCV values are numeric, finite, and positive where required."""
    safety_config = config or SafetyConfig()
    shape_result = check_market_data_shape(market_data, config=safety_config)
    if not shape_result.passed:
        return shape_result

    numeric_data = market_data.loc[:, list(safety_config.required_ohlcv_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric_data.isna().any().any():
        return _failed("market_data_values", "market_data_contains_nan_values")

    finite_mask = numeric_data.map(lambda value: isfinite(float(value)))
    if not finite_mask.all().all():
        return _failed("market_data_values", "market_data_contains_non_finite_values")

    price_columns = [
        column for column in ("open", "high", "low", "close") if column in numeric_data
    ]
    if (numeric_data[price_columns] <= 0).any().any():
        return _failed("market_data_values", "market_data_contains_non_positive_prices")

    if "volume" in numeric_data and (numeric_data["volume"] < 0).any():
        return _failed("market_data_values", "market_data_contains_negative_volume")

    return _passed("market_data_values", "market_data_values_valid")


def check_market_data_freshness(
    market_data: pd.DataFrame,
    *,
    now: datetime | None = None,
    config: SafetyConfig | None = None,
) -> SafetyCheckResult:
    """Validate the latest market data timestamp is recent enough."""
    safety_config = config or SafetyConfig()

    if not isinstance(market_data, pd.DataFrame) or market_data.empty:
        return _failed("market_data_freshness", "market_data_empty")

    if not isinstance(market_data.index, pd.DatetimeIndex):
        return _warning("market_data_freshness", "market_data_index_not_datetime")

    latest_timestamp = market_data.index.max()
    if latest_timestamp.tzinfo is None:
        latest_time = latest_timestamp.to_pydatetime().replace(tzinfo=UTC)
    else:
        latest_time = latest_timestamp.to_pydatetime().astimezone(UTC)

    current_time = now or datetime.now(UTC)
    age_seconds = (current_time - latest_time).total_seconds()

    if age_seconds < 0:
        return _warning(
            "market_data_freshness",
            "market_data_timestamp_in_future",
            {"age_seconds": age_seconds},
        )

    if age_seconds > safety_config.max_market_data_age_seconds:
        return _failed(
            "market_data_freshness",
            "market_data_stale",
            {
                "age_seconds": age_seconds,
                "max_market_data_age_seconds": safety_config.max_market_data_age_seconds,
            },
        )

    return _passed(
        "market_data_freshness",
        "market_data_fresh",
        {"age_seconds": age_seconds},
    )


def check_order_inputs(
    *,
    symbol: str,
    quantity: float,
    price: float,
) -> SafetyCheckResult:
    """Validate raw order inputs before limit/risk checks."""
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        return _failed("order_inputs", "symbol_empty")

    if not isfinite(float(quantity)) or quantity <= 0:
        return _failed("order_inputs", "quantity_invalid", {"quantity": quantity})

    if not isfinite(float(price)) or price <= 0:
        return _failed("order_inputs", "price_invalid", {"price": price})

    return _passed(
        "order_inputs",
        "order_inputs_valid",
        {"symbol": normalized_symbol, "quantity": quantity, "price": price},
    )


def run_preflight_safety_checks(
    *,
    model: Any,
    exchange: Any,
    market_data: pd.DataFrame,
    config: SafetyConfig | None = None,
    now: datetime | None = None,
) -> SafetyReport:
    """Run the standard pre-trade safety checks for a trading cycle."""
    safety_config = config or SafetyConfig()
    results = (
        check_model_loaded(model, config=safety_config),
        check_exchange_available(exchange, config=safety_config),
        check_market_data_shape(market_data, config=safety_config),
        check_market_data_values(market_data, config=safety_config),
        check_market_data_freshness(market_data, now=now, config=safety_config),
    )

    return SafetyReport(
        passed=not any(result.status == SafetyStatus.FAILED for result in results),
        results=results,
    )
