from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

from app.schemas.trade_schema import (
    TradeBase,
    TradeCreate,
    TradeDeleteResponse,
    TradeRead,
    TradeUpdate,
)


def test_trade_base_normalizes_symbol_and_side() -> None:
    trade = TradeBase(
        symbol=" btc/usdt ",
        side="buy",
        quantity=cast(Any, "0.01"),
        price=cast(Any, "68000"),
    )

    assert trade.symbol == "BTCUSDT"
    assert trade.side == "BUY"
    assert trade.quantity == Decimal("0.01")
    assert trade.price == Decimal("68000")


def test_trade_create_uses_shared_trade_validation() -> None:
    trade = TradeCreate(
        symbol="eth-usdt", side="SELL", quantity=Decimal("0.2"), price=Decimal("3500")
    )

    assert trade.symbol == "ETHUSDT"
    assert trade.side == "SELL"


def test_trade_base_rejects_invalid_symbol_side_and_non_positive_numbers() -> None:
    with pytest.raises(ValidationError):
        TradeBase(symbol="??", side="BUY", quantity=cast(Any, "0.01"), price=cast(Any, "68000"))

    with pytest.raises(ValidationError):
        TradeBase(
            symbol="BTCUSDT",
            side="HOLD",
            quantity=cast(Any, "0.01"),
            price=cast(Any, "68000"),
        )

    with pytest.raises(ValidationError):
        TradeBase(symbol="BTCUSDT", side="BUY", quantity=cast(Any, "0"), price=cast(Any, "68000"))

    with pytest.raises(ValidationError):
        TradeBase(symbol="BTCUSDT", side="BUY", quantity=cast(Any, "0.01"), price=cast(Any, "0"))


def test_trade_update_accepts_partial_payloads() -> None:
    empty_update = TradeUpdate()
    update = TradeUpdate(status="FILLED", order_id="order-123", confidence=Decimal("0.88"))

    assert empty_update.status is None
    assert update.status == "FILLED"
    assert update.order_id == "order-123"
    assert update.confidence == Decimal("0.88")


def test_trade_update_bounds_confidence_and_lengths() -> None:
    with pytest.raises(ValidationError):
        TradeUpdate(confidence=Decimal("1.1"))

    with pytest.raises(ValidationError):
        TradeUpdate(status="x" * 21)

    with pytest.raises(ValidationError):
        TradeUpdate(order_id="x" * 81)


def test_trade_read_validates_from_orm_like_object() -> None:
    timestamp = datetime(2026, 5, 16, tzinfo=UTC)
    db_trade = SimpleNamespace(
        id=7,
        symbol="bnb-usdt",
        side="sell",
        quantity=Decimal("0.5"),
        price=Decimal("610"),
        confidence=Decimal("0.73"),
        status="FILLED",
        order_id="binance-7",
        timestamp=timestamp,
    )

    trade = TradeRead.model_validate(db_trade)

    assert trade.id == 7
    assert trade.symbol == "BNBUSDT"
    assert trade.side == "SELL"
    assert trade.timestamp == timestamp


def test_trade_delete_response_exposes_message() -> None:
    response = TradeDeleteResponse(message="trade deleted")

    assert response.message == "trade deleted"
