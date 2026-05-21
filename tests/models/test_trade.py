"""Unit tests for the Trade model."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.trade import Trade


def test_trade_model_persists_required_and_optional_fields(model_session: Session) -> None:
    """Trade rows should preserve core order and execution metadata."""
    trade = Trade(
        symbol="BTCUSDT",
        side="BUY",
        quantity=Decimal("0.125"),
        price=Decimal("65000.12345678"),
        confidence=Decimal("0.987654"),
        status="FILLED",
        order_id="order-123",
        fill_price=Decimal("65001.00000000"),
        commission=Decimal("0.00010000"),
        commission_asset="BTC",
    )

    model_session.add(trade)
    model_session.commit()
    model_session.refresh(trade)

    loaded = model_session.query(Trade).filter_by(order_id="order-123").one()

    assert loaded.id is not None
    assert loaded.symbol == "BTCUSDT"
    assert loaded.side == "BUY"
    assert loaded.quantity == Decimal("0.12500000")
    assert loaded.qty == loaded.quantity
    assert loaded.price == Decimal("65000.12345678")
    assert loaded.confidence == Decimal("0.987654")
    assert loaded.status == "FILLED"
    assert loaded.fill_price == Decimal("65001.00000000")
    assert loaded.commission == Decimal("0.00010000")
    assert loaded.commission_asset == "BTC"
    assert isinstance(loaded.timestamp, datetime)


def test_trade_qty_synonym_updates_quantity() -> None:
    """The legacy qty synonym should stay wired to quantity."""
    trade = Trade(symbol="ETHUSDT", side="SELL", quantity=Decimal("1"), price=Decimal("10"))

    trade.qty = Decimal("2.5")

    assert trade.quantity == Decimal("2.5")


def test_trade_order_id_is_unique(model_session: Session) -> None:
    """Duplicate exchange order IDs should be rejected."""
    first = Trade(symbol="BTCUSDT", side="BUY", quantity=Decimal("1"), price=Decimal("100"))
    second = Trade(symbol="ETHUSDT", side="SELL", quantity=Decimal("1"), price=Decimal("100"))
    first.order_id = "duplicate-order"
    second.order_id = "duplicate-order"

    model_session.add_all([first, second])

    with pytest.raises(IntegrityError):
        model_session.commit()
