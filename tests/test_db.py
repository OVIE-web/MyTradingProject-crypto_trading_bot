"""Tests for database models and operations."""

from datetime import UTC, datetime
from decimal import Decimal

from _pytest.logging import LogCaptureFixture
from sqlalchemy.orm import Session

from src.db import (  # Import engine for drop/create_all in tests
    Trade,
)


def test_init_db(db_session: Session, caplog: LogCaptureFixture) -> None:
    """Tests that the database session is properly initialized."""
    # The db_session fixture already creates all tables in in-memory SQLite
    assert db_session is not None
    result = db_session.query(Trade).all()
    assert result == []  # Should be empty, proving tables exist


def test_trade_model_creation_and_retrieval(db_session: Session) -> None:
    """Tests creating and retrieving a Trade record."""

    new_trade = Trade(
        symbol="BTCUSDT",
        side="BUY",
        quantity=Decimal("0.00100000"),
        price=Decimal("25000.00000000"),
        timestamp=datetime.now(UTC),
        order_id="test_order_123",
        fill_price=Decimal("25000.10000000"),
        commission=Decimal("0.00010000"),
        commission_asset="BNB",
    )
    db_session.add(new_trade)
    db_session.commit()

    retrieved_trade = db_session.query(Trade).filter_by(symbol="BTCUSDT").first()
    assert retrieved_trade is not None
    assert retrieved_trade.symbol == "BTCUSDT"
    assert retrieved_trade.side == "BUY"
    assert retrieved_trade.quantity == Decimal("0.00100000")
    assert retrieved_trade.price == Decimal("25000.00000000")
    assert retrieved_trade.order_id == "test_order_123"
    assert retrieved_trade.fill_price == Decimal("25000.10000000")
    assert retrieved_trade.commission == Decimal("0.00010000")
    assert retrieved_trade.commission_asset == "BNB"


def test_trade_model_nullable_fields(db_session: Session) -> None:
    """Tests creating a Trade record with optional fields as None."""

    trade_without_details = Trade(
        symbol="ETHUSDT",
        side="SELL",
        quantity=Decimal("0.00500000"),
        price=Decimal("2000.00000000"),
        timestamp=datetime.now(UTC),
    )
    db_session.add(trade_without_details)
    db_session.commit()

    retrieved_trade = db_session.query(Trade).filter_by(symbol="ETHUSDT").first()
    assert retrieved_trade is not None
    assert retrieved_trade.order_id is None
    assert retrieved_trade.fill_price is None
    assert retrieved_trade.commission is None
    assert retrieved_trade.commission_asset is None
