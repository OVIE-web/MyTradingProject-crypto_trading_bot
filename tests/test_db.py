import pytest
import logging
from datetime import datetime
from sqlalchemy.exc import OperationalError

from src.db import init_db, SessionLocal, Trade, Base, engine  # Import engine for drop/create_all in tests


def test_init_db(db_session, caplog):
    """Tests database initialization (table creation)."""
    with caplog.at_level(logging.INFO):
        init_db()
        assert "Database tables created or already exist." in caplog.text


def test_trade_model_creation_and_retrieval(db_session):
    """Tests creating and retrieving a Trade record."""
    new_trade = Trade(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.001,
        price=25000.0,
        timestamp=datetime.utcnow(),
        order_id="test_order_123",
        fill_price=25000.1,
        commission=0.0001,
        commission_asset="BNB"
    )
    db_session.add(new_trade)
    db_session.commit()

    retrieved_trade = db_session.query(Trade).filter_by(symbol="BTCUSDT").first()
    assert retrieved_trade is not None
    assert retrieved_trade.symbol == "BTCUSDT"
    assert retrieved_trade.side == "BUY"
    assert retrieved_trade.quantity == 0.001
    assert retrieved_trade.price == 25000.0
    assert retrieved_trade.order_id == "test_order_123"
    assert retrieved_trade.fill_price == 25000.1
    assert retrieved_trade.commission == 0.0001
    assert retrieved_trade.commission_asset == "BNB"


def test_trade_model_nullable_fields(db_session):
    """Tests creating a Trade record with optional fields as None."""
    trade_without_details = Trade(
        symbol="ETHUSDT",
        side="SELL",
        quantity=0.005,
        price=2000.0,
        timestamp=datetime.utcnow()
    )
    db_session.add(trade_without_details)
    db_session.commit()

    retrieved_trade = db_session.query(Trade).filter_by(symbol="ETHUSDT").first()
    assert retrieved_trade is not None
    assert retrieved_trade.order_id is None
    assert retrieved_trade.fill_price is None
    assert retrieved_trade.commission is None
    assert retrieved_trade.commission_asset is None
