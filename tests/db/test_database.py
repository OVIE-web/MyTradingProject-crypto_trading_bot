"""Tests for the database module, ensuring that the SQLAlchemy engine, session factory, and metadata are correctly configured and functional."""

from __future__ import annotations

from collections.abc import Generator
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.db import database
from app.models.trade import Trade


@pytest.fixture
def sqlite_session() -> Generator[Session, None, None]:
    """Create an isolated in-memory database for ORM tests."""
    engine = create_engine("sqlite:///:memory:", future=True)
    database.Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = TestingSessionLocal()

    try:
        yield session
    finally:
        session.close()
        database.Base.metadata.drop_all(bind=engine)
        engine.dispose()


def test_database_engine_is_configured() -> None:
    """The shared database module should expose a configured SQLAlchemy engine."""
    assert isinstance(database.engine, Engine)


def test_session_factory_uses_expected_defaults() -> None:
    """SessionLocal should use explicit transaction-friendly defaults."""
    assert database.SessionLocal.kw["autoflush"] is False
    assert database.SessionLocal.kw["autocommit"] is False
    assert database.SessionLocal.kw["bind"] is database.engine


def test_base_metadata_registers_trade_table(sqlite_session: Session) -> None:
    """The SQLAlchemy metadata should know about the trades table."""
    inspector = inspect(sqlite_session.get_bind())

    assert "trades" in inspector.get_table_names()


def test_trade_can_be_persisted_and_loaded(sqlite_session: Session) -> None:
    """A Trade model should round-trip through the configured metadata."""
    trade = Trade(
        symbol="BTCUSDT",
        side="BUY",
        quantity=Decimal("0.001"),
        price=Decimal("30000.50"),
        confidence=Decimal("0.950000"),
    )

    sqlite_session.add(trade)
    sqlite_session.commit()
    sqlite_session.refresh(trade)

    loaded = sqlite_session.query(Trade).filter_by(symbol="BTCUSDT").one()

    assert loaded.id is not None
    assert loaded.side == "BUY"
    assert loaded.quantity == Decimal("0.00100000")
    assert loaded.qty == loaded.quantity
    assert loaded.price == Decimal("30000.50000000")
    assert loaded.confidence == Decimal("0.950000")
    assert loaded.timestamp is not None
