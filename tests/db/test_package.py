"""Tests for the app.db package."""

from __future__ import annotations

from sqlalchemy.engine import Engine

import app.db as db_package
from app.db import database
from app.db.session import get_db
from app.models.trade import Trade


def test_db_package_re_exports_common_database_objects() -> None:
    """app.db should provide ergonomic imports for common DB primitives."""
    assert db_package.Base is database.Base
    assert db_package.SessionLocal is database.SessionLocal
    assert db_package.engine is database.engine
    assert db_package.get_db is get_db
    assert db_package.Trade is Trade
    assert db_package.Engine is Engine


def test_db_package_all_matches_public_exports() -> None:
    """__all__ should document the intended public package surface."""
    assert set(db_package.__all__) == {
        "Base",
        "Engine",
        "LOG",
        "SessionLocal",
        "Trade",
        "engine",
        "get_db",
        "init_db",
    }
