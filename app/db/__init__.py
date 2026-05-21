# app/db/__init__.py
from __future__ import annotations

from typing import Any

from sqlalchemy.engine import Engine

from app.db.database import LOG, Base, SessionLocal, engine
from app.db.init_db import init_db
from app.db.session import get_db

__all__ = ["Base", "Engine", "LOG", "SessionLocal", "Trade", "engine", "init_db", "get_db"]


def __getattr__(name: str) -> Any:
    """Lazily expose ORM models without creating package import cycles."""
    if name == "Trade":
        from app.models.trade import Trade

        return Trade

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
