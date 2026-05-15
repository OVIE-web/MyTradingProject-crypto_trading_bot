# app/db/__init__.py
from app.db.database import Base, SessionLocal, engine
from app.db.session import get_db


def init_db() -> None:
    """Initialize database schema without creating model import cycles."""
    from app.db.init_db import init_db as _init_db

    _init_db()


__all__ = ["Base", "SessionLocal", "engine", "init_db", "get_db"]
