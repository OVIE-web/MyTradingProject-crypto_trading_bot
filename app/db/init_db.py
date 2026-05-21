# app/db/init_db.py
from __future__ import annotations

import logging

from app.db.database import Base, engine

LOG = logging.getLogger(__name__)


def import_models() -> None:
    """Import ORM models so SQLAlchemy registers them before schema operations."""
    import app.models.prediction  # noqa: F401
    import app.models.trade  # noqa: F401
    import app.models.user  # noqa: F401


def init_db() -> None:
    """Initialize database schema."""
    import_models()
    LOG.info("Creating database tables if they do not exist...")
    Base.metadata.create_all(bind=engine)
    LOG.info("Database tables ready.")
