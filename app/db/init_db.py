# app/db/init_db.py
from __future__ import annotations

import logging

from app.db.database import Base, engine

# Import ORM models so SQLAlchemy registers them before create_all()
from app.models.prediction import Prediction  # noqa: F401
from app.models.trade import Trade  # noqa: F401
from app.models.user import User  # noqa: F401

LOG = logging.getLogger(__name__)


def init_db() -> None:
    """Initialize database schema."""
    LOG.info("Creating database tables if they do not exist...")
    Base.metadata.create_all(bind=engine)
    LOG.info("Database tables ready.")
