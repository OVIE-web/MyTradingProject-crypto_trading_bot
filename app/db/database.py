# app/db/database.py
from __future__ import annotations

import logging

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import DATABASE_URL

LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# SQLAlchemy Base
# --------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Base class for all ORM models."""


# --------------------------------------------------------------------------
# Engine & Session
# --------------------------------------------------------------------------

engine: Engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    class_=Session,
)
