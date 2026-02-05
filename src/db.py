# src/db.py
from __future__ import annotations

import logging
from collections.abc import Generator
from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, Numeric, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.settings import settings

LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# SQLAlchemy Base
# --------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Base class for all ORM models."""


# --------------------------------------------------------------------------
# ORM Models
# --------------------------------------------------------------------------


class Trade(Base):
    """ORM model for storing trade records."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    order_id: Mapped[str | None] = mapped_column(String(50), unique=True)
    fill_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
    commission: Mapped[float | None] = mapped_column(Numeric(20, 8))
    commission_asset: Mapped[str | None] = mapped_column(String(10))


# --------------------------------------------------------------------------
# Engine & Session
# --------------------------------------------------------------------------

engine: Engine = create_engine(
    settings.DATABASE_URL,
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

# --------------------------------------------------------------------------
# Initial DataBase Setup
# --------------------------------------------------------------------------

def init_db() -> None:
    """
    Intialize database schema.
    Used by tests and applications startup

    """
    Base.metadata.create_all(bind=engine)

# --------------------------------------------------------------------------
# Session helpers
# --------------------------------------------------------------------------


def get_db() -> Generator[Session, None, None]:
    """
    Provide a transactional database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



