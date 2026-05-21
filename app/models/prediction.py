from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class Prediction(Base):
    """ORM model for storing model prediction records."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)

    features: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    model_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    symbol: Mapped[str | None] = mapped_column(String(20), index=True, nullable=True)
    source: Mapped[str | None] = mapped_column(String(80), nullable=True)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
