from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import DateTime, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, synonym

from app.db.database import Base


class Trade(Base):
    """ORM model for persisted trade records."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)

    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    qty = synonym("quantity")

    price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)

    status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    order_id: Mapped[str | None] = mapped_column(String(80), unique=True, nullable=True)

    fill_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    commission: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    commission_asset: Mapped[str | None] = mapped_column(String(10), nullable=True)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
