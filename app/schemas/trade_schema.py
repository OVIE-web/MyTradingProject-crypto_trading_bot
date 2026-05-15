from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.utils.validators import normalize_symbol as normalize_trade_symbol
from app.utils.validators import normalize_trade_side


class TradeBase(BaseModel):
    """Shared trade schema fields."""

    symbol: str = Field(..., min_length=3, max_length=20, examples=["BTCUSDT"])
    side: str = Field(..., examples=["BUY"])
    quantity: Decimal = Field(..., gt=0, examples=["0.001"])
    price: Decimal = Field(..., gt=0, examples=["68000.00"])

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return normalize_trade_symbol(value)

    @field_validator("side")
    @classmethod
    def normalize_side(cls, value: str) -> str:
        return normalize_trade_side(value)


class TradeCreate(TradeBase):
    """Schema for creating a new trade entry."""


class TradeUpdate(BaseModel):
    """Schema for partial trade updates."""

    status: str | None = Field(default=None, max_length=20)
    order_id: str | None = Field(default=None, max_length=80)
    confidence: Decimal | None = Field(default=None, ge=0, le=1)


class TradeRead(TradeBase):
    """Schema for returning trade records."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    confidence: Decimal | None = None
    status: str | None = None
    order_id: str | None = None
    timestamp: datetime


class TradeDeleteResponse(BaseModel):
    """Response returned after deleting a trade."""

    message: str
