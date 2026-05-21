# app/api/routes/trades.py
from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.models.trade import Trade
from app.schemas.trade_schema import TradeCreate, TradeDeleteResponse, TradeRead

router = APIRouter()


# --------------------------------------------------------------------------
# Database Dependency
# --------------------------------------------------------------------------
def get_db() -> Generator[Session, Any, None]:
    """Provide a transactional scope for DB operations."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@router.post("/", response_model=TradeRead)
def create_trade(trade: TradeCreate, db: Session = Depends(get_db)) -> Trade:
    """Create a new trade record."""
    db_trade = Trade(**trade.model_dump(), timestamp=datetime.now(UTC))
    db.add(db_trade)
    db.commit()
    db.refresh(db_trade)
    return db_trade


@router.get("/", response_model=list[TradeRead])
def get_trades(db: Session = Depends(get_db)) -> list[Trade]:
    """Retrieve all trade records."""
    return cast(list[Trade], db.query(Trade).all())


@router.get("/{trade_id}", response_model=TradeRead)
def get_trade(trade_id: int, db: Session = Depends(get_db)) -> Trade:
    """Retrieve a single trade by ID."""
    trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    return cast(Trade, trade)


@router.delete("/{trade_id}", response_model=TradeDeleteResponse)
def delete_trade(trade_id: int, db: Session = Depends(get_db)) -> TradeDeleteResponse:
    """Delete a trade by ID."""
    trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    db.delete(trade)
    db.commit()
    return TradeDeleteResponse(message=f"Trade {trade_id} deleted successfully.")
