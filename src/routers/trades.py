# src/routers/trades.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

from src.db import SessionLocal, Trade

router = APIRouter()

# --------------------------------------------------------------------------
# Dependency to get DB session
# --------------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --------------------------------------------------------------------------
# Pydantic Schemas
# --------------------------------------------------------------------------
class TradeBase(BaseModel):
    symbol: str = Field(..., example="BTCUSDT")
    side: str = Field(..., example="BUY")
    quantity: float = Field(..., gt=0, example=0.01)
    price: float = Field(..., gt=0, example=65000.0)

class TradeCreate(TradeBase):
    pass

class TradeResponse(TradeBase):
    id: int
    timestamp: datetime

    class Config:
        orm_mode = True

# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@router.post("/", response_model=TradeResponse)
def create_trade(trade: TradeCreate, db: Session = Depends(get_db)):
    """Create a new trade record."""
    db_trade = Trade(
        symbol=trade.symbol,
        side=trade.side,
        quantity=trade.quantity,
        price=trade.price,
    )
    db.add(db_trade)
    db.commit()
    db.refresh(db_trade)
    return db_trade


@router.get("/", response_model=List[TradeResponse])
def get_trades(db: Session = Depends(get_db)):
    """Retrieve all trade records."""
    return db.query(Trade).all()


@router.get("/{trade_id}", response_model=TradeResponse)
def get_trade(trade_id: int, db: Session = Depends(get_db)):
    """Retrieve a single trade by ID."""
    trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    return trade


@router.delete("/{trade_id}")
def delete_trade(trade_id: int, db: Session = Depends(get_db)):
    """Delete a trade by ID."""
    trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    db.delete(trade)
    db.commit()
    return {"message": f"Trade {trade_id} deleted successfully."}
