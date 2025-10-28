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
# --- Pydantic Schemas ---
class TradeBase(BaseModel):
    symbol: str
    side: str
    quantity: float
    price: float

class TradeCreate(TradeBase):
    pass

class TradeRead(TradeBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True

# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@router.post("/", response_model=TradeRead)
def create_trade(trade: TradeCreate, db: Session = Depends(get_db)):
    db_trade = Trade(**trade.dict(), timestamp=datetime.now(UTC))
    db.add(db_trade)
    db.commit()
    db.refresh(db_trade)
    return db_trade


@router.get("/", response_model=List[TradeRead])
def get_trades(db: Session = Depends(get_db)):
    """Retrieve all trade records."""
    return db.query(Trade).all()


@router.get("/{trade_id}", response_model=TradeRead)
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
uvicorn src.main_api:app --reload
