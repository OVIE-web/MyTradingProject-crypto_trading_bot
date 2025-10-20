# src/db.py

import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, UTC

from src.config import DATABASE_URL

# Mask sensitive info in logs
safe_url = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
logging.info(f"Connecting to database at: {safe_url}")

# Base class for declarative models
Base = declarative_base()

class Trade(Base):
    """SQLAlchemy model for storing trade records."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    side = Column(String, nullable=False)  # "BUY" or "SELL"
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    order_id = Column(String, nullable=True)  # Binance order ID
    fill_price = Column(Float, nullable=True)  # Actual filled price
    commission = Column(Float, nullable=True)
    commission_asset = Column(String, nullable=True)

    def __repr__(self):
        return (
            f"<Trade(symbol='{self.symbol}', side='{self.side}', "
            f"quantity={self.quantity}, price={self.price}, timestamp='{self.timestamp}')>"
        )

# Database engine
engine = create_engine(DATABASE_URL, future=True)

# SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes the database by creating all tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logging.info("Database tables created or already exist.")
    except Exception as e:
        logging.critical(f"Failed to initialize database: {e}", exc_info=True)
        raise