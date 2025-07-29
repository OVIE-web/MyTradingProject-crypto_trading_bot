# src/db.py

import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from src.config import DATABASE_URL


logging.info(f"Database URL: {DATABASE_URL}")

# Base class for declarative models
Base = declarative_base()

class Trade(Base):
    """SQLAlchemy model for storing trade records."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    side = Column(String, nullable=False) # "BUY" or "SELL"
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    order_id = Column(String, nullable=True) # To store Binance order ID
    fill_price = Column(Float, nullable=True) # Actual filled price
    commission = Column(Float, nullable=True)
    commission_asset = Column(String, nullable=True)

    def __repr__(self):
        return f"<Trade(symbol='{self.symbol}', side='{self.side}', quantity={self.quantity}, price={self.price}, timestamp='{self.timestamp}')>"

# Database engine
engine = create_engine(DATABASE_URL) # Ensure DATABASE_URL is defined in src/config.py

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