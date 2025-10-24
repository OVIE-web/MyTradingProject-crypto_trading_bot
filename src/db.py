# src/db.py
"""
Database initialization and SQLAlchemy ORM models.
"""

import logging
from datetime import datetime, UTC
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from src.config import DATABASE_URL

# --------------------------------------------------------------------------
# 1️⃣ Configure logging
# --------------------------------------------------------------------------
safe_url = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.info(f"📦 Connecting to database at: {safe_url}")

# --------------------------------------------------------------------------
# 2️⃣ Base model for all ORM classes
# --------------------------------------------------------------------------
Base = declarative_base()

# --------------------------------------------------------------------------
# 3️⃣ Trade model
# --------------------------------------------------------------------------
class Trade(Base):
    """ORM model for storing trade records."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    side = Column(String, nullable=False)  # "BUY" or "SELL"
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)

    def __repr__(self):
        return (
            f"<Trade(symbol='{self.symbol}', side='{self.side}', "
            f"quantity={self.quantity}, price={self.price}, timestamp='{self.timestamp}')>"
        )

# --------------------------------------------------------------------------
# 4️⃣ Engine and Session configuration
# --------------------------------------------------------------------------
engine = create_engine(DATABASE_URL, future=True, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --------------------------------------------------------------------------
# 5️⃣ Database initialization helper
# --------------------------------------------------------------------------
def init_db():
    """Create all tables in the database (if they don't exist)."""
    print(f"🔗 Connecting to DB: {DATABASE_URL}")
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created or already exist.")
    except Exception as e:
        logging.critical(f"❌ Failed to initialize database: {e}", exc_info=True)
        raise
