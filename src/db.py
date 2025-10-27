# src/db.py
import logging
from datetime import datetime, UTC
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, inspect
from sqlalchemy.orm import declarative_base, sessionmaker
from src.config import DATABASE_URL

# --------------------------------------------------------------------------
# Setup logging
# --------------------------------------------------------------------------
safe_url = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.info(f"📦 Connecting to database at: {safe_url}")

# --------------------------------------------------------------------------
# Base and Model
# --------------------------------------------------------------------------
Base = declarative_base()

class Trade(Base):
    """ORM model for storing trade records."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)

# --------------------------------------------------------------------------
# Engine and Session
# --------------------------------------------------------------------------
engine = create_engine(DATABASE_URL, future=True, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def init_db():
    """Initialize the database schema (create tables if not exist)."""
    logging.info("🚀 Starting database initialization...")
    Base.metadata.create_all(bind=engine)
    logging.info("✅ Database tables created or already exist.")

def list_tables():
    """Return a list of existing tables in the database."""
    inspector = inspect(engine)
    return inspector.get_table_names()
