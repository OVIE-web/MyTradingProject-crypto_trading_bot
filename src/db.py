# src/db.py
import logging
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Generator, List

import psycopg2
from sqlalchemy import Column, DateTime, Integer, Numeric, String, create_engine, inspect
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.config import DATABASE_URL

# --------------------------------------------------------------------------
# Setup logging
# --------------------------------------------------------------------------
if DATABASE_URL is None:
    raise ValueError("DATABASE_URL environment variable is not set")

safe_url = str(make_url(DATABASE_URL)).split("@")[-1]
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
    symbol = Column(String(20), index=True, nullable=False)
    side = Column(String(10), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8), nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    order_id = Column(String(50), unique=True, nullable=True)
    fill_price = Column(Numeric(20, 8), nullable=True)
    commission = Column(Numeric(20, 8), nullable=True)
    commission_asset = Column(String(10), nullable=True)


# --------------------------------------------------------------------------
# Engine and Session
# --------------------------------------------------------------------------
engine = create_engine(DATABASE_URL, future=True, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Get a database session using context manager."""
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        db.rollback()
        logging.error(f"❌ Database error: {e}")
        raise
    finally:
        db.close()


def init_db() -> None:
    """Initialize the database schema (create tables if not exist)."""
    logging.info("🚀 Starting database initialization...")
    try:
        Base.metadata.create_all(bind=engine)
        logging.info("✅ Database tables created or already exist.")
    except Exception as e:
        logging.error(f"❌ Failed to initialize database: {e}")
        raise


def list_tables() -> List[str]:
    """Return a list of existing tables in the database."""
    inspector = inspect(engine)
    return inspector.get_table_names()


def wait_for_postgres(timeout_seconds: int = 30) -> bool:
    """
    Wait for PostgreSQL to become available.

    Args:
        timeout_seconds: Maximum time to wait for connection.

    Returns:
        bool: True if connection successful, raises TimeoutError if not.
    """

    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        try:
            conn = psycopg2.connect(DATABASE_URL)
            conn.close()
            print("✅ Postgres is ready")
            return True
        except psycopg2.OperationalError:
            time.sleep(1)

    raise TimeoutError("Failed to connect to PostgreSQL within timeout period")
