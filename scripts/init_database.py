# scripts/init_database.py
"""
Initialize the database schema (creates tables if missing).
Run with:
    python -m scripts.init_database [--test]
"""
import argparse
import logging
import os
import sys

from src.config import DATABASE_URL
from src.db import init_db, list_tables

# Ensure project root is in import path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def init_database(is_test: bool = False) -> None:
    """Initialize main or test database schema."""
    if is_test:
        # Use test database URL
        test_db_url = DATABASE_URL.replace("/tradingbot", "/tradingbot_test")
        os.environ["DATABASE_URL"] = test_db_url
        logging.info("🧪 Initializing test database...")
    else:
        logging.info("🚀 Starting database initialization...")

    try:
        init_db()
        tables = list_tables()
        logging.info(f"📊 Tables in database: {tables}")
        logging.info("✅ Database initialization complete.")
    except Exception as e:
        logging.critical(f"❌ Database initialization failed: {e}", exc_info=True)
        sys.exit(1)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Initialize the database schema.")
    parser.add_argument(
        "--test", action="store_true", help="Initialize test database instead of main database"
    )
    args = parser.parse_args()

    init_database(is_test=args.test)


if __name__ == "__main__":
    main()
