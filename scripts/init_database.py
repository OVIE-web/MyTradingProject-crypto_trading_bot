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
from src.db import Trade, init_db

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
        trades = Trade()
        logging.info(f"📊 Tables in database: {trades}")
        logging.info("✅ Database initialization complete.")
    except Exception as e:
        logging.critical(f"❌ Database initialization failed: {e}", exc_info=True)
        sys.exit(1)


def main() -> None:
    """Main entry point for the script.

    Initializes the logging module and an argument parser.

    The argument parser is configured to accept a single boolean flag
    (--test) which determines whether the main or test database
    should be initialized.

    The init_database function is called with the parsed argument
    value for --test.

    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Initialize the database schema.")
    parser.add_argument(
        "--test", action="store_true", help="Initialize test database instead of main database"
    )
    args = parser.parse_args()

    init_database(is_test=args.test)


if __name__ == "__main__":
    main()
