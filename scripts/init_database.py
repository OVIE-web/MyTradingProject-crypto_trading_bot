# scripts/init_database.py
"""
Initialize the database schema (creates tables if missing).
Run with:
    python -m scripts.init_database
"""

import sys
import os
import logging

# Ensure project root is in import path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.db import init_db, list_tables


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("🚀 Starting database initialization...")

    try:
        init_db()
        tables = list_tables()
        logging.info(f"📊 Tables in database: {tables}")
        logging.info("✅ Database initialization complete.")
    except Exception as e:
        logging.critical(f"❌ Database initialization failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
