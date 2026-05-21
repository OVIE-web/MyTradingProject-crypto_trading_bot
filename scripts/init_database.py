"""Initialize the database schema.

Run with:
    python -m scripts.init_database [--test]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def init_database(is_test: bool = False) -> None:
    """Initialize main or test database schema."""
    if is_test:
        test_db_url = config.DATABASE_URL.replace("/tradingbot", "/tradingbot_test")
        os.environ["DATABASE_URL"] = test_db_url
        config.DATABASE_URL = test_db_url
        logger.info("Initializing test database...")
    else:
        logger.info("Starting database initialization...")

    try:
        from app.db import Base, init_db

        init_db()
        table_names = ", ".join(sorted(Base.metadata.tables)) or "no registered tables"
        logger.info("Tables in database metadata: %s", table_names)
        logger.info("Database initialization complete.")
    except Exception as exc:
        logger.critical("Database initialization failed: %s", exc, exc_info=True)
        sys.exit(1)


def main() -> None:
    """Parse CLI arguments and initialize the requested database schema."""
    parser = argparse.ArgumentParser(description="Initialize the database schema.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Initialize test database instead of main database",
    )
    args = parser.parse_args()

    init_database(is_test=args.test)


if __name__ == "__main__":
    main()
