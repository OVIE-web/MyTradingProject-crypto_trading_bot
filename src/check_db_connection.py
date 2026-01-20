# src/check_db_connection.py

import logging
import os
import time

import psycopg2
from psycopg2 import OperationalError

logger = logging.getLogger(__name__)


def wait_for_postgres() -> bool:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL is not set in environment variables.")

    logging.info(f"🔄 Checking database connection: {db_url}")

    for i in range(30):  # retry up to 30 times (≈ 90 seconds total)
        try:
            conn = psycopg2.connect(db_url)
            conn.close()
            logging.info("✅ Database connection successful!")
            return True
        except OperationalError as exc:
            logging.warning("⏳ Waiting for database... (%d/30): %s", i + 1, exc)
            time.sleep(3)
    raise TimeoutError("❌ Database not reachable after 90 seconds.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    wait_for_postgres()
    logger.info("Database connection successful")
