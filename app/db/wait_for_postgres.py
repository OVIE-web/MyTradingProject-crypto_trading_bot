# app/db/wait_for_postgres.py
from __future__ import annotations

import logging
import time

import psycopg2
from psycopg2 import OperationalError

from app.core.config import DATABASE_URL

LOG = logging.getLogger(__name__)


def wait_for_postgres(max_retries: int = 30, delay: int = 3) -> bool:
    """
    Wait for PostgreSQL to become available.

    Used by Docker, startup scripts, tests, and local development.

    Args:
        max_retries: Maximum number of connection attempts.
        delay: Seconds to wait between attempts.

    Returns:
        True if connection succeeds.

    Raises:
        ValueError: If DATABASE_URL is missing.
        TimeoutError: If PostgreSQL is unreachable after all retries.
    """
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set.")

    LOG.info("🔄 Checking database connection...")

    for attempt in range(1, max_retries + 1):
        try:
            conn = psycopg2.connect(DATABASE_URL)
            conn.close()
            LOG.info("✅ PostgreSQL is ready.")
            return True

        except OperationalError as exc:
            LOG.warning(
                "⏳ PostgreSQL not ready yet (%s/%s): %s",
                attempt,
                max_retries,
                exc,
            )

            if attempt < max_retries:
                time.sleep(delay)

    raise TimeoutError(f"❌ PostgreSQL not reachable after {max_retries} attempts.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    wait_for_postgres()
    LOG.info("Database connection successful")
