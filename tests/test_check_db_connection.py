import pytest
import psycopg2
from psycopg2 import OperationalError
import os

def test_database_connection():
    """Ensure that the DATABASE_URL environment variable connects successfully."""
    db_url = os.getenv("DATABASE_URL")
    assert db_url, "DATABASE_URL must be set for this test"

    try:
        conn = psycopg2.connect(db_url)
        conn.close()
    except OperationalError as e:
        pytest.fail(f"Database connection failed: {e}")
