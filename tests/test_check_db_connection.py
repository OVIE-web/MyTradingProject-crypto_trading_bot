
import psycopg2
import pytest
from psycopg2 import OperationalError

from src.config import DATABASE_URL


def test_database_connection(monkeypatch):
    """Ensure database connection works with test configuration."""
    # Use test database URL
    test_db_url = DATABASE_URL.replace("/tradingbot", "/tradingbot_test")
    monkeypatch.setenv("DATABASE_URL", test_db_url)
    
    try:
        conn = psycopg2.connect(test_db_url)
        cursor = conn.cursor()
        
        # Verify we can execute queries
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1, "Basic query failed"
        
        cursor.close()
        conn.close()
    except OperationalError as e:
        pytest.fail(f"Database connection failed: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during database test: {e}")
