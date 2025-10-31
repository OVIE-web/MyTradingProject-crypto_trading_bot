# src/wait_for_postgres.py
import time
import logging
import psycopg2
from psycopg2 import OperationalError
from src.config import DATABASE_URL

def wait_for_postgres(max_retries: int = 10, delay: int = 2) -> bool:
    """
    Try to connect to Postgres until it's ready or retries are exhausted.
    
    Args:
        max_retries (int): Maximum number of connection attempts
        delay (int): Seconds to wait between attempts
        
    Returns:
        bool: True if connection successful, raises TimeoutError if not
        
    Raises:
        ValueError: If DATABASE_URL is not set
        TimeoutError: If unable to connect after max_retries
    """
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set")

    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(DATABASE_URL)
            conn.close()
            print("✅ Postgres is ready")
            return True
        except OperationalError as e:
            logging.warning(f"⏳ Postgres not ready (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:  # Don't sleep on last attempt
                time.sleep(delay)
    
    raise TimeoutError(f"❌ Could not connect to Postgres after {max_retries} attempts")