# src/wait_for_postgres.py
import time
import psycopg2
from psycopg2 import OperationalError
from src.config import DATABASE_URL   # <-- now consistent

def wait_for_postgres(max_retries: int = 10, delay: int = 2) -> None:
    """Try to connect to Postgres until it's ready or retries are exhausted."""
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(DATABASE_URL)
            conn.close()
            print("✅ Postgres is ready")
            return
        except OperationalError:
            print(f"⏳ Postgres not ready, retrying ({attempt+1}/{max_retries})...")
            time.sleep(delay)
    raise RuntimeError("❌ Could not connect to Postgres after retries")