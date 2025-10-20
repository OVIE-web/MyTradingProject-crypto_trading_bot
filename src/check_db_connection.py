import os
import time
import psycopg2
from psycopg2 import OperationalError

def wait_for_postgres():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL is not set in environment variables.")

    print(f"🔄 Checking database connection: {db_url}")

    for i in range(30):  # retry up to 30 times (≈ 90 seconds total)
        try:
            conn = psycopg2.connect(db_url)
            conn.close()
            print("✅ Database connection successful!")
            return True
        except OperationalError as e:
            print(f"⏳ Waiting for database... ({i+1}/30): {e}")
            time.sleep(3)
    raise TimeoutError("❌ Database not reachable after 90 seconds.")

if __name__ == "__main__":
    wait_for_postgres()
    print("✅ Database connection successful")

