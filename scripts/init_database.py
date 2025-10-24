# init_database.py
import logging
from src.db import init_db
from sqlalchemy import inspect
from src.db import engine

# Configure logging to always print
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

if __name__ == "__main__":
    logging.info("🚀 Starting database initialization...")
    try:
        init_db()  # create tables
        logging.info("✅ Database tables created or already exist.")

        # Optional: list all current tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        logging.info(f"📊 Tables in database: {tables}")

    except Exception as e:
        logging.critical(f"❌ Database initialization failed: {e}", exc_info=True)
