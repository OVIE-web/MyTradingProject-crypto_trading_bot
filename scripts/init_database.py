import logging
from src.db import init_db

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info("Initializing database...")
    init_db()
    logging.info("Database initialization completed.")