# init_database.py
import logging
from src.db import init_db

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Initializing database schema...")
    init_db()