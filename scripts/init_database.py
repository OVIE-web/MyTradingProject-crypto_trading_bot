import logging
from src.db import init_db

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Initializing database schema...")
    init_db()
    logging.info("✅ Database initialization complete!")
    logging.info("You can now run the bot with 'python main.py'")