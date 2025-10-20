# src/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL=postgresql://testuser:testpass@db:5432/tradingbot_test")

settings = Settings()