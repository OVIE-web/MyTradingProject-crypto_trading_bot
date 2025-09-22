# src/config.py

import os
from dotenv import load_dotenv
load_dotenv()

# --- Data Paths ---
# Create a 'data' folder in your project root and place test_df_features.csv inside it.
DATA_FILE_PATH = 'data/test_df_features.csv'

# -- Model Configuration ---
MODEL_TYPE = 'xgboost' # Type of model to use
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_model.json') # Path to save/load model

# -- Postgres DATABASE Configuration --
# IMPORTANT: When running with Docker Compose, 'localhost' in the bot container refers to the container itself.
# Use the service name 'db' to refer to the PostgreSQL container.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://tradinguser:tradingpass@localhost:5432/tradingbot") # Default for Docker Compose
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable not set! Please check your .env file or default.")

# --- Binance API Configuration ---
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
BINANCE_TESTNET = True # Set to False for real trading, but START WITH TRUE
BINANCE_API_URL = "https://api.binance.com"
BINANCE_TESTNET_API_URL = "https://testnet.binance.vision/api"

# --- Live Trading Parameters ---
TRADE_SYMBOL = 'BTCUSDT'
TRADE_INTERVAL = '1h'
TRADE_QUANTITY = 0.001
INITIAL_CANDLES_HISTORY = 100

# --- Technical Indicator Parameters (existing) ---
RSI_WINDOW = 14
BB_WINDOW = 20
BB_WINDOW_DEV = 2
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 50
ATR_WINDOW = 14

# --- RSI Signal Quantile Thresholds (for dynamic thresholds) ---
RSI_LOWER_QUANTILE = 0.2
RSI_UPPER_QUANTILE = 0.8

# --- Model Parameters (existing) ---
TARGET_COLUMN = 'signal'
TEST_SIZE = 0.2
RANDOM_STATE = 42
CONFIDENCE_THRESHOLD = 0.30

# --- Backtesting Parameters (existing) ---
INITIAL_BALANCE = 10000
TRANSACTION_FEE_PCT = 0.001

# --- Feature Columns (existing) ---
FEATURE_COLUMNS = [
    'rsi', 'bb_upper', 'bb_lower', 'bb_mid', 'bb_pct_b',
    'sma_20', 'sma_50', 'ma_cross', 'price_momentum',
    'atr', 'atr_pct'
]

# --- Email Notification Configuration ---
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("EMAIL_TO")

# --- Telegram Notification Configuration (existing) ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# --- JWT Configuration (for future web UI, not currently used by bot core) ---
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')