# src/config.py
import os
import logging
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# 1️⃣ Load environment variables
# --------------------------------------------------------------------------
# Load global .env (if present) and the local project one (.env.local)
load_dotenv()
load_dotenv(dotenv_path=".env.local", override=True)

# --------------------------------------------------------------------------
# 2️⃣ Logging setup (for early debug visibility)
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --------------------------------------------------------------------------
# 3️⃣ Binance Configuration
# --------------------------------------------------------------------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "False").lower() == "true"

BINANCE_API_URL = "https://api.binance.com"
BINANCE_TESTNET_API_URL = "https://testnet.binance.vision/api"

def get_binance_keys():
    """Return Binance API credentials and testnet flag."""
    return BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET

# --------------------------------------------------------------------------
# 4️⃣ Database Configuration
# --------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("❌ DATABASE_URL not found. Check your .env.local file.")

# Optional: mask sensitive info for logs
safe_db_url = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
logging.info(f"📦 Using database at: {safe_db_url}")

# --------------------------------------------------------------------------
# 5️⃣ Paths and Model Configuration
# --------------------------------------------------------------------------
DATA_FILE_PATH = os.path.join("data", "test_df_features.csv")
MODEL_TYPE = "xgboost"
# Allow overriding the model directory via environment variable for flexibility
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
MODEL_SAVE_FILENAME = os.getenv("MODEL_SAVE_FILENAME", "xgboost_model.json")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, MODEL_SAVE_FILENAME)

# --------------------------------------------------------------------------
# 6️⃣ Trading Parameters
# --------------------------------------------------------------------------
TRADE_SYMBOL = "BTCUSDT"
TRADE_INTERVAL = "4h"
TRADE_QUANTITY = 0.001
INITIAL_CANDLES_HISTORY = 100

# --------------------------------------------------------------------------
# 7️⃣ Technical Indicator Parameters
# --------------------------------------------------------------------------
RSI_WINDOW = 14
BB_WINDOW = 20
BB_WINDOW_DEV = 2
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 50
ATR_WINDOW = 14

# RSI quantile thresholds (for dynamic signal levels)
RSI_LOWER_QUANTILE = 0.2
RSI_UPPER_QUANTILE = 0.8

# --------------------------------------------------------------------------
# 8️⃣ Model Training Parameters
# --------------------------------------------------------------------------
TARGET_COLUMN = "signal"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CONFIDENCE_THRESHOLD = 0.30

# --------------------------------------------------------------------------
# 9️⃣ Backtesting Parameters
# --------------------------------------------------------------------------
INITIAL_BALANCE = 10000
TRANSACTION_FEE_PCT = 0.001

FEATURE_COLUMNS = [
    "rsi", "bb_upper", "bb_lower", "bb_mid", "bb_pct_b",
    "sma_20", "sma_50", "ma_cross", "price_momentum",
    "atr", "atr_pct"
]

# --------------------------------------------------------------------------
# 🔔 10️⃣ Notification Configuration
# --------------------------------------------------------------------------
# Email
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("EMAIL_TO")

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --------------------------------------------------------------------------
# 🔐 11️⃣ JWT (for optional web dashboard)
# --------------------------------------------------------------------------
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

# --------------------------------------------------------------------------
# 12️⃣ Runtime Optimization
# --------------------------------------------------------------------------
# Disable joblib multiprocessing (for XGBoost + pytest stability)
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
