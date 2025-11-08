import os
import logging
from dotenv import load_dotenv
from dataclasses import dataclass

# --------------------------------------------------------------------------
# 1️⃣ Load environment variables
# --------------------------------------------------------------------------
load_dotenv()  # global .env
load_dotenv(dotenv_path=".env.local", override=True)  # project-specific overrides

# --------------------------------------------------------------------------
# 2️⃣ Logging setup (early visibility for stakeholders)
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
    raise RuntimeError("❌ DATABASE_URL not found. Please set it in .env.local")

# Mask sensitive info for logs
safe_db_url = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
logging.info(f"📦 Using database at: {safe_db_url}")

# --------------------------------------------------------------------------
# 5️⃣ Paths and Model Configuration
# --------------------------------------------------------------------------
DATA_FILE_PATH = os.path.join("data", "test_df_features.csv")
if not os.path.exists(DATA_FILE_PATH):
    logging.warning(
        f"⚠️ Data file not found at {DATA_FILE_PATH}. "
        "Ensure the file exists or update DATA_FILE_PATH in config.py"
    )

MODEL_TYPE = "xgboost"
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
MODEL_SAVE_FILENAME = os.getenv("MODEL_SAVE_FILENAME", "xgboost_model.json")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, MODEL_SAVE_FILENAME)
MODEL_METADATA_PATH = MODEL_SAVE_PATH + ".meta.json"

# --------------------------------------------------------------------------
# 6️⃣ Trading Parameters
# --------------------------------------------------------------------------
TRADE_SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSDT")
TRADE_INTERVAL = os.getenv("TRADE_INTERVAL", "4h")
TRADE_QUANTITY = float(os.getenv("TRADE_QUANTITY", 0.001))
INITIAL_CANDLES_HISTORY = int(os.getenv("INITIAL_CANDLES_HISTORY", 5000))

# --------------------------------------------------------------------------
# 7️⃣ Technical Indicator Parameters
# --------------------------------------------------------------------------
RSI_WINDOW = 14
BB_WINDOW = 20
BB_WINDOW_DEV = 2
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 50
ATR_WINDOW = 14

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
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", 10000))
TRANSACTION_FEE_PCT = float(os.getenv("TRANSACTION_FEE_PCT", 0.001))

FEATURE_COLUMNS = [
    "rsi", "bb_upper", "bb_lower", "bb_mid", "bb_pct_b",
    "sma_20", "sma_50", "ma_cross", "price_momentum",
    "atr", "atr_pct"
]

# --------------------------------------------------------------------------
# 🔔 10️⃣ Notification Configuration
# --------------------------------------------------------------------------
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("EMAIL_TO")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --------------------------------------------------------------------------
# 🔐 11️⃣ JWT (for optional web dashboard)
# --------------------------------------------------------------------------
class Settings:
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "supersecretkey123")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "adminpass")

settings = Settings()


# --------------------------------------------------------------------------
# 12️⃣ Runtime Optimization
# --------------------------------------------------------------------------
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

# --------------------------------------------------------------------------
# 📊 Stakeholder-friendly summary
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfigSummary:
    db: str = safe_db_url
    model_path: str = MODEL_SAVE_PATH
    trading_symbol: str = TRADE_SYMBOL
    interval: str = TRADE_INTERVAL
    initial_balance: float = INITIAL_BALANCE
    notifications_enabled: bool = bool(EMAIL_USER or TELEGRAM_BOT_TOKEN)

CONFIG_SUMMARY = ConfigSummary()
logging.info(f"✅ Config loaded: {CONFIG_SUMMARY}")

# --------------------------------------------------
