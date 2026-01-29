# src/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field


def get_env_str(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None else default


def get_env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val is not None else default


def get_env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val is not None else default


@dataclass(frozen=True)
class Config:
    # Binance
    binance_api_key: str | None = os.getenv("BINANCE_API_KEY")
    binance_api_secret: str | None = os.getenv("BINANCE_API_SECRET")
    binance_testnet: bool = os.getenv("BINANCE_TESTNET", "False").lower() == "true"

    # Database
    database_url: str = get_env_str(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/trading"
    )

    # Paths & Models
    model_type: str = "xgboost"
    model_dir: str = get_env_str(
        "MODEL_DIR", os.path.join(os.path.abspath(os.path.dirname(__file__)), "models")
    )
    model_save_filename: str = get_env_str("MODEL_SAVE_FILENAME", "xgboost_model.json")
    model_save_path: str = os.path.join(model_dir, model_save_filename)
    model_metadata_path: str = model_save_path + ".meta.json"

    # Trading
    trade_symbol: str = get_env_str("TRADE_SYMBOL", "BTCUSDT")
    trade_interval: str = get_env_str("TRADE_INTERVAL", "4h")
    trade_quantity: float = get_env_float("TRADE_QUANTITY", 0.001)
    initial_candles_history: int = get_env_int("INITIAL_CANDLES_HISTORY", 5000)

    # Backtesting
    initial_balance: int = get_env_int("INITIAL_BALANCE", 10000)
    transaction_fee_pct: float = get_env_float("TRANSACTION_FEE_PCT", 0.001)

    # Indicators
    rsi_window: int = 14
    bb_window: int = 20
    bb_window_dev: int = 2
    sma_short_window: int = 20
    sma_long_window: int = 50
    atr_window: int = 14

    rsi_lower_quantile: float = 0.2
    rsi_upper_quantile: float = 0.8

    # Training
    target_column: str = "signal"
    test_size: float = 0.2
    random_state: int = 42
    confidence_threshold: float = 0.30

    feature_columns: list[str] = field(
        default_factory=lambda: [
            "rsi",
            "bb_upper",
            "bb_lower",
            "bb_mid",
            "bb_pct_b",
            "sma_20",
            "sma_50",
            "ma_cross",
            "price_momentum",
            "atr",
            "atr_pct",
        ]
    )

    # Notifications
    email_host: str | None = os.getenv("EMAIL_HOST")
    email_port: int = get_env_int("EMAIL_PORT", 587)
    email_user: str | None = os.getenv("EMAIL_USER")
    email_pass: str | None = os.getenv("EMAIL_PASS")
    email_to: str | None = os.getenv("EMAIL_TO")
    telegram_bot_token: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: int = get_env_int("TELEGRAM_CHAT_ID", 0)


# Instantiate config
config = Config()

# ---- Legacy compatibility constants (uppercase required by tests) ----

# Binance
BINANCE_API_KEY = config.binance_api_key
BINANCE_API_SECRET = config.binance_api_secret
BINANCE_TESTNET = config.binance_testnet

# Database
DATABASE_URL = config.database_url

# Trading / Backtesting
INITIAL_BALANCE = config.initial_balance
TRANSACTION_FEE_PCT = config.transaction_fee_pct
INITIAL_CANDLES_HISTORY = config.initial_candles_history
# Trading constants
TRADE_SYMBOL = config.trade_symbol
TRADE_INTERVAL = config.trade_interval
TRADE_QUANTITY = config.trade_quantity

# Indicators
RSI_LOWER_QUANTILE = config.rsi_lower_quantile
RSI_UPPER_QUANTILE = config.rsi_upper_quantile
RSI_WINDOW = config.rsi_window
BB_WINDOW = config.bb_window
BB_WINDOW_DEV = config.bb_window_dev
SMA_SHORT_WINDOW = config.sma_short_window
SMA_LONG_WINDOW = config.sma_long_window
ATR_WINDOW = config.atr_window

# Notifications
EMAIL_HOST = config.email_host
EMAIL_PORT = config.email_port
EMAIL_USER = config.email_user
EMAIL_PASS = config.email_pass
TELEGRAM_BOT_TOKEN = config.telegram_bot_token
TELEGRAM_CHAT_ID = config.telegram_chat_id

# ML / Training
FEATURE_COLUMNS = config.feature_columns
TARGET_COLUMN = config.target_column
TEST_SIZE = config.test_size
RANDOM_STATE = config.random_state
CONFIDENCE_THRESHOLD = config.confidence_threshold

# Model paths
MODEL_SAVE_FILENAME = config.model_save_filename
MODEL_SAVE_PATH = config.model_save_path

# ------------------------------------------------------------------
# Security constants
# ------------------------------------------------------------------
ISSUER = "crypto-trading-bot"
AUDIENCE = "crypto-trading-api"
