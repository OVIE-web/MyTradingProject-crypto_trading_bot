import os
from dataclasses import dataclass, field
from typing import Optional

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
    binance_api_key: Optional[str] = os.getenv("BINANCE_API_KEY")
    binance_api_secret: Optional[str] = os.getenv("BINANCE_API_SECRET")
    binance_testnet: bool = os.getenv("BINANCE_TESTNET", "False").lower() == "true"

    # Database
    database_url: str = get_env_str("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/trading")

    # Paths & Models
    model_type: str = "xgboost"
    model_dir: str = get_env_str("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
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
    feature_columns: list[str] = field(default_factory=lambda: 
        ["rsi", "bb_upper", "bb_lower", "bb_mid", "bb_pct_b",
     "sma_20", "sma_50", "ma_cross", "price_momentum",
     "atr", "atr_pct"])

    # Notifications
    email_host: Optional[str] = os.getenv("EMAIL_HOST")
    email_port: int = get_env_int("EMAIL_PORT", 587)
    email_user: Optional[str] = os.getenv("EMAIL_USER")
    email_pass: Optional[str] = os.getenv("EMAIL_PASS")
    email_to: Optional[str] = os.getenv("EMAIL_TO")
    telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: int = get_env_int("TELEGRAM_CHAT_ID", 0)

    # JWT / Admin
    jwt_secret_key: str = get_env_str("JWT_SECRET_KEY", "supersecretkey123")
    jwt_algorithm: str = get_env_str("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = get_env_int("ACCESS_TOKEN_EXPIRE_MINUTES", 30)
    admin_username: str = get_env_str("ADMIN_USERNAME", "admin")
    admin_password: str = get_env_str("ADMIN_PASSWORD", "adminpass")
