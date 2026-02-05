import os
import sys
import warnings
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import sqlalchemy.orm.session
from pytest import MonkeyPatch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db import Base
from src.feature_engineer import calculate_technical_indicators  # avoid circular import

# Insert repo root (one level up from tests/) so project `src` package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(autouse=True, scope="session")
def silence_deprecation_warnings() -> None:
    # Suppress specific known deprecation warnings
    warnings.filterwarnings(
        "ignore",
        message="datetime.datetime.utcnow() is deprecated",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        module="websockets",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        module="dateparser.timezone_parser",
        category=DeprecationWarning,
    )


@pytest.fixture
def mock_binance_client(monkeypatch: MonkeyPatch) -> Generator[MagicMock, Any, None]:
    """Mocks the Binance Client for tests."""

    with patch("src.binance_manager.Client") as mock_client:
        instance = MagicMock()
        mock_client.return_value = instance
        instance.ping.return_value = {}
        instance.get_account.return_value = {"balances": []}
        yield instance


@pytest.fixture(autouse=True, scope="session")
def setup_test_env() -> None:
    """Set up all required environment variables for testing."""
    os.environ["ENV"] = "test"

    # JWT Settings
    if not os.getenv("JWT_SECRET_KEY"):
        os.environ["JWT_SECRET_KEY"] = "test-secret-key-32-characters-minimum-for-jwt-testing"

    os.environ.setdefault("JWT_ALGORITHM", "HS256")
    os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "30")

    # Admin Credentials
    os.environ.setdefault("ADMIN_USERNAME", "testadmin")
    os.environ.setdefault("ADMIN_PASSWORD", "testpass123")

    # Database
    os.environ.setdefault(
        "DATABASE_URL", "postgresql://testuser:testpass@localhost:5432/tradingbot_test"
    )


@pytest.fixture(autouse=True)
def mock_env_vars() -> Generator[None, Any, None]:
    """Mocks essential environment variables for tests."""
    original_environ = os.environ.copy()
    mock_env = {
        "BINANCE_API_KEY": "mock_api_key",
        "BINANCE_API_SECRET": "mock_api_secret",
        "TELEGRAM_BOT_TOKEN": "mock_telegram_token",
        "TELEGRAM_CHAT_ID": "123456789",
        "EMAIL_HOST": "mock.smtp.com",
        "EMAIL_PORT": "587",
        "EMAIL_USER": "mock@example.com",
        "EMAIL_PASS": "mock_pass",
        "EMAIL_TO": "mock_to@example.com",
        # Do NOT overwrite DATABASE_URL if the user has provided one (e.g., for Postgres testing).
    }
    for key, value in mock_env.items():
        # Only set vars that are not already set in the environment
        if key not in os.environ:
            os.environ[key] = value
    yield
    os.environ.clear()
    os.environ.update(original_environ)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Provides a sample OHLCV DataFrame for testing."""
    # Set random seed for reproducibility
    np.random.seed(42)

    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    base_price = 1000.0

    # Generate correlated price movements
    price_changes = np.random.normal(0, 20, 100).cumsum()
    base_prices = base_price + price_changes

    data = {
        "open_time": dates,
        "open": base_prices + np.random.normal(0, 5, 100),
        "high": base_prices + np.random.uniform(10, 20, 100),
        "low": base_prices - np.random.uniform(10, 20, 100),
        "close": base_prices + np.random.normal(0, 5, 100),
        "volume": np.random.uniform(1000, 10000, 100),
        "close_time": dates + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
        "quote_asset_volume": np.random.uniform(100000, 1000000, 100),
        "number_of_trades": np.random.randint(100, 1000, 100),
        "taker_buy_base_asset_volume": np.random.uniform(500, 5000, 100),
        "taker_buy_quote_asset_volume": np.random.uniform(50000, 500000, 100),
        "ignore": [0] * 100,
    }

    # Ensure high is always highest and low is always lowest
    data["high"] = np.maximum.reduce(
        [np.array(data["high"]), np.array(data["open"]), np.array(data["close"])]
    )
    data["low"] = np.minimum.reduce(
        [np.array(data["low"]), np.array(data["open"]), np.array(data["close"])]
    )

    df = pd.DataFrame(data).set_index("open_time")
    return df.copy()


@pytest.fixture
def sample_df_with_indicators(sample_ohlcv_data: pd.DataFrame) -> pd.DataFrame:
    """Provides a sample DataFrame with indicators for testing feature_engineer."""

    df = calculate_technical_indicators(sample_ohlcv_data.copy())

    return df


@pytest.fixture
def mock_xgboost_model() -> Mock:
    """Mocks a trained XGBoost model for predictions."""
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]] * 10)  # mostly hold
    mock_model.predict.return_value = np.array([1] * 10)  # class 1 = hold
    return mock_model


@pytest.fixture
def db_session() -> Generator[sqlalchemy.orm.session.Session, Any, None]:
    """Provides a clean, in-memory SQLAlchemy session for each database test."""
    # Use SQLite in-memory engine for tests (fast and reliable).
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
