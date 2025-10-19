import pytest
import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock

# Insert repo root (one level up from tests/) so project `src` package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import project symbols (after adding repo root to sys.path)
from src.db import Base

@pytest.fixture(autouse=True)
def mock_env_vars():
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

def sample_ohlcv_data():
    """Provides a sample OHLCV DataFrame for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = {
        'open_time': dates,
        'open': np.random.uniform(100, 200, 100).cumsum() + 1000,
        'high': np.random.uniform(100, 200, 100).cumsum() + 1010,
        'low': np.random.uniform(100, 200, 100).cumsum() + 990,
        'close': np.random.uniform(100, 200, 100).cumsum() + 1005,
        'volume': np.random.uniform(1000, 10000, 100),
        'close_time': dates + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
        'quote_asset_volume': np.random.uniform(100000, 1000000, 100),
        'number_of_trades': np.random.randint(100, 1000, 100),
        'taker_buy_base_asset_volume': np.random.uniform(500, 5000, 100),
        'taker_buy_quote_asset_volume': np.random.uniform(50000, 500000, 100),
        'ignore': [0] * 100
    }
    df = pd.DataFrame(data).set_index('open_time')
    return df.copy()

@pytest.fixture
def sample_df_with_indicators(sample_ohlcv_data):
    """Provides a sample DataFrame with indicators for testing feature_engineer."""
    from src.feature_engineer import calculate_technical_indicators  # avoid circular import
    df = calculate_technical_indicators(sample_ohlcv_data.copy())
    return df

@pytest.fixture
def mock_xgboost_model():
    """Mocks a trained XGBoost model for predictions."""
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]] * 10)  # mostly hold
    mock_model.predict.return_value = np.array([1] * 10)  # class 1 = hold
    return mock_model

@pytest.fixture
def db_session():
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