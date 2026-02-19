import pytest

from src.config import Config


@pytest.mark.unit
def test_initial_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test default initial balance."""
    monkeypatch.delenv("INITIAL_BALANCE", raising=False)
    cfg = Config()
    assert cfg.initial_balance == 10000


@pytest.mark.unit
def test_default_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test default trading configuration values."""
    # Clear env vars to force defaults
    monkeypatch.delenv("TRADE_SYMBOL", raising=False)
    monkeypatch.delenv("TRADE_INTERVAL", raising=False)
    monkeypatch.delenv("TRADE_QUANTITY", raising=False)

    cfg = Config()

    assert cfg.trade_symbol == "BTCUSDT"
    assert cfg.trade_interval == "4h"
    assert isinstance(cfg.trade_quantity, float)
    assert cfg.trade_quantity == 0.001


@pytest.mark.unit
def test_custom_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test custom environment variables are loaded correctly."""
    monkeypatch.setenv("TRADE_SYMBOL", "ETHUSDT")
    monkeypatch.setenv("TRADE_INTERVAL", "1h")
    monkeypatch.setenv("TRADE_QUANTITY", "0.05")
    monkeypatch.setenv("INITIAL_BALANCE", "5000")
    monkeypatch.setenv("EMAIL_PORT", "587")

    # Reload config to pick up new env vars
    import importlib

    import src.config

    importlib.reload(src.config)
    from src.config import Config as ReloadedConfig

    cfg = ReloadedConfig()

    assert cfg.trade_symbol == "ETHUSDT"
    assert cfg.trade_interval == "1h"
    assert cfg.trade_quantity == 0.05
    assert cfg.initial_balance == 5000
    assert cfg.email_port == 587


@pytest.mark.unit
def test_database_url() -> None:
    """Test database URL is properly configured."""
    cfg = Config()

    # Verify it's a valid PostgreSQL URL (not exact match, since env var might change)
    assert isinstance(cfg.database_url, str), "database_url should be a string"
    assert cfg.database_url.startswith("postgresql://"), "Should be PostgreSQL URL"
    assert ":" in cfg.database_url, "Should contain password separator"
    assert "@" in cfg.database_url, "Should contain host separator"
    assert "5432" in cfg.database_url, "Should contain PostgreSQL port"


@pytest.mark.unit
def test_jwt_settings() -> None:
    """Test JWT configuration is properly set."""
    cfg = Config()

    # Check that JWT-related attributes exist
    assert hasattr(cfg, "jwt_secret_key") or hasattr(cfg, "JWT_SECRET_KEY"), (
        "Config should have JWT secret key attribute"
    )

    # Verify JWT secret key is a non-empty string
    jwt_key = getattr(cfg, "jwt_secret_key", None) or getattr(cfg, "JWT_SECRET_KEY", None)
    assert jwt_key, "JWT secret key should not be empty"
    assert isinstance(jwt_key, str), "JWT secret key should be a string"
