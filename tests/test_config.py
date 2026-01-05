import os

from config import Config


def test_initial_balance(monkeypatch):
    monkeypatch.delenv("INITIAL_BALANCE", raising=False)
    initial_balance = os.getenv("INITIAL_BALANCE")
    cfg = Config()

    assert cfg.initial_balance == 10000

def test_default_values(monkeypatch):
    # Clear env vars to force defaults
    monkeypatch.delenv("TRADE_SYMBOL", raising=False)
    monkeypatch.delenv("TRADE_INTERVAL", raising=False)
    monkeypatch.delenv("TRADE_QUANTITY", raising=False)

    cfg = Config()

    assert cfg.trade_symbol == "BTCUSDT"
    assert cfg.trade_interval == "4h"
    assert isinstance(cfg.trade_quantity, float)
    assert cfg.trade_quantity == 0.001


def test_custom_env(monkeypatch):
    monkeypatch.setenv("TRADE_SYMBOL", "BTCUSDT")
    monkeypatch.setenv("TRADE_INTERVAL", "4h")
    monkeypatch.setenv("TRADE_QUANTITY", "0.001")
    monkeypatch.setenv("INITIAL_BALANCE", "10000")
    monkeypatch.setenv("EMAIL_PORT", "587")

    cfg = Config()

    assert cfg.trade_symbol == "BTCUSDT"
    assert cfg.trade_interval == "4h"
    assert cfg.trade_quantity == 0.001
    assert cfg.initial_balance == 10000
    assert cfg.email_port == 587


def test_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/trading")
    cfg = Config()
    assert cfg.database_url.startswith("postgresql://user:pass@localhost:5432/trading")


def test_jwt_settings(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "supersecretkey123")
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")

    cfg = Config()

    assert cfg.jwt_secret_key == "JWT_SECRET_KEY", "supersecretkey123"
    assert cfg.access_token_expire_minutes == 60