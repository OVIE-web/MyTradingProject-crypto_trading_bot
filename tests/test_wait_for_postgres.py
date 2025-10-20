import os
import pytest
from unittest.mock import patch, MagicMock
from src.wait_for_postgres import wait_for_postgres

# --------------- SUCCESS CASE ----------------
@patch("psycopg2.connect")
def test_wait_for_postgres_success(mock_connect, monkeypatch):
    """Should return True when DB connection succeeds immediately."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://testuser:testpass@localhost:5432/tradingbot_test")

    mock_connect.return_value = MagicMock()  # simulate successful connection

    result = wait_for_postgres()

    assert result is True
    mock_connect.assert_called_once()


# --------------- FAILURE CASE ----------------
@patch("psycopg2.connect", side_effect=Exception("Connection failed"))
def test_wait_for_postgres_failure(mock_connect, monkeypatch):
    """Should raise TimeoutError when DB never becomes available."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://testuser:testpass@localhost:5432/tradingbot_test")

    with pytest.raises(TimeoutError):
        wait_for_postgres()


# --------------- ENV VAR MISSING CASE ----------------
def test_wait_for_postgres_missing_env(monkeypatch):
    """Should raise ValueError if DATABASE_URL is missing."""
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValueError):
        wait_for_postgres()
