from unittest.mock import MagicMock, Mock, patch

import pytest
from psycopg2 import OperationalError
from pytest import MonkeyPatch

from src.wait_for_postgres import DATABASE_URL, wait_for_postgres


# --------------- SUCCESS CASE ----------------
@patch("psycopg2.connect")
def test_wait_for_postgres_success(mock_connect: Mock, monkeypatch: MonkeyPatch) -> None:
    """Should return True when DB connection succeeds immediately."""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    # Use monkeypatch to mock modeule src.wait_for_postgres.DATABASE_URL
    monkeypatch.setattr("src.wait_for_postgres.DATABASE_URL", DATABASE_URL)

    result = wait_for_postgres(max_retries=1)

    assert result is True
    mock_connect.assert_called_once_with(DATABASE_URL)
    mock_conn.close.assert_called_once()


# --------------- RETRY SUCCESS CASE ----------------
@patch("psycopg2.connect")
@patch("time.sleep")
def test_wait_for_postgres_retry_success(
    mock_sleep: Mock, mock_connect: Mock, monkeypatch: MonkeyPatch
) -> None:
    """Should retry and succeed when DB becomes available."""
    mock_conn = MagicMock()
    mock_connect.side_effect = [OperationalError("connection failed"), mock_conn]

    # Use monkeypatch to mock modeule src.wait_for_postgres.DATABASE_URL
    monkeypatch.setattr("src.wait_for_postgres.DATABASE_URL", DATABASE_URL)
    result = wait_for_postgres(max_retries=2, delay=1)

    assert result is True
    assert mock_connect.call_count == 2
    mock_sleep.assert_called_once_with(1)
    mock_conn.close.assert_called_once()


# --------------- FAILURE CASE ----------------
@patch("psycopg2.connect")
@patch("time.sleep")
def test_wait_for_postgres_failure(
    mock_sleep: Mock, mock_connect: Mock, monkeypatch: MonkeyPatch
) -> None:
    """Should raise TimeoutError when DB never becomes available."""
    mock_connect.side_effect = OperationalError("connection failed")
    max_retries = 3

    with pytest.raises(TimeoutError) as exc_info:
        wait_for_postgres(max_retries=max_retries, delay=1)

    assert f"Could not connect to Postgres after {max_retries} attempts" in str(exc_info.value)
    assert mock_connect.call_count == max_retries
    assert mock_sleep.call_count == max_retries - 1  # No sleep after last attempt


# --------------- ENV VAR MISSING CASE ----------------
def test_wait_for_postgres_missing_url() -> None:
    """Should raise ValueError if DATABASE_URL is missing."""
    with patch("src.wait_for_postgres.DATABASE_URL", None):
        with pytest.raises(ValueError) as exc_info:
            wait_for_postgres()

    assert "DATABASE_URL environment variable is not set" in str(exc_info.value)
