"""Tests for the PostgreSQL connection waiting logic in app.db.wait_for_postgres."""

from __future__ import annotations

import importlib

import pytest
from psycopg2 import OperationalError

wait_module = importlib.import_module("app.db.wait_for_postgres")


def test_wait_for_postgres_returns_true_on_first_success(mocker, monkeypatch) -> None:
    """A successful first connection should close the connection and return True."""
    database_url = "postgresql://user:pass@localhost:5432/trading"
    monkeypatch.setattr(wait_module, "DATABASE_URL", database_url)
    mock_connection = mocker.MagicMock()
    mock_connect = mocker.patch.object(
        wait_module.psycopg2,
        "connect",
        return_value=mock_connection,
    )

    result = wait_module.wait_for_postgres(max_retries=1, delay=0)

    assert result is True
    mock_connect.assert_called_once_with(database_url)
    mock_connection.close.assert_called_once_with()


def test_wait_for_postgres_retries_operational_errors(mocker, monkeypatch) -> None:
    """Operational errors should be retried until a connection succeeds."""
    database_url = "postgresql://user:pass@localhost:5432/trading"
    monkeypatch.setattr(wait_module, "DATABASE_URL", database_url)
    mock_connection = mocker.MagicMock()
    mocker.patch.object(
        wait_module.psycopg2,
        "connect",
        side_effect=[OperationalError("not ready"), mock_connection],
    )
    mock_sleep = mocker.patch.object(wait_module.time, "sleep")

    result = wait_module.wait_for_postgres(max_retries=2, delay=1)

    assert result is True
    mock_sleep.assert_called_once_with(1)
    mock_connection.close.assert_called_once_with()


def test_wait_for_postgres_raises_timeout_after_retries(mocker, monkeypatch) -> None:
    """A database that never becomes available should raise TimeoutError."""
    monkeypatch.setattr(wait_module, "DATABASE_URL", "postgresql://user:pass@localhost/db")
    mock_connect = mocker.patch.object(
        wait_module.psycopg2,
        "connect",
        side_effect=OperationalError("not ready"),
    )
    mock_sleep = mocker.patch.object(wait_module.time, "sleep")

    with pytest.raises(TimeoutError, match="PostgreSQL not reachable after 3 attempts"):
        wait_module.wait_for_postgres(max_retries=3, delay=2)

    assert mock_connect.call_count == 3
    assert mock_sleep.call_count == 2


def test_wait_for_postgres_rejects_missing_database_url(monkeypatch) -> None:
    """Missing configuration should fail before any connection attempt."""
    monkeypatch.setattr(wait_module, "DATABASE_URL", "")

    with pytest.raises(ValueError, match="DATABASE_URL is not set."):
        wait_module.wait_for_postgres()


def test_wait_for_postgres_does_not_swallow_unexpected_errors(mocker, monkeypatch) -> None:
    """Only psycopg2 OperationalError is retried by the wait loop."""
    monkeypatch.setattr(wait_module, "DATABASE_URL", "postgresql://user:pass@localhost/db")
    mocker.patch.object(
        wait_module.psycopg2,
        "connect",
        side_effect=RuntimeError("unexpected"),
    )
    mock_sleep = mocker.patch.object(wait_module.time, "sleep")

    with pytest.raises(RuntimeError, match="unexpected"):
        wait_module.wait_for_postgres(max_retries=3, delay=1)

    mock_sleep.assert_not_called()
