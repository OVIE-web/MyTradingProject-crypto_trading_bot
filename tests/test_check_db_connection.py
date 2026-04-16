"""
Tests for src/check_db_connection.py module.
Tests database connection waiting functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.check_db_connection import wait_for_postgres


class TestWaitForPostgres:
    """Test cases for wait_for_postgres function."""

    def test_wait_for_postgres_success_first_try(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful connection on first attempt."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

        with patch("psycopg2.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn

            result = wait_for_postgres()

            assert result is True
            mock_connect.assert_called_once_with("postgresql://test:test@localhost:5432/test")
            mock_conn.close.assert_called_once()

    def test_wait_for_postgres_success_after_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful connection after some failed attempts."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

        with (
            patch("psycopg2.connect") as mock_connect,
            patch("time.sleep") as mock_sleep,
            patch("src.check_db_connection.logging") as mock_logging,
        ):
            # First 2 attempts fail, 3rd succeeds
            mock_connect.side_effect = [
                Exception("Connection failed"),
                Exception("Connection failed"),
                MagicMock(),  # Success
            ]

            result = wait_for_postgres()

            assert result is True
            assert mock_connect.call_count == 3
            assert mock_sleep.call_count == 2  # Only sleep on failures
            mock_sleep.assert_called_with(3)

    def test_wait_for_postgres_timeout_after_max_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test timeout when connection fails for all 30 attempts."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

        with patch("psycopg2.connect") as mock_connect, patch("time.sleep") as mock_sleep:
            mock_connect.side_effect = Exception("Connection failed")

            with pytest.raises(TimeoutError, match="Database not reachable after 90 seconds"):
                wait_for_postgres()

            assert mock_connect.call_count == 30
            assert mock_sleep.call_count == 30

    def test_wait_for_postgres_no_database_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test error when DATABASE_URL environment variable is not set."""
        monkeypatch.delenv("DATABASE_URL", raising=False)

        with pytest.raises(ValueError, match="DATABASE_URL is not set in environment variables"):
            wait_for_postgres()

    def test_wait_for_postgres_empty_database_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test error when DATABASE_URL is empty."""
        monkeypatch.setenv("DATABASE_URL", "")

        with pytest.raises(ValueError, match="DATABASE_URL is not set in environment variables"):
            wait_for_postgres()
