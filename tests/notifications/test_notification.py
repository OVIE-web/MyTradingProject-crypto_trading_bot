from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pytest import MonkeyPatch

from src.notification import send_email_notification, send_telegram_notification


@pytest.fixture
def mock_env(monkeypatch: MonkeyPatch) -> Generator[None, Any, None]:
    """Mock environment variables for consistent notification tests."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "mock_token_123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "mock_chat_id_456")
    monkeypatch.setenv("SMTP_HOST", "smtp.test.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "sender@test.com")
    monkeypatch.setenv("SMTP_PASS", "password123")
    monkeypatch.setenv("EMAIL_TO", "receiver@test.com")
    yield


# =====================================================================================
# TELEGRAM TESTS
# =====================================================================================


@pytest.mark.usefixtures("mock_env")
def test_send_telegram_notification_success(mock_env: None) -> None:
    """✅ Should send Telegram message successfully."""
    with patch("src.notification.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)

        send_telegram_notification("Test OK")

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        # Verify the Telegram API URL was called
        assert "botmock_token_123" in args[0]
        assert kwargs["json"]["text"] == "Test OK"


@pytest.mark.usefixtures("mock_env")
def test_send_telegram_notification_http_error(mock_env: None) -> None:
    """⚠️ Should log error when Telegram API fails."""
    with patch("src.notification.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=500)

        with patch("src.notification.logger.error") as mock_log:
            send_telegram_notification("Bad HTTP")

            # Verify error was logged
            assert mock_log.call_count >= 1
            # Check that error message mentions Telegram API
            error_msg = str(mock_log.call_args_list[0])
            assert "Telegram" in error_msg or "API" in error_msg or "500" in error_msg


def test_send_telegram_notification_missing_config(monkeypatch: MonkeyPatch) -> None:
    """❌ Should log error when environment variables are missing."""
    # Remove required Telegram environment variables
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    with patch("src.notification.logger.error") as mock_log:
        send_telegram_notification("No Config")

        assert mock_log.call_count >= 1
        error_msg = str(mock_log.call_args[0][0])
        assert "Telegram" in error_msg or "missing" in error_msg.lower()


# =====================================================================================
# EMAIL TESTS
# =====================================================================================


@pytest.mark.usefixtures("mock_env")
def test_send_email_notification_success(mock_env: None) -> None:
    """✅ Should send email successfully using mock SMTP."""
    with patch("smtplib.SMTP") as mock_smtp_cls:
        # Setup mock SMTP instance
        mock_smtp_instance = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = mock_smtp_instance
        mock_smtp_instance.sendmail.return_value = {}

        send_email_notification("Test Subject", "Test body message")

        # Verify SMTP sendmail was called
        mock_smtp_instance.sendmail.assert_called_once()

        # Extract the message content
        call_args = mock_smtp_instance.sendmail.call_args
        sender, recipient, message = call_args[0]

        # Verify message content
        assert "Subject: Test Subject" in message
        assert "Test body message" in message
        # Just verify email addresses are not empty and are strings
        assert isinstance(sender, str) and len(sender) > 0
        assert isinstance(recipient, str) and len(recipient) > 0
        # Verify SMTP connection was established
        mock_smtp_cls.assert_called_once()


def test_send_email_notification_missing_config(monkeypatch: MonkeyPatch) -> None:
    """❌ Should log error when email environment is incomplete."""
    email_vars = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "EMAIL_TO"]

    # Remove all email config variables
    for var in email_vars:
        monkeypatch.delenv(var, raising=False)

    with patch("src.notification.logger.error") as mock_log:
        send_email_notification("No Env", "Missing Config")

        assert mock_log.call_count >= 1
        error_msg = str(mock_log.call_args[0][0])
        assert "email" in error_msg.lower() or "missing" in error_msg.lower()
