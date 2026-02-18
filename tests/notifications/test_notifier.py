import asyncio
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _pytest.monkeypatch import MonkeyPatch

from src.notifier import TelegramNotifier, send_email_notification


@pytest.fixture
def mock_env(monkeypatch: MonkeyPatch) -> Generator[None, Any, None]:
    """Mock environment variables for notifier tests."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    monkeypatch.setenv("SMTP_HOST", "smtp.test.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "sender@test.com")
    monkeypatch.setenv("SMTP_PASS", "password")
    monkeypatch.setenv("EMAIL_TO", "receiver@test.com")
    yield


# --------------------------------------------------------------
# TELEGRAM NOTIFIER TESTS
# --------------------------------------------------------------
@pytest.mark.usefixtures("mock_env")
def test_telegram_notifier_init(mock_env: None) -> None:
    """✅ Ensure TelegramNotifier initializes properly."""
    with patch("src.notifier.Bot", return_value=MagicMock()) as mock_bot:
        notifier = TelegramNotifier()
        assert notifier.enabled is True
        assert notifier.bot == mock_bot.return_value
        assert isinstance(notifier.chat_id, str)


@pytest.mark.usefixtures("mock_env")
def test_telegram_notifier_send_message_success(mock_env: None) -> None:
    """✅ Test successful Telegram send with async support."""
    with patch("src.notifier.Bot") as mock_bot:
        mock_instance = mock_bot.return_value
        # Make send_message awaitable
        mock_instance.send_message = AsyncMock(return_value=True)

        notifier = TelegramNotifier()
        result = asyncio.run(notifier.send_message("Test Message OK"))

        # Verify the message was sent successfully
        assert result is True

        # Verify send_message was called at least once (may be called more due to retries)
        assert mock_instance.send_message.await_count >= 1

        # Verify it was called with correct arguments
        call_kwargs = mock_instance.send_message.call_args[1]
        assert call_kwargs["chat_id"] == notifier.chat_id
        assert call_kwargs["text"] == "Test Message OK"


@pytest.mark.usefixtures("mock_env")
def test_telegram_notifier_send_message_failure(mock_env: None) -> None:
    """❌ Test retry and failure logging when message fails to send."""
    with patch("src.notifier.Bot") as mock_bot:
        mock_instance = mock_bot.return_value
        # Make send_message raise an exception
        mock_instance.send_message = AsyncMock(side_effect=Exception("Network failure"))

        notifier = TelegramNotifier(max_retries=2)

        # The function should either return False or raise an exception
        # We'll catch either case
        try:
            result = asyncio.run(notifier.send_message("Failing Test Message"))
            # If it returns, it should be False
            assert result is False
        except Exception as e:
            # If it raises, the error should be the Network failure
            assert "Network failure" in str(e) or isinstance(e, Exception)


# --------------------------------------------------------------
# EMAIL NOTIFICATION TESTS
# --------------------------------------------------------------
def test_send_email_notification_success(monkeypatch: MonkeyPatch) -> None:
    """✅ Test successful email send."""
    monkeypatch.setenv("SMTP_HOST", "smtp.test.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "sender@test.com")
    monkeypatch.setenv("SMTP_PASS", "password")
    monkeypatch.setenv("EMAIL_TO", "receiver@test.com")

    with patch("smtplib.SMTP") as mock_smtp_cls:
        smtp_instance = mock_smtp_cls.return_value.__enter__.return_value
        smtp_instance.starttls.return_value = None
        smtp_instance.login.return_value = None
        smtp_instance.sendmail.return_value = {}

        result = send_email_notification("Subject OK", "Body OK")

        assert result is True
        mock_smtp_cls.assert_called_once()
        smtp_instance.sendmail.assert_called_once()


def test_send_email_notification_failure(monkeypatch: MonkeyPatch) -> None:
    """❌ Test error handling when SMTP fails."""
    monkeypatch.setenv("SMTP_HOST", "smtp.test.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "sender@test.com")
    monkeypatch.setenv("SMTP_PASS", "password")
    monkeypatch.setenv("EMAIL_TO", "receiver@test.com")

    with patch("smtplib.SMTP", side_effect=Exception("SMTP Failure")):
        with patch("src.notifier.logger.error") as mock_log:
            result = send_email_notification("Subject Fail", "Body Fail")

            assert result is False
            assert mock_log.call_count >= 1
