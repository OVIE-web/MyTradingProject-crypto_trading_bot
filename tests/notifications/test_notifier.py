# tests/notifications/test_notifier.py
import os
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.notifier import TelegramNotifier, send_email_notification


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    yield


# --------------------------------------------------------------
# TELEGRAM NOTIFIER TESTS
# --------------------------------------------------------------
def test_telegram_notifier_init(mock_env):
    """✅ Ensure TelegramNotifier initializes properly."""
    with patch("src.notifier.Bot", return_value=MagicMock()) as mock_bot:
        notifier = TelegramNotifier()
        assert notifier.enabled is True
        assert notifier.bot == mock_bot.return_value
        assert isinstance(notifier.chat_id, str)


def test_telegram_notifier_send_message_success(mock_env):
    """✅ Test successful Telegram send with async support."""
    with patch("src.notifier.Bot") as mock_bot:
        mock_instance = mock_bot.return_value
        # make send_message awaitable
        mock_instance.send_message = AsyncMock(return_value=True)

        notifier = TelegramNotifier()
        result = asyncio.run(notifier.send_message("Test Message OK"))

        assert result is True
        mock_instance.send_message.assert_awaited_once_with(
            chat_id=notifier.chat_id,
            text="Test Message OK"
        )


def test_telegram_notifier_send_message_failure(mock_env):
    """❌ Test retry and failure logging."""
    with patch("src.notifier.Bot") as mock_bot:
        mock_instance = mock_bot.return_value
        mock_instance.send_message.side_effect = Exception("Network failure")

        notifier = TelegramNotifier(max_retries=2)
        result = asyncio.run(notifier.send_message("Failing Test Message"))
        assert result is False


# --------------------------------------------------------------
# EMAIL NOTIFICATION TESTS
# --------------------------------------------------------------
def test_send_email_notification_success(monkeypatch):
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

        assert send_email_notification("Subject OK", "Body OK") is True
        mock_smtp_cls.assert_called_once_with("smtp.test.com", 587)
        smtp_instance.login.assert_called_once_with("sender@test.com", "password")


def test_send_email_notification_failure(mock_env):
    """❌ Should log error if SMTP fails."""
    with patch("smtplib.SMTP", side_effect=Exception("SMTP Failure")):
        with patch("src.notifier.logger.error") as mock_log:
            result = send_email_notification("Subject Fail", "Body Fail")
            assert result is False
            logged = "Email not sent" in str(mock_log.call_args_list[-1][0][0])
            assert logged
