import os
import pytest
import logging
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Callable

from src.notifier import TelegramNotifier, send_email_notification


# --------------------------------------------------
# FIXTURES
# --------------------------------------------------
@pytest.fixture
def mock_bot():
    """Fixture to create a mock async Telegram bot instance."""
    bot = AsyncMock()
    bot.send_message = AsyncMock()
    return bot


# --------------------------------------------------
# TELEGRAM NOTIFIER TESTS
# --------------------------------------------------
@pytest.mark.asyncio
async def test_telegram_notifier_init(mock_bot):
    """Should initialize TelegramNotifier with a mock bot successfully."""
    with patch("src.notifier.Bot", return_value=mock_bot):
        notifier = TelegramNotifier()
        assert notifier.enabled is True
        assert notifier.bot == mock_bot
        assert hasattr(notifier, "chat_id")
        assert isinstance(notifier.chat_id, str)


@pytest.mark.asyncio
async def test_telegram_notifier_send_message_success(mock_bot):
    """Should send message successfully using mocked Telegram bot."""
    with patch("src.notifier.Bot", return_value=mock_bot):
        notifier = TelegramNotifier()
        await notifier.send_message("Test Message")

        mock_bot.send_message.assert_awaited_once_with(
            chat_id=notifier.chat_id,
            text="Test Message"
        )


@pytest.mark.asyncio
async def test_telegram_notifier_send_message_failure(mock_bot):
    """Should log an error when Telegram message sending fails."""
    with patch("src.notifier.Bot", return_value=mock_bot):
        mock_bot.send_message.side_effect = Exception("Mocked error")
        notifier = TelegramNotifier()

        with patch("src.notifier.logger.error") as mock_log:
            await notifier.send_message("Test Message")
            mock_log.assert_called_once_with("Failed to send Telegram message: Mocked error")


# --------------------------------------------------
# EMAIL NOTIFICATION TESTS
# --------------------------------------------------
def test_send_email_notification_success():
    """Should send email successfully via mocked SMTP connection."""
    with patch("smtplib.SMTP") as mock_smtp_cls:
        mock_smtp_instance = mock_smtp_cls.return_value.__enter__.return_value
        mock_smtp_instance.starttls.return_value = None
        mock_smtp_instance.login.return_value = None
        mock_smtp_instance.sendmail.return_value = {}

        with patch.dict(os.environ, {
            "EMAIL_HOST": "mock.smtp.com",
            "EMAIL_PORT": "587",
            "EMAIL_USER": "mock@example.com",
            "EMAIL_PASS": "mock_pass",
            "EMAIL_TO": "mock_to@example.com",
        }):
            send_email_notification("Test Subject", "Test Message")

            mock_smtp_cls.assert_called_once_with("mock.smtp.com", 587)
            mock_smtp_instance.starttls.assert_called_once()
            mock_smtp_instance.login.assert_called_once_with("mock@example.com", "mock_pass")

            # Match MIME formatting instead of raw string
            assert mock_smtp_instance.sendmail.called
            args, kwargs = mock_smtp_instance.sendmail.call_args
            assert "Subject: Test Subject" in args[2]
            assert "Test Message" in args[2]


def test_send_email_notification_failure():
    """Should log an error if email sending fails."""
    with patch("smtplib.SMTP", side_effect=Exception("Mocked error")):
        with patch.dict(os.environ, {
            "EMAIL_HOST": "mock.smtp.com",
            "EMAIL_PORT": "587",
            "EMAIL_USER": "mock@example.com",
            "EMAIL_PASS": "mock_pass",
            "EMAIL_TO": "mock_to@example.com",
        }):
            with patch("src.notifier.logger.error") as mock_log:
                send_email_notification("Test Subject", "Test Message")
                mock_log.assert_called_once_with("Failed to send email notification: Mocked error")
