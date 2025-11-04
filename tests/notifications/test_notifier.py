# File: tests/notifications/test_notifier.py
import os
import pytest
import logging
from unittest.mock import patch, MagicMock
from src.notifier import TelegramNotifier, send_email_notification


# --------------------------------------------------------------
# TELEGRAM NOTIFIER TESTS
# --------------------------------------------------------------
def test_telegram_notifier_init(mock_env):
    """✅ Ensure TelegramNotifier initializes with environment vars."""
    with patch("telegram.Bot", return_value=MagicMock()) as mock_bot:
        notifier = TelegramNotifier()
        assert notifier.enabled is True
        assert notifier.bot == mock_bot.return_value
        assert isinstance(notifier.chat_id, str)
        assert notifier.chat_id


def test_telegram_notifier_send_message_success(mock_env):
    """✅ Should send Telegram message successfully."""
    with patch("telegram.Bot", return_value=MagicMock()) as mock_bot:
        notifier = TelegramNotifier()
        notifier.send_message("Test Message OK")
        mock_bot.return_value.send_message.assert_called_once_with(
            chat_id=notifier.chat_id,
            text="Test Message OK"
        )


def test_telegram_notifier_send_message_failure(mock_env):
    """⚠️ Should log error when Telegram send fails."""
    with patch("telegram.Bot", return_value=MagicMock()) as mock_bot:
        mock_bot.return_value.send_message.side_effect = Exception("Mocked Failure")
        notifier = TelegramNotifier()

        with patch("logging.error") as mock_log:
            notifier.send_message("Failing Test Message")
            assert mock_log.call_count >= 1
            first_message = mock_log.call_args_list[0][0][0]
            assert "Failed to send Telegram message" in first_message


# --------------------------------------------------------------
# EMAIL NOTIFICATION TESTS
# --------------------------------------------------------------
def test_send_email_notification_success(mock_env):
    """✅ Should send email successfully with proper config."""
    with patch("smtplib.SMTP") as mock_smtp_cls:
        smtp_instance = mock_smtp_cls.return_value.__enter__.return_value
        smtp_instance.starttls.return_value = None
        smtp_instance.login.return_value = None
        smtp_instance.sendmail.return_value = {}

        send_email_notification("Subject OK", "Body OK")

        mock_smtp_cls.assert_called_once_with("smtp.test.com", 587)
        smtp_instance.starttls.assert_called_once()
        smtp_instance.login.assert_called_once_with("sender@test.com", "password")
        smtp_instance.sendmail.assert_called_once()


def test_send_email_notification_failure(mock_env):
    """❌ Should log error if SMTP connection fails."""
    with patch("smtplib.SMTP", side_effect=Exception("SMTP Failure")):
        with patch("logging.error") as mock_log:
            send_email_notification("Subject Fail", "Body Fail")
            assert mock_log.call_count >= 1
            assert "Failed to send email notification" in str(mock_log.call_args[0][0])
