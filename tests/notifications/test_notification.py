# File: tests/notifications/test_notifications.py
import logging
from unittest.mock import patch, MagicMock
from src.notification import send_telegram_notification, send_email_notification


# =====================================================================================
# TELEGRAM TESTS
# =====================================================================================
def test_send_telegram_notification_success(mock_env):
    """✅ Should send Telegram message successfully."""
    with patch("requests.post", return_value=MagicMock(status_code=200)) as mock_post:
        send_telegram_notification("Test OK")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "botmock_token" in args[0]
        assert kwargs["json"]["text"] == "Test OK"


def test_send_telegram_notification_http_error(mock_env):
    """⚠️ Should log error when Telegram API fails."""
    with patch("requests.post", return_value=MagicMock(status_code=500)):
        with patch("logging.error") as mock_log:
            send_telegram_notification("Bad HTTP")
            assert mock_log.call_count >= 1
            assert "Telegram API error" in str(mock_log.call_args_list[0][0][0])


def test_send_telegram_notification_missing_config():
    """❌ Should log config error when environment missing."""
    with patch("logging.error") as mock_log:
        send_telegram_notification("No Config")
        assert mock_log.call_count >= 1
        assert "Missing Telegram configuration" in str(mock_log.call_args[0][0])


# =====================================================================================
# EMAIL TESTS
# =====================================================================================
def test_send_email_notification_success(mock_env):
    """✅ Should send email successfully using mock SMTP."""
    with patch("smtplib.SMTP") as mock_smtp_cls:
        smtp_instance = mock_smtp_cls.return_value.__enter__.return_value
        smtp_instance.sendmail.return_value = {}

        send_email_notification("OK", "Body test message")
        smtp_instance.sendmail.assert_called_once()
        args, _ = smtp_instance.sendmail.call_args
        assert "Subject: OK" in args[2]
        assert "Body test message" in args[2]


def test_send_email_notification_missing_config():
    """❌ Should log error when email environment is incomplete."""
    with patch("logging.error") as mock_log:
        send_email_notification("No Env", "Missing Config")
        assert mock_log.call_count >= 1
        assert "Missing email configuration" in str(mock_log.call_args[0][0])
