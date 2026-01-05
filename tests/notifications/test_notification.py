from unittest.mock import MagicMock, patch

import pytest

from src.notification import send_email_notification, send_telegram_notification


@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables for consistent notification tests."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "mock_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "mock_chat_id")
    monkeypatch.setenv("EMAIL_HOST", "smtp.test.com")
    monkeypatch.setenv("EMAIL_PORT", "587")
    monkeypatch.setenv("EMAIL_USER", "sender@test.com")
    monkeypatch.setenv("EMAIL_PASS", "password")
    monkeypatch.setenv("EMAIL_TO", "receiver@test.com")
    yield


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
        with patch("src.notification.logger.error") as mock_log:
            send_telegram_notification("Bad HTTP")
            assert mock_log.call_count >= 1
            assert any("Telegram API error" in str(c[0][0]) for c in mock_log.call_args_list)


def test_send_telegram_notification_missing_config(monkeypatch):
    """❌ Should log config error when environment missing."""
    for var in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]:
        monkeypatch.delenv(var, raising=False)

    with patch("src.notification.logger.error") as mock_log:
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


def test_send_email_notification_missing_config(monkeypatch):
    """❌ Should log error when email environment is incomplete."""
    for var in ["EMAIL_HOST", "EMAIL_PORT", "EMAIL_USER", "EMAIL_PASS", "EMAIL_TO"]:
        monkeypatch.delenv(var, raising=False)

    with patch("src.notification.logger.error") as mock_log:
        send_email_notification("No Env", "Missing Config")
        assert mock_log.call_count >= 1
        assert "Missing email configuration" in str(mock_log.call_args[0][0])
