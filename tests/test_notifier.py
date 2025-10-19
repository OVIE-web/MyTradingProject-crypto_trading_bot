# tests/test_notifier.py
from unittest.mock import patch
from MyTradingProject.crypto_trading_bot.src.notifier import send_email_notification, TelegramNotifier

def test_send_email_notification_success():
    with patch('smtplib.SMTP') as mock_smtp_cls:
        mock_smtp_instance = mock_smtp_cls.return_value
        mock_smtp_instance.starttls.return_value = None
        mock_smtp_instance.login.return_value = None
        mock_smtp_instance.sendmail.return_value = {}

        with patch.dict('os.environ', {
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
            mock_smtp_instance.sendmail.assert_called_once_with("mock@example.com", ["mock_to@example.com"], 
                                                               "Subject: Test Subject\r\n\r\nTest Message")

def test_send_email_notification_failure():
    with patch('smtplib.SMTP') as mock_smtp_cls:
        mock_smtp_instance = mock_smtp_cls.return_value
        mock_smtp_instance.starttls.return_value = None
        mock_smtp_instance.login.return_value = None
        mock_smtp_instance.sendmail.side_effect = Exception("Mocked error")

        with patch.dict('os.environ', {
            "EMAIL_HOST": "mock.smtp.com",
            "EMAIL_PORT": "587",
            "EMAIL_USER": "mock@example.com",
            "EMAIL_PASS": "mock_pass",
            "EMAIL_TO": "mock_to@example.com",
        }):
            with patch('logging.error') as mock_logging_error:
                send_email_notification("Test Subject", "Test Message")
                mock_logging_error.assert_called_once_with("Failed to send email notification: Mocked error")

def test_telegram_notifier_init():
    with patch('telegram.Bot') as mock_bot_cls:
        mock_bot_instance = mock_bot_cls.return_value
        notifier = TelegramNotifier()
        assert notifier.enabled == True
        assert notifier.bot == mock_bot_instance
        assert notifier.chat_id == "mock_chat_id"

def test_telegram_notifier_send_message_success():
    with patch('telegram.Bot') as mock_bot_cls:
        mock_bot_instance = mock_bot_cls.return_value
        mock_bot_instance.send_message.return_value = None
        notifier = TelegramNotifier()
        notifier.send_message("Test Message")
        mock_bot_instance.send_message.assert_called_once_with(chat_id="mock_chat_id", text="Test Message")

def test_telegram_notifier_send_message_failure():
    with patch('telegram.Bot') as mock_bot_cls:
        mock_bot_instance = mock_bot_cls.return_value
        mock_bot_instance.send_message.side_effect = Exception("Mocked error")
        notifier = TelegramNotifier()
        with patch('logging.error') as mock_logging_error:
            notifier.send_message("Test Message")
            mock_logging_error.assert_called_once_with("Failed to send Telegram message: Mocked error")