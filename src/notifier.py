import os
import smtplib
import logging
from email.mime.text import MIMEText
from telegram import Bot

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def send_email_notification(subject: str, message: str):
    """Send an email notification using environment variables."""
    try:
        host = os.getenv("EMAIL_HOST")
        port = int(os.getenv("EMAIL_PORT", "587"))
        user = os.getenv("EMAIL_USER")
        password = os.getenv("EMAIL_PASS")
        recipient = os.getenv("EMAIL_TO")

        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = recipient

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, recipient, msg.as_string())

        logger.info(f"✅ Email sent successfully: {subject}")
    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")


class TelegramNotifier:
    """Simple Telegram Bot wrapper for notifications."""

    def __init__(self):
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(token and chat_id)

        if self.enabled:
            self.bot = Bot(token=token)
            self.chat_id = chat_id
            logger.info("Telegram Notifier initialized.")
        else:
            self.bot = None
            self.chat_id = None
            logger.warning("Telegram notifier disabled: missing token or chat ID.")

    def send_message(self, message: str):
        """Send message to Telegram chat."""
        if not self.enabled or not self.bot:
            logger.warning("Telegram notifier is disabled or uninitialized.")
            return

        try:
            # Call send_message synchronously for test mock compatibility
            self.bot.send_message(chat_id=self.chat_id, text=message)
            logger.info(f"Telegram message sent: '{message}'")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
