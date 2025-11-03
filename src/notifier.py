# src/notifier.py
import os
import smtplib
import logging
from email.mime.text import MIMEText
from telegram import Bot

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)
        self.bot = Bot(token=self.token) if self.enabled else None
        logger.info("Telegram Notifier initialized.")

    async def send_message(self, message: str):
        """Send a Telegram message asynchronously."""
        if not self.enabled:
            logger.warning("Telegram Notifier not enabled.")
            return
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
            logger.info(f"Telegram message sent: '{message}'")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")


def send_email_notification(subject: str, message: str):
    """Send an email notification."""
    try:
        host = os.getenv("EMAIL_HOST")
        port = int(os.getenv("EMAIL_PORT", 587))
        user = os.getenv("EMAIL_USER")
        password = os.getenv("EMAIL_PASS")
        to_email = os.getenv("EMAIL_TO")

        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to_email

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, to_email, msg.as_string())

        logger.info(f"✅ Email sent successfully: {subject}")

    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")
