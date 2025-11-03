import os
import smtplib
import logging
import asyncio
import random
from email.mime.text import MIMEText
from telegram import Bot

logger = logging.getLogger(__name__)


def _backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """
    Exponential backoff with jitter.
    attempt: current retry attempt (1-based)
    base: base delay in seconds
    cap: maximum delay in seconds
    """
    exp = min(cap, base * (2 ** (attempt - 1)))
    # Add jitter so retries from multiple bots don't synchronize
    return exp / 2 + random.uniform(0, exp / 2)


class TelegramNotifier:
    def __init__(self, max_retries: int = 3):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)
        self.bot = Bot(token=self.token) if self.enabled else None
        self.max_retries = max_retries
        logger.info("Telegram Notifier initialized.")

    async def send_message(self, message: str):
        """Send a Telegram message asynchronously with retry + backoff."""
        if not self.enabled:
            logger.warning("Telegram Notifier not enabled.")
            return

        for attempt in range(1, self.max_retries + 1):
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=message)
                logger.info(f"Telegram message sent: '{message}'")
                return
            except Exception as e:
                logger.error(f"Attempt {attempt} failed to send Telegram message: {e}")
                if attempt < self.max_retries:
                    delay = _backoff_delay(attempt)
                    logger.warning(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("All retry attempts exhausted. Message not sent.")


def send_email_notification(subject: str, message: str, max_retries: int = 3):
    """Send an email notification with retry + backoff."""
    host = os.getenv("EMAIL_HOST")
    port = int(os.getenv("EMAIL_PORT", 587))
    user = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")
    to_email = os.getenv("EMAIL_TO")

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_email

    for attempt in range(1, max_retries + 1):
        try:
            with smtplib.SMTP(host, port) as server:
                server.starttls()
                server.login(user, password)
                server.sendmail(user, to_email, msg.as_string())
            logger.info(f"✅ Email sent successfully: {subject}")
            return
        except Exception as e:
            logger.error(f"Attempt {attempt} failed to send email: {e}")
            if attempt < max_retries:
                delay = _backoff_delay(attempt)
                logger.warning(f"Retrying email in {delay:.2f} seconds...")
                # For sync code, use time.sleep instead of asyncio.sleep
                import time
                time.sleep(delay)
            else:
                logger.error("All retry attempts exhausted. Email not sent.")