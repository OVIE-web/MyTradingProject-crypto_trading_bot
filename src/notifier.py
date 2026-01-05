# src/notifier.py
import asyncio
import logging
import os
import random
import smtplib
import time
import typing
from email.mime.text import MIMEText
from typing import Optional

from telegram import Bot

logger = logging.getLogger(__name__)


def _backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Calculate exponential backoff with jitter."""
    exp = min(cap, base * (2 ** (attempt - 1)))
    return exp / 2 + random.uniform(0, exp / 2)


class TelegramNotifier:
    """Handles asynchronous Telegram notifications with retries."""

    def __init__(self, max_retries: int = 3):
        self.token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)
        self.bot = Bot(token=typing.cast(str, self.token)) if self.enabled else None
        self.max_retries = max_retries
        logger.info("Telegram Notifier initialized.")

    async def send_message(self, message: str) -> bool:
        """Send Telegram message asynchronously with retry mechanism."""
        if not self.enabled:
            logger.warning("Telegram Notifier not enabled.")
            return False

        if self.bot is None:
            logger.error("Telegram Bot is not initialized.")
            return False

        if self.chat_id is not None:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
        else:
            logger.error("Missing Telegram chat ID.")
            return False

        for attempt in range(1, self.max_retries + 1):
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=message)
                logger.info(f"Telegram message sent: '{message}'")
                return True
            except Exception as e:
                logger.error(f"Attempt {attempt} failed to send Telegram message: {e}")
                if attempt < self.max_retries:
                    delay = _backoff_delay(attempt)
                    logger.warning(f"Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("All retry attempts exhausted for Telegram message.")
        return False


def send_email_notification(subject: str, message: str, max_retries: int = 3) -> bool:
    """Send email using SMTP with retry logic."""
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", 587))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    to_email = os.getenv("EMAIL_TO")

    if not (host and user and password and to_email):
        logger.error("Missing SMTP configuration.")
        return False

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
            logger.info(f"✅ Email notification sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Attempt {attempt} failed to send email: {e}")
            if attempt < max_retries:
                delay = _backoff_delay(attempt)
                logger.warning(f"Retrying email in {delay:.2f}s...")
                time.sleep(delay)
            else:
                logger.error("All retry attempts exhausted. Email not sent.")
    return False
