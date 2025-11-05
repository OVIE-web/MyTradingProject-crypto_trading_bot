# src/notifier.py
import os
import asyncio
import logging
import random
import smtplib
import time
from email.mime.text import MIMEText
from typing import Optional

import backoff
from telegram import Bot, constants  # python-telegram-bot >= 20

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Config defaults
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
EMAIL_HOST = os.getenv("SMTP_HOST") or os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("SMTP_PORT") or os.getenv("EMAIL_PORT") or 587)
EMAIL_USER = os.getenv("SMTP_USER") or os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("SMTP_PASS") or os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("SMTP_TO") or os.getenv("EMAIL_TO")

# small helper backoff jitter
def _backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    exp = min(cap, base * (2 ** (attempt - 1)))
    return exp / 2 + random.uniform(0, exp / 2)


class TelegramNotifier:
    """Async Telegram notifier with retry/backoff.
       Uses python-telegram-bot Bot which supports async send_message calls (v20+).
    """

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None, max_retries: int = 3):
        self.token = token or TELEGRAM_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)
        self.max_retries = max_retries
        self.bot: Optional[Bot] = None

        if self.enabled:
            # Do not log the token value — just confirm initialization
            self.bot = Bot(token=self.token, base_url=None)
            logger.info("Telegram Notifier initialized (chat_id=%s).", str(self.chat_id))
        else:
            logger.warning("Telegram Notifier disabled: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing.")

    async def send_message(self, message: str) -> bool:
        """Send a Telegram message asynchronously. Returns True on success."""
        if not self.enabled or not self.bot:
            logger.debug("Telegram notifier disabled or bot not initialized; skipping send_message.")
            return False

        for attempt in range(1, self.max_retries + 1):
            try:
                # Use parse_mode or disable_web_page_preview as needed
                await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode=constants.ParseMode.HTML)
                logger.info("Telegram message sent.")
                return True
            except Exception as exc:
                logger.error("Attempt %d failed to send Telegram message: %s", attempt, str(exc))
                if attempt < self.max_retries:
                    delay = _backoff_delay(attempt)
                    logger.debug("Retrying Telegram in %.2fs...", delay)
                    await asyncio.sleep(delay)
                else:
                    logger.error("All retry attempts exhausted for Telegram message.")
                    return False


def send_email_notification(subject: str, message: str, max_retries: int = 3) -> bool:
    """Synchronous email send with retry/backoff. Returns True on success."""
    if not (EMAIL_HOST and EMAIL_USER and EMAIL_PASS and EMAIL_TO):
        logger.error("Missing email configuration; cannot send email.")
        return False

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO

    for attempt in range(1, max_retries + 1):
        try:
            with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT, timeout=15) as smtp:
                smtp.starttls()
                smtp.login(EMAIL_USER, EMAIL_PASS)
                smtp.sendmail(EMAIL_USER, [EMAIL_TO], msg.as_string())
            logger.info("✅ Email sent successfully: %s", subject)
            return True
        except Exception as exc:
            logger.error("Attempt %d failed to send email: %s", attempt, str(exc))
            if attempt < max_retries:
                delay = _backoff_delay(attempt)
                logger.warning("Retrying email in %.2f seconds...", delay)
                time.sleep(delay)
            else:
                logger.error("All retry attempts exhausted. Email not sent.")
                return False
