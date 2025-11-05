# src/notification.py
import os
import logging
import smtplib
import time
import requests
from email.mime.text import MIMEText
from typing import Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
EMAIL_HOST = os.getenv("SMTP_HOST") or os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("SMTP_PORT") or os.getenv("EMAIL_PORT") or 587)
EMAIL_USER = os.getenv("SMTP_USER") or os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("SMTP_PASS") or os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("SMTP_TO") or os.getenv("EMAIL_TO")


def send_telegram_notification(message: str) -> bool:
    """Synchronous Telegram send via requests (useful in scripts/tests)."""
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        logger.error("Missing Telegram configuration (token/chat_id).")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            logger.error("Telegram API error: %s - %s", r.status_code, getattr(r, "text", r))
            return False
        logger.info("Telegram notification sent: '%s'", message)
        return True
    except Exception as exc:
        logger.error("Failed to send Telegram notification (requests error): %s", str(exc))
        return False


def send_email_notification(subject: str, message: str) -> bool:
    """Synchronous email send (no retries here)."""
    if not (EMAIL_HOST and EMAIL_USER and EMAIL_PASS and EMAIL_TO):
        logger.error("Missing email configuration; cannot send email.")
        return False

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO
    try:
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT, timeout=15) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.sendmail(EMAIL_USER, [EMAIL_TO], msg.as_string())
        logger.info("✅ Email notification sent: %s", subject)
        return True
    except Exception as exc:
        logger.error("Failed to send email notification: %s", str(exc))
        return False
