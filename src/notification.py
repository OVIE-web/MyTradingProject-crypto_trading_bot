# src/notification.py
import logging
import os
import smtplib
from email.mime.text import MIMEText

import requests

logger = logging.getLogger(__name__)


def send_telegram_notification(message: str) -> bool:
    """Sync fallback Telegram notification via requests."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.error("Missing Telegram configuration.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logger.error(f"Telegram API error: {response.status_code} - {response.text}")
            return False
        logger.info(f"Telegram notification sent: '{message}'")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")
        return False


def send_email_notification(subject: str, message: str) -> bool:
    """Sync fallback email notification."""
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", 587))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    to_email = os.getenv("EMAIL_TO")

    if not (host and user and password and to_email):
        logger.error("Missing email configuration.")
        return False

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_email

    try:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, to_email, msg.as_string())
        logger.info(f"✅ Email notification sent: {subject}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")
        return False
