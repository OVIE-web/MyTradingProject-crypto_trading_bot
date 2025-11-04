"""
notification.py
----------------
Lightweight, synchronous notification utilities for Telegram and Email.

This complements notifier.py by offering simpler, stateless
notification functions used in tests, pipelines, or quick alerts.
"""

import os
import smtplib
import logging
import requests
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


# =====================================================================================
# TELEGRAM NOTIFICATION
# =====================================================================================

def send_telegram_notification(message: str) -> None:
    """
    Send a Telegram message synchronously using Telegram Bot API.
    
    Environment Variables:
    - TELEGRAM_BOT_TOKEN
    - TELEGRAM_CHAT_ID
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logger.error("Missing Telegram configuration: BOT_TOKEN or CHAT_ID.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            logger.error(f"Telegram API error: {response.status_code} - {response.text}")
        else:
            logger.info(f"Telegram notification sent: '{message}'")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Telegram notification (HTTP request error): {e}")
    except Exception as e:
        logger.exception(f"Unexpected error sending Telegram message: {e}")


# =====================================================================================
# EMAIL NOTIFICATION
# =====================================================================================

def send_email_notification(subject: str, message: str) -> None:
    """
    Send an email notification via SMTP.

    Environment Variables:
    - EMAIL_HOST
    - EMAIL_PORT
    - EMAIL_USER
    - EMAIL_PASS
    - EMAIL_TO
    """
    host = os.getenv("EMAIL_HOST")
    port = int(os.getenv("EMAIL_PORT", "587"))
    user = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")
    to_email = os.getenv("EMAIL_TO")

    if not all([host, port, user, password, to_email]):
        logger.error("Missing email configuration in environment variables.")
        return

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
    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")
