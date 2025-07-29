# File: ./tests/test_notifications.py

import pytest
import smtplib
import requests
from email.mime.text import MIMEText
import os
import logging  # Ensure logging is imported

# --- Email Notification Tests ---

def test_send_email_notification_success(caplog, monkeypatch):
    # Mock the SMTP connection and sendmail method
    def mock_smtp(*args, **kwargs):
        smtp = smtplib.SMTP(*args, **kwargs)
        smtp.starttls = lambda: None
        smtp.login = lambda user, password: None
        smtp.sendmail = lambda sender, receiver, message: {}
        return smtp

    monkeypatch.setattr(smtplib, 'SMTP', mock_smtp)

    from main import send_email_notification  # Adjust this import as needed

    with caplog.at_level(logging.INFO):
        send_email_notification("Test Subject", "Test Message")
        assert "Email notification sent" in caplog.text

def test_send_email_notification_missing_config(caplog, monkeypatch):
    monkeypatch.delenv("EMAIL_USER", raising=False)
    monkeypatch.setenv("EMAIL_HOST", "somehost")

    from main import send_email_notification

    with caplog.at_level(logging.WARNING):
        send_email_notification("Subject", "Message")
        assert "Email notification details not fully configured" in caplog.text

# --- Telegram Notification Tests ---

def test_send_telegram_notification_success(caplog, monkeypatch):
    def mock_post(*args, **kwargs):
        response = requests.Response()
        response.raise_for_status = lambda: None
        return response

    monkeypatch.setattr(requests, 'post', mock_post)

    from main import send_telegram_notification

    with caplog.at_level(logging.INFO):
        send_telegram_notification("Hello from bot!")
        assert "Telegram notification sent" in caplog.text

def test_send_telegram_notification_http_error(caplog, monkeypatch):
    def mock_post(*args, **kwargs):
        response = requests.Response()
        response.raise_for_status = lambda: requests.exceptions.RequestException("HTTP Error")
        return response

    monkeypatch.setattr(requests, 'post', mock_post)

    from main import send_telegram_notification

    with caplog.at_level(logging.ERROR):
        send_telegram_notification("Error Test")
        assert "Failed to send Telegram notification (HTTP request error)" in caplog.text

def test_send_telegram_notification_missing_config(caplog, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    from main import send_telegram_notification

    with caplog.at_level(logging.WARNING):
        send_telegram_notification("Config Test")
        assert "Telegram Bot Token or Chat ID not found" in caplog.text