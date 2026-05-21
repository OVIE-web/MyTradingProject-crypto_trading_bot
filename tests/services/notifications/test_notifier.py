from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import MonkeyPatch

import app.services.notifications as notifications
from app.services.notifications.notifier import (
    TelegramNotifier,
    send_email_notification,
    send_telegram_notification,
)


def test_notifications_package_exports_notifier_api() -> None:
    assert notifications.TelegramNotifier is TelegramNotifier
    assert notifications.send_email_notification is send_email_notification
    assert notifications.send_telegram_notification is send_telegram_notification
    assert notifications.__all__ == [
        "TelegramNotifier",
        "send_email_notification",
        "send_telegram_notification",
    ]


@pytest.mark.usefixtures("mock_env")
def test_telegram_notifier_initializes_when_configured() -> None:
    with patch("app.services.notifications.notifier.Bot", return_value=MagicMock()) as mock_bot:
        notifier = TelegramNotifier()

    assert notifier.enabled is True
    assert notifier.chat_id == "mock-chat-id"
    assert notifier.bot == mock_bot.return_value
    mock_bot.assert_called_once_with(token="mock-token")


def test_telegram_notifier_disables_when_config_missing(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    notifier = TelegramNotifier()
    result = asyncio.run(notifier.send_message("Hello"))

    assert notifier.enabled is False
    assert notifier.bot is None
    assert result is False


@pytest.mark.usefixtures("mock_env")
def test_telegram_notifier_send_message_success() -> None:
    with patch("app.services.notifications.notifier.Bot") as mock_bot:
        mock_instance = mock_bot.return_value
        mock_instance.send_message = AsyncMock(return_value=True)

        notifier = TelegramNotifier()
        result = asyncio.run(notifier.send_message("Trade executed"))

    assert result is True
    mock_instance.send_message.assert_awaited_once_with(
        chat_id="mock-chat-id",
        text="Trade executed",
    )


@pytest.mark.usefixtures("mock_env")
def test_telegram_notifier_retries_and_returns_false_on_failure() -> None:
    with (
        patch("app.services.notifications.notifier.Bot") as mock_bot,
        patch("app.services.notifications.notifier._backoff_delay", return_value=0.0),
        patch("app.services.notifications.notifier.asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        mock_instance = mock_bot.return_value
        mock_instance.send_message = AsyncMock(side_effect=RuntimeError("network down"))

        notifier = TelegramNotifier(max_retries=2)
        result = asyncio.run(notifier.send_message("Trade failed"))

    assert result is False
    assert mock_instance.send_message.await_count == 2
    sleep.assert_awaited_once_with(0.0)


@pytest.mark.usefixtures("mock_env")
def test_send_telegram_notification_success() -> None:
    response = MagicMock(status_code=200, text="ok")

    with patch("app.services.notifications.notifier.requests.post", return_value=response) as post:
        result = send_telegram_notification("Trade executed")

    assert result is True
    post.assert_called_once_with(
        "https://api.telegram.org/botmock-token/sendMessage",
        json={"chat_id": "mock-chat-id", "text": "Trade executed"},
        timeout=10,
    )


@pytest.mark.usefixtures("mock_env")
def test_send_telegram_notification_returns_false_on_http_error() -> None:
    response = MagicMock(status_code=500, text="bad gateway")

    with patch("app.services.notifications.notifier.requests.post", return_value=response):
        result = send_telegram_notification("Trade executed")

    assert result is False


def test_send_telegram_notification_returns_false_without_config(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    result = send_telegram_notification("Trade executed")

    assert result is False


@pytest.mark.usefixtures("mock_env")
def test_send_email_notification_success() -> None:
    with patch("app.services.notifications.notifier.smtplib.SMTP") as smtp:
        server = smtp.return_value.__enter__.return_value
        result = send_email_notification("Trade Alert", "A BUY order was filled")

    assert result is True
    smtp.assert_called_once_with("smtp.test.com", 587)
    server.starttls.assert_called_once_with()
    server.login.assert_called_once_with("sender@test.com", "password")
    server.sendmail.assert_called_once()

    sender, recipient, message = server.sendmail.call_args.args
    assert sender == "sender@test.com"
    assert recipient == "receiver@test.com"
    assert "Subject: Trade Alert" in message
    assert "A BUY order was filled" in message


def test_send_email_notification_returns_false_without_config(monkeypatch: MonkeyPatch) -> None:
    for key in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS", "EMAIL_TO"):
        monkeypatch.delenv(key, raising=False)

    result = send_email_notification("Trade Alert", "Missing config")

    assert result is False


@pytest.mark.usefixtures("mock_env")
def test_send_email_notification_retries_and_returns_false_on_failure() -> None:
    with (
        patch("app.services.notifications.notifier.smtplib.SMTP", side_effect=OSError("smtp down")),
        patch("app.services.notifications.notifier._backoff_delay", return_value=0.0),
        patch("app.services.notifications.notifier.time.sleep") as sleep,
    ):
        result = send_email_notification("Trade Alert", "SMTP is down", max_retries=2)

    assert result is False
    sleep.assert_called_once_with(0.0)
