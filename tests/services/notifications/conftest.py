from __future__ import annotations

from collections.abc import Iterator

import pytest
from pytest import MonkeyPatch


@pytest.fixture
def mock_env(monkeypatch: MonkeyPatch) -> Iterator[None]:
    """Provide notifier environment variables without touching real credentials."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "mock-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "mock-chat-id")
    monkeypatch.setenv("SMTP_HOST", "smtp.test.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "sender@test.com")
    monkeypatch.setenv("SMTP_PASS", "password")
    monkeypatch.setenv("EMAIL_TO", "receiver@test.com")
    yield
