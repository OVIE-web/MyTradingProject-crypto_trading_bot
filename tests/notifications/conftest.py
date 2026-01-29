# File: tests/notifications/conftest.py
from typing import Any, Generator

import pytest
from pytest import MonkeyPatch


@pytest.fixture
def mock_env(monkeypatch: MonkeyPatch) -> Generator[None, Any, None]:
    """Provides consistent environment variables for all notification tests."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "mock_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "mock_chat_id")
    monkeypatch.setenv("EMAIL_HOST", "smtp.test.com")
    monkeypatch.setenv("EMAIL_PORT", "587")
    monkeypatch.setenv("EMAIL_USER", "sender@test.com")
    monkeypatch.setenv("EMAIL_PASS", "password")
    monkeypatch.setenv("EMAIL_TO", "receiver@test.com")
    yield
