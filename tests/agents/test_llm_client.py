from __future__ import annotations

import pytest

from app.agents import (
    GeminiLLMClient,
    GeminiLLMConfig,
    LLMConfigurationError,
    PromptMessage,
)


class FakeGeminiResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class FakeGeminiModels:
    def __init__(self, response_text: str | None = "Generated rationale") -> None:
        self.response_text = response_text
        self.last_model: str | None = None
        self.last_contents: str | None = None

    def generate_content(self, *, model: str, contents: str) -> FakeGeminiResponse:
        self.last_model = model
        self.last_contents = contents
        return FakeGeminiResponse(self.response_text)


class FakeGeminiClient:
    def __init__(self, response_text: str | None = "Generated rationale") -> None:
        self.models = FakeGeminiModels(response_text=response_text)


def test_gemini_config_reads_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setenv("GEMINI_LLM_ENABLED", "true")

    config = GeminiLLMConfig.from_env()

    assert config.api_key == "test-key"
    assert config.model == "gemini-test-model"
    assert config.enabled is True


def test_gemini_config_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_LLM_ENABLED", "false")

    config = GeminiLLMConfig.from_env()

    assert config.enabled is False


def test_generate_text_uses_configured_model_and_prompt_messages() -> None:
    fake_client = FakeGeminiClient(response_text=" Risk-aware summary ")
    client = GeminiLLMClient(
        GeminiLLMConfig(api_key="test-key", model="gemini-test-model"),
        client=fake_client,
    )

    result = client.generate_text(
        (
            PromptMessage("system", "Stay safe."),
            PromptMessage("user", "Explain BTCUSDT signal."),
        )
    )

    assert result == "Risk-aware summary"
    assert fake_client.models.last_model == "gemini-test-model"
    assert fake_client.models.last_contents is not None
    assert "SYSTEM:" in fake_client.models.last_contents
    assert "Explain BTCUSDT signal." in fake_client.models.last_contents


def test_try_generate_text_returns_none_when_unconfigured() -> None:
    client = GeminiLLMClient(GeminiLLMConfig(api_key=None))

    assert client.try_generate_text("hello") is None


def test_generate_text_requires_api_key() -> None:
    client = GeminiLLMClient(GeminiLLMConfig(api_key=None))

    with pytest.raises(LLMConfigurationError, match="Missing GOOGLE_API_KEY"):
        client.generate_text("hello")


def test_generate_text_rejects_empty_model_response() -> None:
    client = GeminiLLMClient(
        GeminiLLMConfig(api_key="test-key"),
        client=FakeGeminiClient(response_text=" "),
    )

    with pytest.raises(RuntimeError, match="empty response"):
        client.generate_text("hello")
