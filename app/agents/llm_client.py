"""Module defining LLM clients for agent prompting, starting with Google Gemini."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.agents.prompts import PromptMessage, format_prompt_messages
from app.core.env_loader import load_environment

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class LLMClientError(RuntimeError):
    """Base error raised by agent LLM clients."""


class LLMConfigurationError(LLMClientError):
    """Raised when the LLM client is not configured correctly."""


@dataclass(frozen=True, slots=True)
class GeminiLLMConfig:
    """Runtime configuration for the Gemini prompt client."""

    api_key: str | None = None
    model: str = DEFAULT_GEMINI_MODEL
    enabled: bool = True

    @classmethod
    def from_env(cls) -> GeminiLLMConfig:
        """Create Gemini config from environment variables."""
        load_environment()
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        enabled = os.getenv("GEMINI_LLM_ENABLED", "true").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        return cls(api_key=api_key, model=model, enabled=enabled)


class GeminiLLMClient:
    """Small adapter around the Google Gemini SDK for agent prompts."""

    def __init__(
        self,
        config: GeminiLLMConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config or GeminiLLMConfig.from_env()
        self._client = client

    @property
    def is_configured(self) -> bool:
        """Return whether Gemini prompting has enough config to run."""
        return self.config.enabled and bool(self.config.api_key)

    def generate_text(
        self,
        prompt: str | Sequence[PromptMessage],
        *,
        model: str | None = None,
    ) -> str:
        """Generate text from a raw prompt or provider-neutral prompt messages."""
        if not self.config.enabled:
            raise LLMConfigurationError("Gemini LLM client is disabled.")
        if not self.config.api_key:
            raise LLMConfigurationError(
                "Missing GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )

        response = self._get_client().models.generate_content(
            model=model or self.config.model,
            contents=_prompt_to_contents(prompt),
        )
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise LLMClientError("Gemini returned an empty response.")
        return text.strip()

    def try_generate_text(
        self,
        prompt: str | Sequence[PromptMessage],
        *,
        model: str | None = None,
    ) -> str | None:
        """Generate text when configured, otherwise return None for safe fallback."""
        if not self.is_configured:
            return None
        return self.generate_text(prompt, model=model)

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = _build_google_genai_client(self.config.api_key)
        return self._client


def _build_google_genai_client(api_key: str | None) -> Any:
    try:
        import google.genai as genai
    except ImportError as exc:
        raise LLMConfigurationError(
            "The google-genai package is required for Gemini prompting. "
            "Install it with: uv pip install google-genai"
        ) from exc

    return genai.Client(api_key=api_key)


def _prompt_to_contents(prompt: str | Sequence[PromptMessage]) -> str:
    if isinstance(prompt, str):
        return prompt
    return format_prompt_messages(tuple(prompt))


__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "GeminiLLMClient",
    "GeminiLLMConfig",
    "LLMClientError",
    "LLMConfigurationError",
]
