# app/core/env_loader.py
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_ENV_LOADED = False


def load_environment() -> None:
    """
    Load environment variables once.

    Loading order:
    1. .env.local first
    2. .env second as fallback

    Because load_dotenv does not override existing values by default,
    values in .env.local will take priority over .env.
    """
    global _ENV_LOADED

    if _ENV_LOADED:
        return

    project_root = Path(__file__).resolve().parents[2]

    env_local_path = project_root / ".env.local"
    env_path = project_root / ".env"

    load_dotenv(env_local_path)
    load_dotenv(env_path)

    _ENV_LOADED = True
