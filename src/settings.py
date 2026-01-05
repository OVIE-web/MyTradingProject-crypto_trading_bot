# src/settings.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, Optional

from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: Optional[str] = None) -> str:
    """
    Fetch environment variable with optional default.
    Raises RuntimeError if missing and no default provided.
    """
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return str(value)


@dataclass(frozen=True, slots=True)
class Settings:
    # ------------------------
    # Database
    # ------------------------
    DATABASE_URL: Final[str] = _get_env(
        "DATABASE_URL",
        "postgresql://testuser:testpass@db:5432/tradingbot_test",
    )

    # ------------------------
    # Auth / JWT
    # ------------------------
    JWT_SECRET_KEY: Final[str] = _get_env("JWT_SECRET_KEY", "dev-secret-key")
    JWT_ALGORITHM: Final[str] = _get_env("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: Final[int] = int(_get_env("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # ------------------------
    # Admin user (optional but explicit)
    # ------------------------
    ADMIN_USERNAME: Final[str] = _get_env("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: Final[str] = _get_env("ADMIN_PASSWORD", "admin")


# Singleton-style settings object
settings = Settings()
