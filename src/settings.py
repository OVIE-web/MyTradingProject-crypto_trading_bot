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
    return value


def _get_env_int(name: str, default: Optional[int] = None) -> int:
    """
    Fetch environment variable as integer with default.
    Raises RuntimeError if missing or not an integer.
    """
    value = os.getenv(name)
    if value is None:
        if default is not None:
            return default
        raise RuntimeError(f"Missing required environment variable: {name}")
    try:
        return int(value)
    except ValueError:
        raise RuntimeError(f"Environment variable {name} must be a valid integer, got: {value}")


@dataclass(frozen=True, slots=True)
class Settings:
    # ------------------------
    # Database
    # ------------------------
    DATABASE_URL: Final[str] = _get_env("DATABASE_URL")

    # ------------------------
    # Auth / JWT
    # ------------------------
    JWT_SECRET_KEY: Final[str] = _get_env("JWT_SECRET_KEY")
    JWT_ALGORITHM: Final[str] = _get_env("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: Final[int] = _get_env_int("ACCESS_TOKEN_EXPIRE_MINUTES", 30)

    # ------------------------
    # Admin user (no defaults)
    # ------------------------
    ADMIN_USERNAME: Final[str] = _get_env("ADMIN_USERNAME")
    ADMIN_PASSWORD: Final[str] = _get_env("ADMIN_PASSWORD")


# Singleton-style settings object
settings = Settings()


# --- Startup validation for weak/test credentials ---
def _validate_settings(settings: Settings) -> None:
    if settings.ADMIN_USERNAME == "admin" and settings.ADMIN_PASSWORD == "admin":
        raise RuntimeError(
            "Default admin credentials detected! Set strong ADMIN_USERNAME and ADMIN_PASSWORD."
        )
    if "testuser:testpass" in settings.DATABASE_URL:
        raise RuntimeError("Default/test database credentials detected! Set a secure DATABASE_URL.")


_validate_settings(settings)
