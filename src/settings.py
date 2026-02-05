# src/settings.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_env_int(name: str, default: int | None = None) -> int:
    value = os.getenv(name)
    if value is None:
        if default is not None:
            return default
        raise RuntimeError(f"Missing required environment variable: {name}")
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"Environment variable {name} must be a valid integer, got: {value}"
        ) from exc


@dataclass(frozen=True, slots=True)
class Settings:
    # ------------------------
    # Environment
    # ------------------------
    ENV: Final[str] = os.getenv("ENV", "development")

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
    # Admin user
    # ------------------------
    ADMIN_USERNAME: Final[str] = _get_env("ADMIN_USERNAME")
    ADMIN_PASSWORD: Final[str] = _get_env("ADMIN_PASSWORD")


settings = Settings()


def _validate_settings(settings: Settings) -> None:
    """
    Fail fast on insecure configuration in non-development environments.
    """
    if settings.ENV != "development":
        if settings.ADMIN_USERNAME == "admin" and settings.ADMIN_PASSWORD == "admin":
            raise RuntimeError("Default admin credentials detected in non-development environment.")

        if "testuser:testpass" in settings.DATABASE_URL:
            raise RuntimeError("Test database credentials detected in non-development environment.")

        if len(settings.JWT_SECRET_KEY) < 32:
            raise RuntimeError(
                "JWT_SECRET_KEY is too short. Use a strong, random secret (>=32 chars)."
            )


_validate_settings(settings)
