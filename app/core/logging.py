from __future__ import annotations

import logging as std_logging
import os
from logging.config import dictConfig

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_ACCESS_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _normalize_log_level(level: str | int | None) -> str | int:
    """Return a safe logging level value."""
    if isinstance(level, int):
        return level

    candidate = (level or os.getenv("LOG_LEVEL") or DEFAULT_LOG_LEVEL).upper()
    if candidate in std_logging.getLevelNamesMapping():
        return candidate

    return DEFAULT_LOG_LEVEL


def setup_logging(level: str | int | None = None) -> None:
    """Configure application logging.

    This keeps the app's logs consistent while allowing Uvicorn's own loggers
    to continue working when the API is served with `uvicorn app.main:app`.
    """
    log_level = _normalize_log_level(level)

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": os.getenv("LOG_FORMAT", DEFAULT_LOG_FORMAT),
                    "datefmt": os.getenv("LOG_DATE_FORMAT", DEFAULT_DATE_FORMAT),
                },
                "access": {
                    "format": os.getenv("ACCESS_LOG_FORMAT", DEFAULT_ACCESS_LOG_FORMAT),
                    "datefmt": os.getenv("LOG_DATE_FORMAT", DEFAULT_DATE_FORMAT),
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
                "access_console": {
                    "class": "logging.StreamHandler",
                    "formatter": "access",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "handlers": ["console"],
                "level": log_level,
            },
            "loggers": {
                "app": {
                    "handlers": ["console"],
                    "level": log_level,
                    "propagate": False,
                },
                "uvicorn": {
                    "handlers": ["console"],
                    "level": log_level,
                    "propagate": False,
                },
                "uvicorn.error": {
                    "level": log_level,
                },
                "uvicorn.access": {
                    "handlers": ["access_console"],
                    "level": log_level,
                    "propagate": False,
                },
            },
        }
    )


def get_logger(name: str) -> std_logging.Logger:
    """Return a named logger using the shared application configuration."""
    return std_logging.getLogger(name)
