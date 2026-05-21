"""Tests for the app's logging configuration and utilities."""

from __future__ import annotations

import logging as std_logging

from app.core import logging as app_logging


def test_normalize_log_level_accepts_integer_level() -> None:
    """Integer logging levels should pass through unchanged."""
    assert app_logging._normalize_log_level(std_logging.DEBUG) == std_logging.DEBUG


def test_normalize_log_level_uses_env_value(monkeypatch) -> None:
    """The LOG_LEVEL environment variable should be honored when valid."""
    monkeypatch.setenv("LOG_LEVEL", "debug")

    assert app_logging._normalize_log_level(None) == "DEBUG"


def test_normalize_log_level_falls_back_for_invalid_value(monkeypatch) -> None:
    """Invalid levels should fall back to the safe default."""
    monkeypatch.setenv("LOG_LEVEL", "not-a-real-level")

    assert app_logging._normalize_log_level(None) == app_logging.DEFAULT_LOG_LEVEL


def test_setup_logging_builds_expected_logger_config(mocker, monkeypatch) -> None:
    """setup_logging should configure app and Uvicorn loggers safely."""
    monkeypatch.setenv("LOG_FORMAT", "%(levelname)s:%(name)s:%(message)s")
    monkeypatch.setenv("ACCESS_LOG_FORMAT", "%(message)s")
    mock_dict_config = mocker.patch("app.core.logging.dictConfig")

    app_logging.setup_logging("WARNING")

    config = mock_dict_config.call_args.args[0]

    assert config["disable_existing_loggers"] is False
    assert config["root"]["level"] == "WARNING"
    assert config["formatters"]["default"]["format"] == "%(levelname)s:%(name)s:%(message)s"
    assert config["formatters"]["access"]["format"] == "%(message)s"
    assert config["loggers"]["app"]["propagate"] is False
    assert config["loggers"]["uvicorn.access"]["handlers"] == ["access_console"]
    assert config["loggers"]["uvicorn.access"]["propagate"] is False


def test_get_logger_returns_named_standard_logger() -> None:
    """get_logger should return a standard library logger by name."""
    logger = app_logging.get_logger("app.core.tests")

    assert isinstance(logger, std_logging.Logger)
    assert logger.name == "app.core.tests"
