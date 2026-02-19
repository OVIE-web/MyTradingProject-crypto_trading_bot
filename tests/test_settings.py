"""
Unit tests for application settings and environment variable validation.
...
"""

import importlib
from collections.abc import Generator
from types import ModuleType
from unittest.mock import patch

import pytest

import src.settings


@pytest.fixture(autouse=True)
def mock_load_dotenv() -> Generator[None, None, None]:
    """
    Mock dotenv.load_dotenv to prevent it from loading .env files during tests.

    This ensures that tests run in a completely isolated environment without
    external .env file interference.

    Yields:
        None: Context manager that patches dotenv.load_dotenv
    """
    with patch("dotenv.load_dotenv"):
        yield


def reload_settings() -> ModuleType:
    """Reload the settings module to reflect new environment variables."""
    importlib.reload(src.settings)
    return src.settings  # type: ignore[return-value]


@pytest.mark.unit
def test_get_env_missing_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that _get_env raises RuntimeError if a required env var is missing.

    Validates that the settings module properly enforces the presence of
    the DATABASE_URL environment variable, which is critical for application
    startup.

    Args:
        monkeypatch: pytest fixture for modifying environment variables
    Raises:
        RuntimeError: Expected when DATABASE_URL is not set
    """
    # Set other required environment variables to isolate the test
    monkeypatch.setenv("JWT_SECRET_KEY", "a_super_secret_key_that_is_long_enough")
    monkeypatch.setenv("ADMIN_USERNAME", "test_admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "test_password")

    # Temporarily remove the one we are testing
    monkeypatch.delenv("DATABASE_URL", raising=False)

    # Expect a RuntimeError when the settings module is reloaded
    with pytest.raises(RuntimeError, match="Missing required environment variable: DATABASE_URL"):
        reload_settings()


@pytest.mark.unit
def test_get_env_int_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _get_env_int returns the default value if the env var is not set"""
    monkeypatch.delenv("ACCESS_TOKEN_EXPIRE_MINUTES", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
    monkeypatch.setenv("JWT_SECRET_KEY", "a_super_secret_key_that_is_long_enough")
    monkeypatch.setenv("ADMIN_USERNAME", "test_admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "test_password")

    reloaded_settings = reload_settings()
    assert reloaded_settings.ACCESS_TOKEN_EXPIRE_MINUTES == 30  # type: ignore[attr-defined]


@pytest.mark.unit
def test_get_env_int_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that _get_env_int raises RuntimeError if the env var is not a valid integer.

    Validates that string values are properly rejected when an integer
    configuration value is expected.

    Args:
        monkeypatch: pytest fixture for modifying environment variables
    Raises:
        RuntimeError: Expected when ACCESS_TOKEN_EXPIRE_MINUTES is not a valid integer
    """
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "not-an-int")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
    monkeypatch.setenv("JWT_SECRET_KEY", "a_super_secret_key_that_is_long_enough")
    monkeypatch.setenv("ADMIN_USERNAME", "test_admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "test_password")

    with pytest.raises(RuntimeError, match="must be a valid integer"):
        reload_settings()


@pytest.mark.unit
def test_get_env_int_missing_no_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that _get_env_int raises RuntimeError if a required env var is missing and has no default.

    Validates that required integer environment variables without defaults
    properly raise an error when not provided.

    Args:
        monkeypatch: pytest fixture for modifying environment variables
    Raises:
        RuntimeError: Expected when A_MISSING_INT is not set
    """
    monkeypatch.delenv("A_MISSING_INT", raising=False)
    with pytest.raises(RuntimeError, match="Missing required environment variable: A_MISSING_INT"):
        src.settings._get_env_int("A_MISSING_INT")


@pytest.mark.unit
def test_validate_settings_default_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that a RuntimeError is raised for default admin credentials in non-dev environments.
    Validates that the application enforces security by preventing deployment
    with default admin credentials (username/password both "admin") in production
    or staging environments.

    Args:
        monkeypatch: pytest fixture for modifying environment variables

    Raises:
        RuntimeError: Expected when default admin credentials are detected in production
    """
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "admin")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
    monkeypatch.setenv("JWT_SECRET_KEY", "a_super_secret_key_that_is_long_enough")

    with pytest.raises(RuntimeError, match="Default admin credentials detected"):
        reload_settings()


@pytest.mark.unit
def test_validate_settings_test_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that a RuntimeError is raised for test DB credentials in non-dev environments.

    Validates that the application prevents accidental use of test database
    credentials in production environments, which would be a critical security
    and data integrity issue.

    Args:
        monkeypatch: pytest fixture for modifying environment variables

    Raises:
        RuntimeError: Expected when test database credentials are detected in production
    """
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql://testuser:testpass@host:5432/db")
    monkeypatch.setenv("JWT_SECRET_KEY", "a_super_secret_key_that_is_long_enough")
    monkeypatch.setenv("ADMIN_USERNAME", "test_admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "test_password")

    with pytest.raises(RuntimeError, match="Test database credentials detected"):
        reload_settings()


@pytest.mark.unit
def test_validate_settings_short_jwt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that a RuntimeError is raised for a short JWT secret key in non-dev environments.

    Validates that JWT secret keys meet minimum length requirements in
    production environments to ensure cryptographic security.

    Args:
        monkeypatch: pytest fixture for modifying environment variables

    Raises:
        RuntimeError: Expected when JWT_SECRET_KEY is too short for production
    """
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("JWT_SECRET_KEY", "short_key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
    monkeypatch.setenv("ADMIN_USERNAME", "test_admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "test_password")

    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY is too short"):
        reload_settings()
