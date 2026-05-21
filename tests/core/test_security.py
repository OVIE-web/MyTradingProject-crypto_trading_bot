"""Tests for the app's security utilities, including password hashing and JWT handling."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, status
from jose import jwt

from app.core import security


@pytest.fixture
def test_settings(monkeypatch) -> SimpleNamespace:
    """Patch security settings without mutating the frozen app settings object."""
    settings = SimpleNamespace(
        JWT_SECRET_KEY="test-secret-key-that-is-long-enough",
        ACCESS_TOKEN_EXPIRE_MINUTES=30,
        ADMIN_USERNAME="admin",
        ADMIN_PASSWORD="correct-password",
    )
    monkeypatch.setattr(security, "settings", settings)
    return settings


def test_verify_password_uses_constant_time_comparison() -> None:
    """Password verification should only pass for exact matches."""
    assert security.verify_password("secret", "secret") is True
    assert security.verify_password("secret", "different") is False


def test_authenticate_user_returns_user_for_valid_credentials(test_settings) -> None:
    """Known admin credentials should authenticate successfully."""
    user = security.authenticate_user(
        test_settings.ADMIN_USERNAME,
        test_settings.ADMIN_PASSWORD,
    )

    assert user == {"username": test_settings.ADMIN_USERNAME}


def test_authenticate_user_returns_none_for_invalid_credentials(test_settings) -> None:
    """Invalid usernames or passwords should not authenticate."""
    assert security.authenticate_user("wrong-user", test_settings.ADMIN_PASSWORD) is None
    assert security.authenticate_user(test_settings.ADMIN_USERNAME, "wrong-password") is None


def test_create_access_token_encodes_subject_and_expiration(test_settings) -> None:
    """Created JWTs should include the caller payload and an expiry."""
    token = security.create_access_token(
        {"sub": test_settings.ADMIN_USERNAME},
        expires_delta=timedelta(minutes=5),
    )

    payload = jwt.decode(
        token,
        test_settings.JWT_SECRET_KEY,
        algorithms=[security.ALGORITHM],
    )

    assert payload["sub"] == test_settings.ADMIN_USERNAME
    assert "exp" in payload


def test_create_access_token_raises_http_exception_on_encoding_failure(
    monkeypatch,
    test_settings,
) -> None:
    """JWT creation failures should be converted into a 500 response."""

    def raise_encoding_error(*args, **kwargs) -> str:
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(security.jwt, "encode", raise_encoding_error)

    with pytest.raises(HTTPException) as exc_info:
        security.create_access_token({"sub": test_settings.ADMIN_USERNAME})

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Could not generate authentication token"


@pytest.mark.asyncio
async def test_get_current_user_returns_username_for_valid_token(test_settings) -> None:
    """A valid token for the configured admin should resolve the current user."""
    token = security.create_access_token({"sub": test_settings.ADMIN_USERNAME})

    username = await security.get_current_user(token)

    assert username == test_settings.ADMIN_USERNAME


@pytest.mark.asyncio
async def test_get_current_user_rejects_invalid_token(test_settings) -> None:
    """Malformed tokens should fail with a bearer auth error."""
    with pytest.raises(HTTPException) as exc_info:
        await security.get_current_user("not-a-valid-token")

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


@pytest.mark.asyncio
async def test_get_current_user_rejects_unknown_username(test_settings) -> None:
    """Valid JWTs for a different subject should still be rejected."""
    token = security.create_access_token({"sub": "someone-else"})

    with pytest.raises(HTTPException) as exc_info:
        await security.get_current_user(token)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Could not validate credentials"
