from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.schemas.auth_schema import (
    CurrentUserResponse,
    LoginRequest,
    TokenData,
    TokenResponse,
    UserBase,
    UserCreate,
    UserRead,
)


def test_token_response_defaults_to_bearer_token_type() -> None:
    token = TokenResponse(access_token="access-token")

    assert token.access_token == "access-token"
    assert token.token_type == "bearer"


def test_token_data_allows_missing_username() -> None:
    token_data = TokenData()

    assert token_data.username is None


def test_login_request_requires_username_and_password() -> None:
    login = LoginRequest(username="ovie", password="secret")

    assert login.username == "ovie"
    assert login.password == "secret"

    with pytest.raises(ValidationError):
        LoginRequest(username="", password="secret")

    with pytest.raises(ValidationError):
        LoginRequest(username="ovie", password="")


def test_current_user_response_exposes_username() -> None:
    response = CurrentUserResponse(username="ovie")

    assert response.username == "ovie"


def test_user_base_normalizes_username_and_sets_defaults() -> None:
    user = UserBase(username="  trader  ", email="trader@example.com")

    assert user.username == "trader"
    assert user.email == "trader@example.com"
    assert user.role == "trader"
    assert user.is_active is True
    assert user.is_superuser is False


def test_user_base_rejects_blank_username() -> None:
    with pytest.raises(ValidationError):
        UserBase(username="   ")


def test_user_base_rejects_invalid_email() -> None:
    with pytest.raises(ValidationError):
        UserBase(username="trader", email="not-an-email")


def test_user_create_requires_stronger_password_length() -> None:
    created = UserCreate(username="trader", password="safe-pass")

    assert created.username == "trader"
    assert created.password == "safe-pass"

    with pytest.raises(ValidationError):
        UserCreate(username="trader", password="short")


def test_user_read_validates_from_orm_like_object() -> None:
    now = datetime(2026, 5, 16, tzinfo=UTC)
    db_user = SimpleNamespace(
        id=1,
        username="  ovie  ",
        email="ovie@example.com",
        role="admin",
        is_active=True,
        is_superuser=False,
        last_login_at=None,
        created_at=now,
        updated_at=now,
    )

    user = UserRead.model_validate(db_user)

    assert user.id == 1
    assert user.username == "ovie"
    assert user.email == "ovie@example.com"
    assert user.created_at == now
    assert user.updated_at == now
