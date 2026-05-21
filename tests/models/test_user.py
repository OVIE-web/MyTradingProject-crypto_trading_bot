"""Unit tests for app.models.user."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.user import User


def test_user_model_applies_defaults_and_mark_login(model_session: Session) -> None:
    """User should provide safe defaults and update login timestamp."""
    user = User(username="trader", email="trader@example.com", password_hash="hashed")

    model_session.add(user)
    model_session.commit()
    model_session.refresh(user)

    assert user.id is not None
    assert user.role == "trader"
    assert user.is_active is True
    assert user.is_superuser is False
    assert user.created_at is not None
    assert user.updated_at is not None
    assert user.last_login_at is None

    user.mark_login()

    assert isinstance(user.last_login_at, datetime)


def test_user_unique_username_and_email(model_session: Session) -> None:
    """Usernames and email addresses should remain unique."""
    first = User(username="trader", email="trader@example.com", password_hash="hashed")
    second = User(username="trader", email="trader@example.com", password_hash="hashed")

    model_session.add_all([first, second])

    with pytest.raises(IntegrityError):
        model_session.commit()


def test_user_can_override_defaults() -> None:
    """Explicit user role and privilege flags should be respected."""
    user = User(
        username="admin",
        password_hash="hashed",
        role="admin",
        is_active=False,
        is_superuser=True,
    )

    assert user.role == "admin"
    assert user.is_active is False
    assert user.is_superuser is True
