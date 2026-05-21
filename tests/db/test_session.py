"""Tests for the database session management in app.db.session."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.db.session import get_db


def test_get_db_yields_session_and_closes_after_completion(mocker) -> None:
    """get_db should yield one session and close it when exhausted."""
    mock_session = mocker.MagicMock(spec=Session)
    mock_session_local = mocker.patch("app.db.session.SessionLocal", return_value=mock_session)

    generator = get_db()

    assert next(generator) is mock_session
    with pytest.raises(StopIteration):
        next(generator)

    mock_session_local.assert_called_once_with()
    mock_session.close.assert_called_once_with()


def test_get_db_closes_session_when_generator_is_closed(mocker) -> None:
    """FastAPI cleanup should close the DB session even when the caller stops early."""
    mock_session = mocker.MagicMock(spec=Session)
    mocker.patch("app.db.session.SessionLocal", return_value=mock_session)

    generator = get_db()
    assert next(generator) is mock_session

    generator.close()

    mock_session.close.assert_called_once_with()
