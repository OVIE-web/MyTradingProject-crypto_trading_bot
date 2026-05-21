"""Tests for the app.models package, ensuring that all models are properly registered and that the package exports the expected public API."""

from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy.orm import Session

from app.models import Prediction, Trade, User


def test_model_metadata_registers_expected_tables(model_session: Session) -> None:
    """All app models should be registered in SQLAlchemy metadata."""
    table_names = set(inspect(model_session.get_bind()).get_table_names())

    assert {"trades", "predictions", "users"}.issubset(table_names)


def test_models_package_exports_public_models() -> None:
    """The models package should expose the canonical ORM models."""
    import app.models as models

    assert models.__all__ == ["Prediction", "Trade", "User"]
    assert models.Prediction is Prediction
    assert models.Trade is Trade
    assert models.User is User
