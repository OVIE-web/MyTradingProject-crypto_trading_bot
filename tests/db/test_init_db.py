"""Tests for the app.db.init_db module."""

from __future__ import annotations

import importlib

import pytest

init_db_module = importlib.import_module("app.db.init_db")


def test_init_db_creates_registered_metadata_tables(mocker) -> None:
    """init_db should ask SQLAlchemy metadata to create all registered tables."""
    mock_create_all = mocker.patch.object(init_db_module.Base.metadata, "create_all")

    init_db_module.init_db()

    mock_create_all.assert_called_once_with(bind=init_db_module.engine)


def test_init_db_propagates_schema_creation_errors(mocker) -> None:
    """Schema creation errors should surface to the caller."""
    mocker.patch.object(
        init_db_module.Base.metadata,
        "create_all",
        side_effect=RuntimeError("schema failed"),
    )

    with pytest.raises(RuntimeError, match="schema failed"):
        init_db_module.init_db()


def test_package_init_db_exports_init_db_module_function() -> None:
    """The app.db package should expose the canonical init_db function."""
    import app.db as db_package

    assert db_package.init_db is init_db_module.init_db
