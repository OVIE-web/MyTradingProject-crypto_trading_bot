"""Tests for the app's environment loading logic, ensuring correct file precedence and idempotency."""

from __future__ import annotations

from pathlib import Path

from app.core import env_loader


def test_load_environment_loads_local_then_default_env(mocker) -> None:
    """Environment files should load once, with .env.local taking priority."""
    mock_load_dotenv = mocker.patch("app.core.env_loader.load_dotenv")
    env_loader._ENV_LOADED = False

    env_loader.load_environment()
    env_loader.load_environment()

    called_paths = [call.args[0] for call in mock_load_dotenv.call_args_list]

    assert len(called_paths) == 2
    assert all(isinstance(path, Path) for path in called_paths)
    assert [path.name for path in called_paths] == [".env.local", ".env"]
    assert env_loader._ENV_LOADED is True


def test_load_environment_skips_when_already_loaded(mocker) -> None:
    """Repeated setup should be a no-op after the first successful load."""
    mock_load_dotenv = mocker.patch("app.core.env_loader.load_dotenv")
    env_loader._ENV_LOADED = True

    env_loader.load_environment()

    mock_load_dotenv.assert_not_called()
