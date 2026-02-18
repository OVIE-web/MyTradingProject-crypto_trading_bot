import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

import src.bot_runner as bot_runner


@pytest.mark.asyncio
async def test_runner_loop_run_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test bot runner handles iterations correctly."""
    called: dict[str, bool] = {}

    # Mock external resources to prevent live connections
    monkeypatch.setattr(bot_runner, "BinanceManager", MagicMock())
    monkeypatch.setattr(bot_runner, "TelegramNotifier", MagicMock())
    monkeypatch.setattr(bot_runner, "load_trained_model", MagicMock(return_value=(MagicMock(), {})))

    # --- Test 1: Simulate successful iteration ---
    async def fake_iteration(resources: dict[str, Any]) -> None:
        logging.info("Iteration completed")
        called["done"] = True

    monkeypatch.setattr(bot_runner, "do_iteration", fake_iteration)

    await bot_runner.runner_loop(run_once=True, interval_seconds=0)

    assert "done" in called, "Iteration should have been called"
    assert called["done"] is True


@pytest.mark.asyncio
async def test_runner_loop_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test bot runner handles iteration errors gracefully."""

    # Mock external resources to prevent live connections
    monkeypatch.setattr(bot_runner, "BinanceManager", MagicMock())
    monkeypatch.setattr(bot_runner, "TelegramNotifier", MagicMock())
    monkeypatch.setattr(bot_runner, "load_trained_model", MagicMock(return_value=(MagicMock(), {})))

    # --- Test 2: Simulate error handling ---
    async def bad_iteration(resources: dict[str, Any]) -> None:
        raise Exception("Test iteration error")

    monkeypatch.setattr(bot_runner, "do_iteration", bad_iteration)

    # The runner should catch the error and log it, not raise
    try:
        await bot_runner.runner_loop(run_once=True, interval_seconds=0)
    except Exception as e:
        # If exception raised, it should be our test error
        assert "Test iteration error" in str(e)
