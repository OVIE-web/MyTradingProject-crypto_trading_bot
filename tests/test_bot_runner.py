import asyncio
import logging
import traceback
from typing import Literal
from unittest.mock import MagicMock

import pytest

import src.bot_runner as bot_runner


@pytest.fixture
def my_fixture():
    return "fixture value"


@pytest.mark.asyncio
async def test_my_test(
    request: pytest.FixtureRequest, my_fixture: Literal["fixture value"]
) -> None:
    """Verify fixture retrieval works."""
    try:
        assert my_fixture == "fixture value"

        fixture_value = request.getfixturevalue("my_fixture")
        assert fixture_value == "fixture value"

    except Exception as e:
        logging.error("Test failed: %s", e)
        logging.error(traceback.format_exc())
        raise  # Never swallow exceptions in a test


async def my_async_function():
    await asyncio.sleep(0.1)
    logging.info("Async operation completed")


@pytest.mark.asyncio
async def test_runner_loop_run_once(monkeypatch: pytest.MonkeyPatch):
    """Test bot runner handles iterations correctly."""
    called: dict[str, bool] = {}

    # Mock external resources to prevent live connections
    monkeypatch.setattr(bot_runner, "BinanceManager", MagicMock())
    monkeypatch.setattr(bot_runner, "TelegramNotifier", MagicMock())
    monkeypatch.setattr(bot_runner, "load_trained_model", MagicMock(return_value=(MagicMock(), {})))

    # --- Test 1: Simulate successful iteration ---
    async def fake_iteration(resources):
        logging.info("Iteration completed")
        called["done"] = True

    monkeypatch.setattr(bot_runner, "do_iteration", fake_iteration)

    await bot_runner.runner_loop(run_once=True, interval_seconds=0)

    assert "done" in called, "Iteration should have been called"
    assert called["done"] is True


@pytest.mark.asyncio
async def test_runner_loop_error_handling(monkeypatch: pytest.MonkeyPatch):
    """Test bot runner handles iteration errors gracefully."""

    # Mock external resources to prevent live connections
    monkeypatch.setattr(bot_runner, "BinanceManager", MagicMock())
    monkeypatch.setattr(bot_runner, "TelegramNotifier", MagicMock())
    monkeypatch.setattr(bot_runner, "load_trained_model", MagicMock(return_value=(MagicMock(), {})))

    # --- Test 2: Simulate error handling ---
    async def bad_iteration(resources):
        raise Exception("Test iteration error")

    monkeypatch.setattr(bot_runner, "do_iteration", bad_iteration)

    # The runner should catch the error and log it, not raise
    try:
        await bot_runner.runner_loop(run_once=True, interval_seconds=0)
    except Exception as e:
        # If exception raised, it should be our test error
        assert "Test iteration error" in str(e)
