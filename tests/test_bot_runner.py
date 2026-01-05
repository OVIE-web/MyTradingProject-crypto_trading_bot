import asyncio
import logging
import traceback
from typing import Dict, Literal

import pytest

import src.bot_runner as bot_runner


@pytest.fixture
def my_fixture():
    return "fixture value"


@pytest.mark.asyncio
async def test_my_test(request: pytest.FixtureRequest, my_fixture: Literal["fixture value"]) -> None:
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

    called: Dict[str, bool] = {}

    # --- 1) Simulate do_iteration raising an exception ---
    async def bad_iteration():
        raise Exception("Test exception")

    monkeypatch.setattr(bot_runner, "do_iteration", bad_iteration)

    with pytest.raises(Exception, match="Test exception"):
        await bot_runner.runner_loop(run_once=True, interval_seconds=0)

    # --- 2) Simulate a successful do_iteration call ---
    async def fake_iteration():
        logging.info("Iteration completed")
        called["done"] = True

    monkeypatch.setattr(bot_runner, "do_iteration", fake_iteration)

    await bot_runner.runner_loop(run_once=True, interval_seconds=0)

    assert "done" in called
    assert "done" in called
