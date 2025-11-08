# tests/test_bot_runner.py
import asyncio
import pytest
from typing import List, Any
import src.bot_runner as bot_runner

@pytest.fixture
def monkeypatch(request):
    return request.getfuncargvalue("monkeypatch")
@pytest.mark.asyncio
async def test_runner_loop_run_once(monkeypatch):
    called = {}

    async def fake_iteration(resources):
        called["done"] = True

    monkeypatch.setattr(bot_runner, "do_iteration", fake_iteration)

    await bot_runner.runner_loop(run_once=True, interval_seconds=0)

    assert "done" in called