from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.workers.scheduler as scheduler


def test_scheduler_settings_from_env_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOT_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("BOT_RUN_ONCE", raising=False)

    settings = scheduler.SchedulerSettings.from_env()

    assert settings.interval_seconds == scheduler.DEFAULT_INTERVAL
    assert settings.run_once is False


def test_scheduler_settings_from_env_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_INTERVAL_SECONDS", "45")
    monkeypatch.setenv("BOT_RUN_ONCE", "1")

    settings = scheduler.SchedulerSettings.from_env()

    assert settings.interval_seconds == 45
    assert settings.run_once is True


@pytest.mark.parametrize("value", ["0", "-1"])
def test_scheduler_settings_rejects_non_positive_interval(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("BOT_INTERVAL_SECONDS", value)

    with pytest.raises(ValueError, match="BOT_INTERVAL_SECONDS must be greater than 0"):
        scheduler.SchedulerSettings.from_env()


def test_scheduler_settings_rejects_non_integer_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOT_INTERVAL_SECONDS", "soon")

    with pytest.raises(ValueError, match="BOT_INTERVAL_SECONDS must be an integer"):
        scheduler.SchedulerSettings.from_env()


@pytest.mark.asyncio
async def test_run_scheduler_run_once_executes_single_iteration() -> None:
    settings = scheduler.SchedulerSettings(interval_seconds=5, run_once=True)

    with (
        patch("app.workers.scheduler.run_once_test", new_callable=AsyncMock) as run_once,
        patch("app.workers.scheduler.runner_loop", new_callable=AsyncMock) as runner_loop,
    ):
        await scheduler.run_scheduler(settings)

    run_once.assert_awaited_once_with()
    runner_loop.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_scheduler_delegates_to_runner_loop() -> None:
    settings = scheduler.SchedulerSettings(interval_seconds=5, run_once=False)

    with (
        patch("app.workers.scheduler.run_once_test", new_callable=AsyncMock) as run_once,
        patch("app.workers.scheduler.runner_loop", new_callable=AsyncMock) as runner_loop,
    ):
        await scheduler.run_scheduler(settings)

    run_once.assert_not_awaited()
    runner_loop.assert_awaited_once_with(run_once=False, interval_seconds=5)


def test_run_scheduler_sync_uses_asyncio_run() -> None:
    settings = scheduler.SchedulerSettings(interval_seconds=5, run_once=True)

    with (
        patch(
            "app.workers.scheduler.run_scheduler",
            new=MagicMock(return_value="scheduler-coro"),
        ) as run_scheduler,
        patch("app.workers.scheduler.asyncio.run") as asyncio_run,
    ):
        scheduler.run_scheduler_sync(settings)

    run_scheduler.assert_called_once_with(settings)
    asyncio_run.assert_called_once_with("scheduler-coro")
