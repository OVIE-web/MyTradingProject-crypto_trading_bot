from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import app.workers.celery_worker as celery_worker


def test_create_celery_app_uses_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://redis:6379/2")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "redis://redis:6379/3")
    monkeypatch.setenv("CELERY_TIMEZONE", "Africa/Lagos")
    monkeypatch.setenv("BOT_INTERVAL_SECONDS", "30")

    app = celery_worker.create_celery_app()

    assert app.conf.broker_url == "redis://redis:6379/2"
    assert app.conf.result_backend == "redis://redis:6379/3"
    assert app.conf.timezone == "Africa/Lagos"
    assert app.conf.beat_schedule["run-trading-bot-iteration"]["schedule"] == 30


@pytest.mark.parametrize("value", ["0", "-5"])
def test_create_celery_app_rejects_non_positive_interval(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("BOT_INTERVAL_SECONDS", value)

    with pytest.raises(ValueError, match="BOT_INTERVAL_SECONDS must be greater than 0"):
        celery_worker.create_celery_app()


def test_create_celery_app_rejects_non_integer_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOT_INTERVAL_SECONDS", "fast")

    with pytest.raises(ValueError, match="BOT_INTERVAL_SECONDS must be an integer"):
        celery_worker.create_celery_app()


def test_run_trading_iteration_task_runs_single_iteration() -> None:
    with (
        patch(
            "app.workers.celery_worker.run_once_test",
            new=MagicMock(return_value="run-once-coro"),
        ) as run_once,
        patch("app.workers.celery_worker.asyncio.run", return_value=True) as asyncio_run,
    ):
        result = celery_worker.run_trading_iteration.run()

    assert result == {"ok": True}
    run_once.assert_called_once_with()
    asyncio_run.assert_called_once_with("run-once-coro")


def test_run_trading_loop_task_uses_explicit_interval() -> None:
    with (
        patch(
            "app.workers.celery_worker.runner_loop",
            new=MagicMock(return_value="loop-coro"),
        ) as runner_loop,
        patch("app.workers.celery_worker.asyncio.run") as asyncio_run,
    ):
        result = celery_worker.run_trading_loop.run(interval_seconds=12)

    assert result == {"ok": True, "interval_seconds": 12}
    runner_loop.assert_called_once_with(run_once=False, interval_seconds=12)
    asyncio_run.assert_called_once_with("loop-coro")


def test_run_trading_loop_task_reads_default_interval_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOT_INTERVAL_SECONDS", "22")

    with (
        patch(
            "app.workers.celery_worker.runner_loop",
            new=MagicMock(return_value="loop-coro"),
        ) as runner_loop,
        patch("app.workers.celery_worker.asyncio.run"),
    ):
        result = celery_worker.run_trading_loop.run()

    assert result == {"ok": True, "interval_seconds": 22}
    runner_loop.assert_called_once_with(run_once=False, interval_seconds=22)
