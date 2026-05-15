"""Celery application and tasks for background trading jobs."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from celery import Celery

from app.workers.bot_runner import DEFAULT_INTERVAL, run_once_test, runner_loop

LOG = logging.getLogger(__name__)

DEFAULT_BROKER_URL = "redis://localhost:6379/0"
DEFAULT_RESULT_BACKEND = "redis://localhost:6379/1"


def create_celery_app() -> Celery:
    """Create and configure the Celery app."""
    celery_app = Celery(
        "crypto_trading_bot",
        broker=os.getenv("CELERY_BROKER_URL", DEFAULT_BROKER_URL),
        backend=os.getenv("CELERY_RESULT_BACKEND", DEFAULT_RESULT_BACKEND),
    )

    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone=os.getenv("CELERY_TIMEZONE", "UTC"),
        enable_utc=True,
        task_track_started=True,
        worker_prefetch_multiplier=1,
        broker_connection_retry_on_startup=True,
    )

    celery_app.conf.beat_schedule = {
        "run-trading-bot-iteration": {
            "task": "app.workers.celery_worker.run_trading_iteration",
            "schedule": _get_env_int("BOT_INTERVAL_SECONDS", DEFAULT_INTERVAL),
        }
    }

    return celery_app


celery_app = create_celery_app()


@celery_app.task(name="app.workers.celery_worker.run_trading_iteration")
def run_trading_iteration() -> dict[str, Any]:
    """Run one trading bot iteration as a Celery task."""
    LOG.info("Celery task started: run_trading_iteration")
    success = asyncio.run(run_once_test())
    return {"ok": success}


@celery_app.task(name="app.workers.celery_worker.run_trading_loop")
def run_trading_loop(interval_seconds: int | None = None) -> dict[str, Any]:
    """Run the long-lived trading loop as a Celery task."""
    interval = interval_seconds or _get_env_int("BOT_INTERVAL_SECONDS", DEFAULT_INTERVAL)
    LOG.info("Celery task started: run_trading_loop interval_seconds=%s", interval)
    asyncio.run(runner_loop(run_once=False, interval_seconds=interval))
    return {"ok": True, "interval_seconds": interval}


def _get_env_int(key: str, default: int) -> int:
    raw_value = os.getenv(key)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc

    if value < 1:
        raise ValueError(f"{key} must be greater than 0")

    return value


__all__ = [
    "celery_app",
    "create_celery_app",
    "run_trading_iteration",
    "run_trading_loop",
]
