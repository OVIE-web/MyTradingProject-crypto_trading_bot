"""Scheduling helpers for periodic trading bot jobs."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

from app.workers.bot_runner import DEFAULT_INTERVAL, run_once_test, runner_loop

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SchedulerSettings:
    """Runtime settings for the trading scheduler."""

    interval_seconds: int = DEFAULT_INTERVAL
    run_once: bool = False

    @classmethod
    def from_env(cls) -> SchedulerSettings:
        """Build scheduler settings from environment variables."""
        return cls(
            interval_seconds=_get_env_int("BOT_INTERVAL_SECONDS", DEFAULT_INTERVAL),
            run_once=os.getenv("BOT_RUN_ONCE", "0") == "1",
        )


async def run_scheduler(settings: SchedulerSettings | None = None) -> None:
    """Run the async trading scheduler loop."""
    scheduler_settings = settings or SchedulerSettings.from_env()

    LOG.info(
        "Starting scheduler. interval_seconds=%s run_once=%s",
        scheduler_settings.interval_seconds,
        scheduler_settings.run_once,
    )

    if scheduler_settings.run_once:
        await run_once_test()
        return

    await runner_loop(
        run_once=False,
        interval_seconds=scheduler_settings.interval_seconds,
    )


def run_scheduler_sync(settings: SchedulerSettings | None = None) -> None:
    """Run the scheduler from a synchronous process entrypoint."""
    asyncio.run(run_scheduler(settings))


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


def main() -> None:
    """CLI entrypoint for running the scheduler directly."""
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    run_scheduler_sync()


if __name__ == "__main__":
    main()
