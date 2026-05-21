from __future__ import annotations

import app.workers as workers
from app.workers.scheduler import SchedulerSettings, run_scheduler, run_scheduler_sync


def test_workers_package_exports_scheduler_api() -> None:
    assert workers.__all__ == ["SchedulerSettings", "run_scheduler", "run_scheduler_sync"]


def test_workers_package_exports_match_scheduler_module() -> None:
    assert workers.SchedulerSettings is SchedulerSettings
    assert workers.run_scheduler is run_scheduler
    assert workers.run_scheduler_sync is run_scheduler_sync
