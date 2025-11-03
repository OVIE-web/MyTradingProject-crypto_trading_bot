# src/bot_runner.py
import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import partial
from typing import Callable

import backoff
# optionally: from prometheus_client import start_http_server, Counter, Gauge

# Example imports from your project
from src.binance_manager import BinanceManager
from src.feature_engineer import calculate_technical_indicators
from src.model_manager import load_trained_model
from src.db import SessionLocal
from src.notifier import TelegramNotifier

LOG = logging.getLogger("bot_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_INTERVAL = int(os.getenv("BOT_INTERVAL_SECONDS", "60"))
CONCURRENCY_LIMIT = int(os.getenv("BOT_CONCURRENCY", "3"))

# Optional metrics (requires prometheus_client)
# TASKS_TOTAL = Counter("bot_tasks_total", "Total task runs")
# TASKS_FAILED = Counter("bot_tasks_failed", "Failed task runs")
# LAST_RUN = Gauge("bot_last_run_unixtime", "Last successful run epoch seconds")


@asynccontextmanager
async def lifespan():
    """
    Put startup / teardown here. Example: init DB pool, load model, create manager instances.
    """
    LOG.info("Starting bot lifespan: init resources...")
    model = load_trained_model()
    binance = BinanceManager()
    notifier = TelegramNotifier()
    db = SessionLocal()
    try:
        yield {
             "model": model,
             "binance": binance,
             "notifier": notifier,
             "db": db,
        }
    finally:
        LOG.info("Shutting down bot lifespan: close resources...")
        db.close()


def is_time_to_run(last_run: datetime, interval_seconds: int):
    if last_run is None:
        return True
    return (datetime.now(timezone.utc) - last_run).total_seconds() >= interval_seconds


def should_skip_if_running(last_task_finished: bool) -> bool:
    # Implement policy: skip overlapping run or wait until finished
    return not last_task_finished


@backoff.on_exception(backoff.expo, Exception, max_tries=5, jitter=backoff.full_jitter)
async def do_iteration(resources):
    """
    One iteration of the trading workflow. Keep idempotency in mind.
    - fetch market data
    - calculate indicators/predictions
    - optionally place orders
    - record trades in DB
    """
    LOG.info("Starting iteration")
    #TASKS_TOTAL.inc()
    # Example (pseudocode):
    candles = await resources["binance"].get_latest_ohlcv("BTCUSDT", "4h")
    features = calculate_technical_indicators(candles)
    preds = resources["model"].predict(features)
    
    # Example: notify if condition met

    if preds["signal"] == "BUY":
        resources["notifier"].send_message("Buy signal detected for BTC/USDT")
        
    session = resources["db"]
    session.add(Trade(...))
    session.commit()


# place orders, persist to DB, send notifications
    await asyncio.sleep(0.1)  # replace with actual work
    LOG.info("Iteration complete")
    # LAST_RUN.set(time.time())


async def runner_loop(run_once: bool = False, interval_seconds: int = DEFAULT_INTERVAL):
    last_run_finished = True
    last_run_time = None
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async with lifespan() as resources:
        # Setup graceful shutdown
        stop_event = asyncio.Event()

        def _stop_signal():
            LOG.info("Received stop signal")
            stop_event.set()

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, _stop_signal)
        loop.add_signal_handler(signal.SIGTERM, _stop_signal)

        while not stop_event.is_set():
            if not last_run_finished:
                LOG.warning("Previous run still in progress; skipping this scheduled run.")
            elif is_time_to_run(last_run_time, interval_seconds):
                last_run_finished = False
                last_run_time = datetime.now(timezone.utc)

                async def _run_task():
                    nonlocal last_run_finished
                    try:
                        async with sem:  # limit concurrency
                            await do_iteration(resources)
                    except Exception as exc:
                        LOG.exception("Iteration failed: %s", exc)
                        # TASKS_FAILED.inc()
                    finally:
                        last_run_finished = True

                # Run without awaiting to allow concurrency limited by sem
                asyncio.create_task(_run_task())

            if run_once:
                LOG.info("Run-once mode finished; exiting loop")
                break

            # sleep with cancellation support
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                continue

        LOG.info("Runner loop exiting; waiting for background tasks to finish...")
        # Wait for currently running tasks (naive approach)
        await asyncio.sleep(0.5)
        
        
def run_once(self):
    """Execute a single trading iteration (used for testing)."""
    try:
        latest_data = self.data_loader.get_latest_data()
        features = self.feature_engineer.calculate_indicators(latest_data)
        prediction = self.model_manager.make_predictions(features)
        self.logger.info(f"Predicted signal: {prediction}")
        return prediction
    except Exception as e:
        self.logger.error(f"Error in single run: {e}")
        return None
# --- Single iteration for testing and CI sanity checks ---
async def run_once_test():
    """Run a single trading iteration (used by tests)."""
    async with lifespan() as resources:
        try:
            await do_iteration(resources)
            LOG.info("Test run_once_test() completed successfully")
            return True
        except Exception as e:
            LOG.error(f"Error during single test iteration: {e}")
            return False


def main():
    """Main entry point for trading bot runner."""
    run_once_flag = os.getenv("BOT_RUN_ONCE", "0") == "1"
    interval = int(os.getenv("BOT_INTERVAL_SECONDS", DEFAULT_INTERVAL))

    if run_once_flag:
        # Run a single iteration (for unit test/demo)
        asyncio.run(run_once_test())
    else:
        asyncio.run(runner_loop(run_once=run_once_flag, interval_seconds=interval))


if __name__ == "__main__":
    main()
