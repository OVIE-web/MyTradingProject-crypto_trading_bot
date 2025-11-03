# src/bot_runner.py
import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Optional

import backoff

from src.binance_manager import BinanceManager
from src.feature_engineer import calculate_technical_indicators
from src.model_manager import load_trained_model, make_predictions
from src.db import SessionLocal
from src.notifier import TelegramNotifier, send_email_notification  # ✅ Added email notifier import

LOG = logging.getLogger("bot_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_INTERVAL = int(os.getenv("BOT_INTERVAL_SECONDS", "60"))
CONCURRENCY_LIMIT = int(os.getenv("BOT_CONCURRENCY", "3"))


@asynccontextmanager
async def lifespan():
    """
    Startup/teardown for bot resources.
    Loads the trained model, initializes Binance manager, notifier, and DB session.
    """
    LOG.info("Starting bot lifespan: init resources...")
    model = load_trained_model()
    binance = BinanceManager()
    notifier = TelegramNotifier()
    db = SessionLocal()

    resources: Dict[str, object] = {
        "model": model,
        "binance": binance,
        "notifier": notifier,
        "db": db,
    }

    try:
        yield resources
    finally:
        LOG.info("Shutting down bot lifespan: close resources...")
        try:
            db.close()
        except Exception:
            LOG.exception("Error closing DB session")


def is_time_to_run(last_run: Optional[datetime], interval_seconds: int) -> bool:
    if last_run is None:
        return True
    return (datetime.now(timezone.utc) - last_run).total_seconds() >= interval_seconds


def should_skip_if_running(last_task_finished: bool) -> bool:
    # Policy: skip overlapping runs to avoid race conditions
    return not last_task_finished


@backoff.on_exception(backoff.expo, Exception, max_tries=5, jitter=backoff.full_jitter)
async def do_iteration(resources: Dict[str, object]) -> None:
    """
    One iteration of the trading workflow:
    - fetch market data
    - calculate indicators
    - make predictions with confidence
    - optionally notify/place orders
    - record trades in DB
    """
    LOG.info("Starting iteration")

    candles = await resources["binance"].get_latest_ohlcv("BTCUSDT", "4h")
    features = calculate_technical_indicators(candles)

    decoded_preds, confidence = make_predictions(resources["model"], features)
    latest_signal = int(decoded_preds.iloc[-1]) if not decoded_preds.empty else 0
    latest_conf = float(confidence.iloc[-1]) if not confidence.empty else 0.0

    notifier = resources["notifier"]

    # ✅ --- START: Real-time alert integration block ---
    trade_message = None

    if latest_signal == 1:  # BUY
        trade_message = f"✅ BUY signal detected for BTCUSDT (confidence={latest_conf:.2f})"
        LOG.info(trade_message)

        # Example simulated trade (replace with actual order execution)
        # trade_result = await resources["binance"].place_market_order("BTCUSDT", "BUY", 0.001)
        trade_result = {"symbol": "BTCUSDT", "side": "BUY", "status": "FILLED", "price": "68000", "qty": "0.001"}

        if trade_result.get("status") == "FILLED":
            message = (
                f"✅ Trade Executed Successfully!\n"
                f"Pair: {trade_result['symbol']}\n"
                f"Side: {trade_result['side']}\n"
                f"Quantity: {trade_result['qty']}\n"
                f"Price: {trade_result['price']}\n"
                f"Confidence: {latest_conf:.2f}"
            )
            try:
                notifier.send_message(message)
            except Exception as e:
                LOG.error(f"Telegram notification failed: {e}")

            try:
                send_email_notification("Trade Executed", message)
            except Exception as e:
                LOG.error(f"Email notification failed: {e}")

    elif latest_signal == -1:  # SELL
        trade_message = f"❌ SELL signal detected for BTCUSDT (confidence={latest_conf:.2f})"
        LOG.info(trade_message)

        trade_result = {"symbol": "BTCUSDT", "side": "SELL", "status": "FILLED", "price": "68200", "qty": "0.001"}

        if trade_result.get("status") == "FILLED":
            message = (
                f"✅ Trade Executed Successfully!\n"
                f"Pair: {trade_result['symbol']}\n"
                f"Side: {trade_result['side']}\n"
                f"Quantity: {trade_result['qty']}\n"
                f"Price: {trade_result['price']}\n"
                f"Confidence: {latest_conf:.2f}"
            )
            try:
                notifier.send_message(message)
            except Exception as e:
                LOG.error(f"Telegram notification failed: {e}")

            try:
                send_email_notification("Trade Executed", message)
            except Exception as e:
                LOG.error(f"Email notification failed: {e}")

    else:
        LOG.info("No actionable signal this round (HOLD position).")
    # ✅ --- END: Real-time alert integration block ---

    # Persist to DB (replace with real ORM model and fields)
    session = resources["db"]
    try:
        # trade = Trade(symbol="BTCUSDT", signal=latest_signal, confidence=latest_conf, timestamp=datetime.utcnow())
        # session.add(trade)
        # session.commit()
        pass
    except Exception as exc:
        LOG.exception("DB persistence failed: %s", exc)
        try:
            session.rollback()
        except Exception:
            LOG.exception("DB rollback failed")

    LOG.info("Iteration complete")


async def runner_loop(run_once: bool = False, interval_seconds: int = DEFAULT_INTERVAL):
    last_run_finished = True
    last_run_time: Optional[datetime] = None
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async with lifespan() as resources:
        stop_event = asyncio.Event()

        def _stop_signal():
            LOG.info("Received stop signal")
            stop_event.set()

        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, _stop_signal)
            loop.add_signal_handler(signal.SIGTERM, _stop_signal)
        except NotImplementedError:
            LOG.warning("Signal handlers not supported; relying on stop_event externally")

        background_tasks = set()

        while not stop_event.is_set():
            if should_skip_if_running(last_run_finished):
                LOG.warning("Previous run still in progress; skipping scheduled run.")
            elif is_time_to_run(last_run_time, interval_seconds):
                last_run_finished = False
                last_run_time = datetime.now(timezone.utc)

                async def _run_task():
                    nonlocal last_run_finished
                    try:
                        async with sem:
                            await do_iteration(resources)
                    except Exception as exc:
                        LOG.exception("Iteration failed: %s", exc)
                    finally:
                        last_run_finished = True

                task = asyncio.create_task(_run_task())
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)

            if run_once:
                LOG.info("Run-once mode finished; exiting loop")
                break

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                continue

        LOG.info("Runner loop exiting; waiting for background tasks to finish...")
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)


async def run_once_test() -> bool:
    async with lifespan() as resources:
        try:
            await do_iteration(resources)
            LOG.info("Test run_once_test() completed successfully")
            return True
        except Exception as e:
            LOG.error(f"Error during single test iteration: {e}")
            return False


def main():
    run_once_flag = os.getenv("BOT_RUN_ONCE", "0") == "1"
    interval = int(os.getenv("BOT_INTERVAL_SECONDS", DEFAULT_INTERVAL))

    if run_once_flag:
        asyncio.run(run_once_test())
    else:
        asyncio.run(runner_loop(run_once=run_once_flag, interval_seconds=interval))


if __name__ == "__main__":
    main()
