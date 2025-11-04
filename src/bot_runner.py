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

# --- Notification modules ---
from src.notifier import TelegramNotifier, send_email_notification as send_email_async_safe
from src.notification import send_telegram_notification, send_email_notification  # sync fallback

LOG = logging.getLogger("bot_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_INTERVAL = int(os.getenv("BOT_INTERVAL_SECONDS", "60"))
CONCURRENCY_LIMIT = int(os.getenv("BOT_CONCURRENCY", "3"))


# ======================================================================================
#  RESOURCE LIFESPAN MANAGEMENT
# ======================================================================================
@asynccontextmanager
async def lifespan():
    """Startup/teardown for trading bot resources."""
    LOG.info("Starting bot lifespan: init resources...")

    model, _ = load_trained_model(return_metadata=True)
    binance = BinanceManager()
    notifier = TelegramNotifier(max_retries=3)
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


# ======================================================================================
#  SCHEDULING HELPERS
# ======================================================================================
def is_time_to_run(last_run: Optional[datetime], interval_seconds: int) -> bool:
    if last_run is None:
        return True
    return (datetime.now(timezone.utc) - last_run).total_seconds() >= interval_seconds


def should_skip_if_running(last_task_finished: bool) -> bool:
    # Prevent overlapping runs
    return not last_task_finished


# ======================================================================================
#  CORE TRADING ITERATION
# ======================================================================================
@backoff.on_exception(backoff.expo, Exception, max_tries=5, jitter=backoff.full_jitter)
async def do_iteration(resources: Dict[str, object]) -> None:
    """
    One full trading iteration:
    - Fetch latest OHLCV
    - Compute indicators
    - Make predictions
    - Execute mock trades
    - Send notifications (async + sync)
    """
    LOG.info("Starting iteration")

    candles = await resources["binance"].get_latest_ohlcv("BTCUSDT", "4h")
    features = calculate_technical_indicators(candles)
    decoded_preds, confidence = make_predictions(resources["model"], features)

    latest_signal = int(decoded_preds.iloc[-1]) if not decoded_preds.empty else 0
    latest_conf = float(confidence.iloc[-1]) if not confidence.empty else 0.0
    notifier = resources["notifier"]

    trade_message = None
    symbol = "BTCUSDT"

    # --- BUY SIGNAL ---
    if latest_signal == 1:
        trade_message = (
            f"✅ BUY signal detected for {symbol}\n"
            f"Confidence: {latest_conf:.2f}"
        )
        LOG.info(trade_message)

        trade_result = {"symbol": symbol, "side": "BUY", "status": "FILLED", "price": "68000", "qty": "0.001"}

        if trade_result.get("status") == "FILLED":
            msg = (
                f"🚀 Trade Executed Successfully!\n"
                f"Pair: {symbol}\n"
                f"Side: BUY\n"
                f"Price: {trade_result['price']}\n"
                f"Qty: {trade_result['qty']}\n"
                f"Confidence: {latest_conf:.2f}"
            )
            await notify_all_channels(notifier, "Trade Executed (BUY)", msg)

    # --- SELL SIGNAL ---
    elif latest_signal == -1:
        trade_message = (
            f"❌ SELL signal detected for {symbol}\n"
            f"Confidence: {latest_conf:.2f}"
        )
        LOG.info(trade_message)

        trade_result = {"symbol": symbol, "side": "SELL", "status": "FILLED", "price": "68200", "qty": "0.001"}

        if trade_result.get("status") == "FILLED":
            msg = (
                f"📉 Trade Executed Successfully!\n"
                f"Pair: {symbol}\n"
                f"Side: SELL\n"
                f"Price: {trade_result['price']}\n"
                f"Qty: {trade_result['qty']}\n"
                f"Confidence: {latest_conf:.2f}"
            )
            await notify_all_channels(notifier, "Trade Executed (SELL)", msg)

    else:
        LOG.info("No actionable signal this round (HOLD position).")

    LOG.info("Iteration complete")


# ======================================================================================
#  NOTIFICATION ROUTER (ASYNC + SYNC)
# ======================================================================================
async def notify_all_channels(notifier: TelegramNotifier, subject: str, message: str):
    """
    Sends notifications through all available channels:
    1. Async TelegramNotifier (with retry)
    2. Sync Telegram REST fallback
    3. Sync Email (with retry & backoff)
    """
    # --- Async TelegramNotifier (primary) ---
    try:
        await notifier.send_message(message)
    except Exception as e:
        LOG.error(f"Async Telegram notifier failed: {e}")
        # fallback to sync
        send_telegram_notification(message)

    # --- Email notification (synchronous with retry) ---
    try:
        send_email_async_safe(subject, message)
    except Exception as e:
        LOG.warning(f"Primary email notifier failed: {e}, using sync fallback.")
        try:
            send_email_notification(subject, message)
        except Exception as e2:
            LOG.error(f"Fallback email failed: {e2}")


# ======================================================================================
#  RUNNER LOOP
# ======================================================================================
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


# ======================================================================================
#  ENTRYPOINTS
# ======================================================================================
async def run_once_test() -> bool:
    async with lifespan() as resources:
        try:
            await do_iteration(resources)
            LOG.info("✅ Test run_once_test() completed successfully")
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
