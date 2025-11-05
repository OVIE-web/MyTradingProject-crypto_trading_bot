# src/bot_runner.py
import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Any

import backoff

from src.binance_manager import BinanceManager
from src.feature_engineer import calculate_technical_indicators
from src.model_manager import load_trained_model, make_predictions
from src.db import SessionLocal

# async notifier (primary) + sync fallback helpers
from src.notifier import TelegramNotifier, send_email_notification as send_email_async_safe
from src.notification import send_telegram_notification, send_email_notification as send_email_sync

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

    # load_trained_model may return either model or (model, metadata).
    model_res = load_trained_model()
    if isinstance(model_res, tuple) and len(model_res) == 2:
        model, metadata = model_res
    else:
        model, metadata = model_res, {}

    binance = BinanceManager()
    notifier = TelegramNotifier(max_retries=3)
    db = SessionLocal()

    resources: Dict[str, object] = {
        "model": model,
        "metadata": metadata,
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
#  NOTIFICATION ROUTER (ASYNC PRIMARY + SYNC FALLBACKS)
# ======================================================================================
async def notify_all_channels(notifier: TelegramNotifier, subject: str, message: str) -> None:
    """
    Sends notifications through all available channels:
      1) Async TelegramNotifier (awaited)
      2) Sync Telegram REST fallback (requests)
      3) Async-safe (sync) Email fallback (synchronous with retry in notifier or sync helper)
    The function tries async primary first and falls back to sync helpers on errors.
    """
    # 1) Primary async Telegram notifier
    try:
        sent = await notifier.send_message(message)
        if sent:
            LOG.info("Primary async Telegram notification succeeded.")
        else:
            LOG.warning("Primary async Telegram notifier returned False. Falling back to sync Telegram.")
            # fallback to sync HTTP-based notification (requests)
            send_telegram_notification(message)
    except Exception as exc:
        LOG.error("Async Telegram notifier raised exception: %s. Falling back to sync Telegram.", exc)
        try:
            send_telegram_notification(message)
        except Exception as exc2:
            LOG.error("Sync Telegram fallback also failed: %s", exc2)

    # 2) Email (primary email in src.notifier is sync but production-ready)
    # try the notifier email first (send_email_async_safe is the same sync function in src.notifier)
    try:
        ok = send_email_async_safe(subject, message)
        if ok:
            LOG.info("Primary email notifier succeeded.")
        else:
            LOG.warning("Primary email notifier returned False. Attempting sync fallback email.")
            send_email_sync(subject, message)
    except Exception as exc:
        LOG.warning("Primary email notifier failed with exception: %s. Trying sync fallback.", exc)
        try:
            send_email_sync(subject, message)
        except Exception as exc2:
            LOG.error("Sync email fallback also failed: %s", exc2)


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
      - Execute mock trades (replace with real order logic)
      - Send notifications (async + sync fallback)
    """
    LOG.info("Starting iteration")

    # Fetch data
    candles = await resources["binance"].get_latest_ohlcv("BTCUSDT", "4h")
    features = calculate_technical_indicators(candles)

    # Make predictions (returns pd.Series and confidence)
    decoded_preds, confidence = make_predictions(resources["model"], features)

    latest_signal = int(decoded_preds.iloc[-1]) if not decoded_preds.empty else 0
    latest_conf = float(confidence.iloc[-1]) if not confidence.empty else 0.0
    notifier: TelegramNotifier = resources["notifier"]
    symbol = "BTCUSDT"

    # Prepare trade and notification flow
    if latest_signal == 1:
        LOG.info("BUY signal detected (conf=%.2f).", latest_conf)
        # Simulate order execution (replace with real order call)
        trade_result = {"symbol": symbol, "side": "BUY", "status": "FILLED", "price": "68000", "qty": "0.001"}
        if trade_result.get("status") == "FILLED":
            message = (
                f"🚀 Trade Executed Successfully!\n"
                f"Pair: {symbol}\n"
                f"Side: BUY\n"
                f"Price: {trade_result['price']}\n"
                f"Qty: {trade_result['qty']}\n"
                f"Confidence: {latest_conf:.2f}"
            )
            await notify_all_channels(notifier, "Trade Executed (BUY)", message)

    elif latest_signal == -1:
        LOG.info("SELL signal detected (conf=%.2f).", latest_conf)
        trade_result = {"symbol": symbol, "side": "SELL", "status": "FILLED", "price": "68200", "qty": "0.001"}
        if trade_result.get("status") == "FILLED":
            message = (
                f"📉 Trade Executed Successfully!\n"
                f"Pair: {symbol}\n"
                f"Side: SELL\n"
                f"Price: {trade_result['price']}\n"
                f"Qty: {trade_result['qty']}\n"
                f"Confidence: {latest_conf:.2f}"
            )
            await notify_all_channels(notifier, "Trade Executed (SELL)", message)
    else:
        LOG.info("No actionable signal this round (HOLD position).")

    # Persist trade record to DB here (example)
    session = resources["db"]
    try:
        # Example placeholder for saving trades to DB (uncomment and wire to ORM)
        # trade = Trade(symbol=symbol, side=trade_result['side'], price=trade_result['price'], qty=trade_result['qty'], confidence=latest_conf)
        # session.add(trade); session.commit()
        pass
    except Exception as exc:
        LOG.exception("DB persistence failed: %s", exc)
        try:
            session.rollback()
        except Exception:
            LOG.exception("DB rollback failed")

    LOG.info("Iteration complete")


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
        # install handlers where supported (not on Windows in some event loops)
        try:
            loop.add_signal_handler(signal.SIGINT, _stop_signal)
            loop.add_signal_handler(signal.SIGTERM, _stop_signal)
        except NotImplementedError:
            LOG.warning("Signal handlers not supported in this environment.")

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
            LOG.error("Error during single test iteration: %s", e)
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
