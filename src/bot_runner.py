# src/bot_runner.py
from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import backoff
import numpy as np
import pandas as pd

from src.binance_manager import BinanceManager
from src.config import ATR_WINDOW, FEATURE_COLUMNS, INITIAL_CANDLES_HISTORY
from src.db import SessionLocal, Trade
from src.feature_engineer import calculate_technical_indicators
from src.model_manager import load_trained_model, make_predictions
from src.notification import send_email_notification as send_email_sync
from src.notification import send_telegram_notification
from src.notifier import TelegramNotifier
from src.notifier import send_email_notification as send_email_async_safe

LOG = logging.getLogger("bot_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_INTERVAL = int(os.getenv("BOT_INTERVAL_SECONDS", "60"))
CONCURRENCY_LIMIT = int(os.getenv("BOT_CONCURRENCY", "3"))


# ======================================================================================
#  RESOURCE LIFESPAN MANAGEMENT
# ======================================================================================
@asynccontextmanager
async def lifespan() -> AsyncGenerator[dict[str, Any], None]:
    """
    Startup/teardown for trading bot resources.
    Yields:
        dict containing model, metadata, binance client, notifier, db session
    """
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

    resources: dict[str, Any] = {
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
def is_time_to_run(last_run: datetime | None, interval_seconds: int) -> bool:
    if last_run is None:
        return True
    return (datetime.now(UTC) - last_run).total_seconds() >= interval_seconds


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
      2) Sync Telegram REST fallback (requests via send_telegram_notification)
      3) Sync Email fallback (send_email_sync)
    """
    # Primary async Telegram notifier
    try:
        sent = await notifier.send_message(message)
        if sent:
            LOG.info("Primary async Telegram notification succeeded.")
        else:
            LOG.warning(
                "Primary async Telegram notifier returned False. Falling back to sync Telegram."
            )
            send_telegram_notification(message)
    except Exception as exc:
        LOG.error(
            "Async Telegram notifier raised exception: %s. Falling back to sync Telegram.", exc
        )
        try:
            send_telegram_notification(message)
        except Exception as exc2:
            LOG.error("Sync Telegram fallback also failed: %s", exc2)

    # Email (try notifier's email helper first and fallback to sync)
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
async def do_iteration(resources: dict[str, Any]) -> None:
    """
    One full trading iteration:
      - Fetch latest OHLCV
      - Compute indicators
      - Make predictions
      - Execute mock trades (replace with real order logic)
      - Send notifications (async + sync fallback)
    """
    LOG.info("Starting iteration")

    binance: BinanceManager = resources["binance"]
    model = resources["model"]
    notifier: TelegramNotifier = resources["notifier"]
    session = resources["db"]

    # === 1. Fetch data ===
    try:
        limit = max(INITIAL_CANDLES_HISTORY, ATR_WINDOW + 5)
        candles = binance.get_latest_ohlcv("BTCUSDT", "4h", limit=limit)
    except Exception:
        LOG.exception("Failed to fetch OHLCV data from Binance")
        return

    # === 2. Compute features ===
    try:
        features_df = calculate_technical_indicators(candles)

        # Check all required features exist
        missing_features = [f for f in FEATURE_COLUMNS if f not in features_df.columns]
        if missing_features:
            LOG.error(f"Missing required features: {missing_features}")
            return

        # Select only the model-expected features in correct order
        features_df = features_df[FEATURE_COLUMNS]

        LOG.info(f"Features computed: shape={features_df.shape}, cols={list(features_df.columns)}")

    except Exception:
        LOG.exception("Failed to calculate technical indicators")
        return

    if features_df.empty:
        LOG.warning("No valid features after computation (all NaN rows dropped)")
        return

    # === 3. Make predictions ===
    try:
        # Pass the filtered DataFrame directly
        decoded_preds, confidence = make_predictions(model, features_df)
    except ValueError as e:
        LOG.error(f"Prediction error: {e}")
        return
    except Exception:
        LOG.exception("Prediction failed")
        return

    # === 4. Extract latest signal and confidence ===
    if isinstance(decoded_preds, (list, np.ndarray)):
        latest_signal = int(decoded_preds[-1]) if len(decoded_preds) > 0 else 0
    elif isinstance(decoded_preds, pd.DataFrame) and not decoded_preds.empty:
        latest_signal = int(decoded_preds.iloc[-1].iloc[0])
    else:
        latest_signal = 0

    if isinstance(confidence, (list, np.ndarray)):
        latest_conf = float(confidence[-1]) if len(confidence) > 0 else 0.0
    elif isinstance(confidence, pd.DataFrame) and not confidence.empty:
        latest_conf = float(confidence.iloc[-1].iloc[0])
    else:
        latest_conf = 0.0

    LOG.info(f"Prediction result: signal={latest_signal}, confidence={latest_conf:.4f}")

    symbol = "BTCUSDT"

    # === 5. Execute trades based on signal ===
    trade_result: dict[str, Any] | None = None

    if latest_signal == 1:
        LOG.info("BUY signal detected (conf=%.2f).", latest_conf)
        # Simulate order execution (replace with real order call)
        trade_result = {
            "symbol": symbol,
            "side": "BUY",
            "status": "FILLED",
            "price": "68000",
            "qty": "0.001",
        }
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
        trade_result = {
            "symbol": symbol,
            "side": "SELL",
            "status": "FILLED",
            "price": "68200",
            "qty": "0.001",
        }
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

    # === 6. Persist trade record ===
    if trade_result is not None:
        try:
            assert trade_result is not None
            trade = Trade(
                symbol=symbol,
                side=trade_result["side"],
                price=trade_result["price"],
                qty=trade_result["qty"],
                confidence=latest_conf,
            )
            session.add(trade)
            session.commit()
        except Exception as exc:
            LOG.exception("DB persistence failed: %s", exc)
            try:
                if hasattr(session, "rollback"):
                    session.rollback()
            except Exception:
                LOG.exception("DB rollback failed")

    LOG.info("Iteration complete")


# ======================================================================================
#  RUNNER LOOP
# ======================================================================================
async def runner_loop(run_once: bool = False, interval_seconds: int = DEFAULT_INTERVAL) -> None:
    """
    Main event loop that schedules iterations and handles graceful shutdown.
    """
    last_run_finished = True
    last_run_time: datetime | None = None
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async with lifespan() as resources:
        stop_event = asyncio.Event()

        def _stop_signal() -> None:
            LOG.info("Received stop signal")
            stop_event.set()

        loop = asyncio.get_running_loop()
        # install handlers where supported (not on Windows in some event loops)
        try:
            loop.add_signal_handler(signal.SIGINT, _stop_signal)
            loop.add_signal_handler(signal.SIGTERM, _stop_signal)
        except NotImplementedError:
            LOG.debug("Signal handlers not supported in this environment.")

        background_tasks: set[asyncio.Task[Any]] = set()

        while not stop_event.is_set():
            if should_skip_if_running(last_run_finished):
                LOG.warning("Previous run still in progress; skipping scheduled run.")
            elif is_time_to_run(last_run_time, interval_seconds):
                last_run_finished = False
                last_run_time = datetime.now(UTC)

                async def _run_task() -> None:
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
            except TimeoutError:
                continue

        LOG.info("Runner loop exiting; waiting for background tasks to finish...")
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)


# ======================================================================================
#  ENTRYPOINTS
# ======================================================================================
async def run_once_test() -> bool:
    """
    Convenience helper for running a single iteration (returns True if succeed).
    """
    async with lifespan() as resources:
        try:
            await do_iteration(resources)
            LOG.info("✅ Test run_once_test() completed successfully")
            return True
        except Exception as e:
            LOG.error("Error during single test iteration: %s", e)
            return False


def main() -> None:
    run_once_flag = os.getenv("BOT_RUN_ONCE", "0") == "1"
    interval = int(os.getenv("BOT_INTERVAL_SECONDS", str(DEFAULT_INTERVAL)))

    if run_once_flag:
        asyncio.run(run_once_test())
    else:
        asyncio.run(runner_loop(run_once=run_once_flag, interval_seconds=interval))


if __name__ == "__main__":
    main()
