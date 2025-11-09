# main.py

import os
import sys
import time
import signal
import logging
import asyncio
import warnings
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from sklearn.exceptions import UndefinedMetricWarning

# -------------------- Imports --------------------
from src.config import (
    FEATURE_COLUMNS, MODEL_SAVE_PATH, CONFIDENCE_THRESHOLD, TARGET_COLUMN,
    DATA_FILE_PATH, TRADE_SYMBOL, TRADE_INTERVAL, TRADE_QUANTITY,
    INITIAL_CANDLES_HISTORY, TRANSACTION_FEE_PCT
)
from src.data_loader import load_and_preprocess_data
from src.feature_engineer import calculate_technical_indicators, normalize_features
from src.model_manager import prepare_model_data, train_xgboost_model, make_predictions, load_trained_model
from src.backtester import backtest_strategy
from src.visualizer import visualize_trading_results
from src.binance_manager import BinanceManager
from src.db import init_db, SessionLocal, Trade
from src.notifier import TelegramNotifier, send_email_notification

# -------------------- Setup --------------------
load_dotenv()
if os.path.exists(".env.local"):
    load_dotenv(dotenv_path=".env.local")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s"
)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

init_db()
logger = logging.getLogger(__name__)
telegram_notifier = TelegramNotifier()

# -------------------- Graceful Shutdown --------------------
def shutdown_handler(signum, frame):
    logger.info("🛑 Shutdown signal received — cleaning up before exit...")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


# -------------------- Backtesting Pipeline --------------------
def run_backtesting_pipeline(train_new_model: bool = False):
    """
    Run backtesting pipeline:
    - Loads data, trains/reuses model
    - Generates predictions
    - Backtests strategy
    """
    logger.info("Starting backtesting pipeline...")

    df = load_and_preprocess_data(DATA_FILE_PATH)
    if df is None or df.empty:
        logger.error("❌ No data available for backtesting.")
        return None, None, None

    model_exists = os.path.exists(MODEL_SAVE_PATH)
    if train_new_model or not model_exists:
        logger.info("🚀 Training new XGBoost model...")
        X_train, X_test, y_train, y_test = prepare_model_data(df)
        model = train_xgboost_model(X_train, y_train, X_test, y_test)
        send_email_notification("Model Training Complete", "✅ New XGBoost model trained.")
        asyncio.create_task(telegram_notifier.send_message("✅ Model trained successfully."))
    else:
        model = load_trained_model()
        logger.info("✅ Using existing trained model.")

    decoded_preds, confidence = make_predictions(model, df)
    backtest_results = backtest_strategy(df, decoded_preds, confidence)

    logger.info("✅ Backtesting pipeline completed successfully.")
    return df, model, backtest_results


# -------------------- Live Trading Loop --------------------
async def run_live_trade_loop(model):
    """
    Continuous live trading loop:
    - Fetches latest candles
    - Calculates indicators
    - Predicts trading signals
    - Executes trades (buy/sell)
    """
    binance_manager = BinanceManager()
    logger.info(f"📈 Starting live trading loop for {TRADE_SYMBOL}")

    live_candles_history = binance_manager.get_latest_ohlcv_candles(
        symbol=TRADE_SYMBOL, interval=TRADE_INTERVAL, limit=INITIAL_CANDLES_HISTORY
    )
    if live_candles_history.empty:
        logger.error("❌ Could not fetch live candles. Exiting loop.")
        return

    current_position = 0
    interval_seconds = binance_manager._interval_to_minutes(TRADE_INTERVAL) * 60

    while True:
        try:
            latest_candle = binance_manager.get_latest_ohlcv_candles(
                symbol=TRADE_SYMBOL, interval=TRADE_INTERVAL, limit=1
            )

            if latest_candle.empty:
                await asyncio.sleep(interval_seconds)
                continue

            live_candles_history = pd.concat([live_candles_history, latest_candle]).tail(INITIAL_CANDLES_HISTORY)
            df_live = calculate_technical_indicators(live_candles_history)
            df_live = normalize_features(df_live)

            X_live = df_live[FEATURE_COLUMNS].iloc[[-1]]
            live_signal, confidence = make_predictions(model, X_live)
            signal_value = live_signal.iloc[0]

            message = f"📊 {TRADE_SYMBOL} | Signal: {signal_value} | Confidence: {confidence.iloc[0]:.2f}"
            logger.info(message)
            asyncio.create_task(telegram_notifier.send_message(message))

            # --- Example basic trading logic ---
            if signal_value == 1 and current_position == 0:
                binance_manager.place_market_order(TRADE_SYMBOL, TRADE_QUANTITY, "buy")
                current_position = 1
                asyncio.create_task(telegram_notifier.send_message(f"✅ BUY executed for {TRADE_SYMBOL}"))
            elif signal_value == -1 and current_position == 1:
                binance_manager.place_market_order(TRADE_SYMBOL, TRADE_QUANTITY, "sell")
                current_position = 0
                asyncio.create_task(telegram_notifier.send_message(f"✅ SELL executed for {TRADE_SYMBOL}"))

            await asyncio.sleep(interval_seconds)

        except Exception as e:
            error_msg = f"⚠️ Live trading loop error: {e}"
            logger.error(error_msg, exc_info=True)
            send_email_notification("Trading Bot Error", error_msg)
            await telegram_notifier.send_message(error_msg)
            await asyncio.sleep(60)


# -------------------- Unified Main Command Logic --------------------
async def main(train: bool = False, mode: str = "backtest"):
    """
    Unified entrypoint:
    - Handles both backtesting and live modes.
    - Loads or trains models intelligently.
    """
    try:
        await telegram_notifier.send_message("🚀 Trading bot started.")
        logger.info("🚀 Trading bot initialized.")

        df, model, results = run_backtesting_pipeline(train_new_model=train)

        if mode == "live":
            if model:
                await telegram_notifier.send_message("📡 Switching to LIVE trading mode...")
                await run_live_trade_loop(model)
            else:
                logger.critical("❌ Cannot start live trading — model not available.")
                await telegram_notifier.send_message("❌ Live trading aborted — no model available.")
        else:
            logger.info("📊 Backtesting complete. Generating visualizations...")
            if results is not None:
                visualize_trading_results(results, trades_df, daily_portfolio_df, rsi_lower_thresh=30, rsi_upper_thresh=70)
                await telegram_notifier.send_message("✅ Backtesting and visualization completed.")
            else:
                logger.warning("⚠️ Backtesting results unavailable — skipping visualization.")

    except Exception as e:
        logger.exception("❌ Fatal error in main(): %s", e)
        send_email_notification("❌ Bot Failure", str(e))
        await telegram_notifier.send_message(f"❌ Critical error: {e}")


# -------------------- CLI Entrypoint --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument("--train", action="store_true", help="Force retrain model before trading.")
    parser.add_argument("--mode", choices=["backtest", "live"], default="backtest", help="Choose operation mode.")
    args = parser.parse_args()

    try:
        asyncio.run(main(train=args.train, mode=args.mode))
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user — shutting down gracefully.")
        sys.exit(0)
