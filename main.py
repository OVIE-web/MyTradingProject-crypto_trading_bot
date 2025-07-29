# main.py

import logging
import warnings
import os
import time
import pandas as pd
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
import requests
from datetime import datetime
from sklearn.exceptions import UndefinedMetricWarning
from binance.client import SIDE_BUY, SIDE_SELL

# Import only the config constants actually used in this file
from src.config import (
    FEATURE_COLUMNS, MODEL_SAVE_PATH, CONFIDENCE_THRESHOLD, TARGET_COLUMN,
    DATA_FILE_PATH, TRADE_SYMBOL, TRADE_INTERVAL, TRADE_QUANTITY,
    INITIAL_CANDLES_HISTORY, TRANSACTION_FEE_PCT
)

# Import only the functions/classes used directly in main.py
from src.data_loader import load_and_preprocess_data
from src.feature_engineer import calculate_technical_indicators, get_rsi_quantile_thresholds, apply_rsi_labels, normalize_features
from src.model_manager import prepare_model_data, train_xgboost_model, make_predictions, load_trained_model
from src.backtester import backtest_strategy
from src.visualizer import visualize_trading_results
from src.binance_manager import BinanceManager
from src.db import init_db, SessionLocal, Trade
from src.notifier import TelegramNotifier

# Load environment variables from .env file
load_dotenv()

# Configure logging and warnings globally
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

init_db()

# Initialize Telegram notifier
telegram_notifier = TelegramNotifier()

print("✅ Setup Complete: All required libraries imported and configuration set.")

# Global variables for live trading state
live_candles_history = pd.DataFrame()
current_position_live = 0  # 0: no position, 1: long position

def run_backtesting_pipeline(train_new_model=True):
    """
    Executes the complete backtesting pipeline using historical data.
    """
    logging.info("Starting the backtesting pipeline...")
    try:
        raw_price_data = load_and_preprocess_data(DATA_FILE_PATH)
        
        if raw_price_data.empty:
            raise ValueError("Initial data loading resulted in an empty DataFrame. Cannot proceed.")

        df_for_features_and_signals = raw_price_data.copy()
        df_for_features_and_signals = calculate_technical_indicators(df_for_features_and_signals)
        logging.info('✅ Technical indicators calculated')
        
        current_features = list(df_for_features_and_signals.columns)
        missing_initial_features = [col for col in FEATURE_COLUMNS if col not in current_features]
        if missing_initial_features:
            logging.error(f"Features missing after technical indicator calculation: {missing_initial_features}")
            raise ValueError(f"Required features missing: {missing_initial_features}")
            
        logging.info(f'Available features after indicator calculation: {sorted(current_features)}')
        
        lower_thresh, upper_thresh = get_rsi_quantile_thresholds(df_for_features_and_signals['rsi'])
        df_for_features_and_signals = apply_rsi_labels(df_for_features_and_signals, lower_threshold=lower_thresh, upper_threshold=upper_thresh)
        logging.info('✅ Trading signals generated')
        
        df_normalized = normalize_features(df_for_features_and_signals.copy())
        logging.info('✅ Features normalized')
        
        missing_normalized = [col for col in FEATURE_COLUMNS if col not in df_normalized.columns]
        if missing_normalized:
            logging.error(f"Features missing after normalization: {missing_normalized}")
            raise ValueError(f"Features missing after normalization: {missing_normalized}")
        
        # FIX: Ensure the variable returned matches the one defined
        df_original_aligned = raw_price_data.loc[df_normalized.index] # This line correctly defines df_original_aligned
        
        model = None
        if train_new_model:
            X_train, X_test, y_train, y_test = prepare_model_data(
                df_normalized, feature_cols=FEATURE_COLUMNS, target_col=TARGET_COLUMN
            )
            model, _ = train_xgboost_model(X_train, y_train, X_test, y_test)
            logging.info('✅ Model trained and saved.')
        else:
            try:
                model = load_trained_model()
                logging.info('✅ Existing model loaded.')
            except Exception as e:
                logging.warning(f"Could not load existing model: {e}. Training a new model instead.")
                X_train, X_test, y_train, y_test = prepare_model_data(
                    df_normalized, feature_cols=FEATURE_COLUMNS, target_col=TARGET_COLUMN
                )
                model, _ = train_xgboost_model(X_train, y_train, X_test, y_test)
                logging.info('✅ New model trained and saved (fallback).')

        if model is None:
            raise RuntimeError("Failed to train or load a model.")

        X_full_for_predictions = df_normalized[FEATURE_COLUMNS]
        predictions = make_predictions(model, X_full_for_predictions, confidence_threshold=CONFIDENCE_THRESHOLD)
        logging.info('✅ Predictions generated for backtesting.')

        trades_df, daily_portfolio_df = backtest_strategy(df_original_aligned, predictions)
        logging.info('✅ Backtesting completed.')
        
        df_viz_data = df_for_features_and_signals.loc[predictions.index]

        if not daily_portfolio_df.empty:
            visualize_trading_results(df_viz_data, trades_df, daily_portfolio_df, lower_thresh, upper_thresh)
            logging.info('✅ Visualization displayed.')
        else:
            logging.warning('No daily portfolio data generated for visualization (possibly no trades).')

        logging.info('🚀 Backtesting pipeline completed successfully!')
        # FIX: Return the correctly named variable
        return df_original_aligned, model, trades_df, daily_portfolio_df 
        
    except Exception as e:
        logging.critical(f'❌ Error executing backtesting pipeline: {str(e)}', exc_info=True)
        raise


def run_live_trade_loop(model):
    """
    Executes the live trading loop.
    Fetches live data, calculates indicators, makes predictions, and places trades.
    """
    global live_candles_history, current_position_live

    
    logging.info("Starting live trading loop...")
    binance_manager = BinanceManager()

    # Initial data fetch for indicators
    logging.info(f"Fetching initial {INITIAL_CANDLES_HISTORY} candles for {TRADE_SYMBOL} ({TRADE_INTERVAL})...")
    live_candles_history = binance_manager.get_latest_ohlcv_candles(
        symbol=TRADE_SYMBOL, interval=TRADE_INTERVAL, limit=INITIAL_CANDLES_HISTORY
    )
    if live_candles_history.empty:
        logging.critical("Failed to fetch initial historical data for live trading. Exiting.")
        return

    # Check initial balance
    quote_asset = TRADE_SYMBOL[len(TRADE_SYMBOL) // 2:] # e.g., USDT from BTCUSDT
    base_asset = TRADE_SYMBOL[:len(TRADE_SYMBOL) // 2] # e.g., BTC from BTCUSDT
    
    usdt_balance = binance_manager.get_account_balance(asset=quote_asset)
    crypto_balance = binance_manager.get_account_balance(asset=base_asset)
    
    logging.info(f"Initial live balances: {quote_asset}: {usdt_balance}, {base_asset}: {crypto_balance}")

    # Determine initial position (simple: if we have crypto, assume long)
    if crypto_balance > TRADE_QUANTITY: # If we have more than a trade quantity of crypto
        current_position_live = 1 # Assume we are currently in a long position
        logging.info(f"Detected existing crypto balance ({crypto_balance} {base_asset}). Assuming initial long position.")
    else:
        current_position_live = 0 # No position

    # Determine sleep duration based on interval (e.g., 1h interval -> check every hour)
    interval_minutes = binance_manager._interval_to_minutes(TRADE_INTERVAL)
    sleep_seconds = interval_minutes * 60 # Check every interval period
    
    sleep_seconds_with_buffer = sleep_seconds + 5 # 5 second buffer

    logging.info(f"Live trading for {TRADE_SYMBOL} at {TRADE_INTERVAL} interval. Checking every {sleep_seconds_with_buffer} seconds.")
    logging.info(f"Starting with position: {current_position_live} (0=None, 1=Long)")

    while True:
        try:
            latest_candle_df = binance_manager.get_latest_ohlcv_candles(
                symbol=TRADE_SYMBOL, interval=TRADE_INTERVAL, limit=1
            )
            if latest_candle_df.empty:
                logging.warning("No new candle data fetched. Skipping this iteration.")
                time.sleep(sleep_seconds_with_buffer)
                continue

            if not live_candles_history.empty and latest_candle_df.index[-1] <= live_candles_history.index[-1]:
                logging.debug("Latest candle is not new or already processed. Waiting for next.")
                time.sleep(sleep_seconds_with_buffer)
                continue
            
            live_candles_history = pd.concat([live_candles_history, latest_candle_df])
            live_candles_history = live_candles_history.iloc[-INITIAL_CANDLES_HISTORY*2:] 
            
            df_for_features_live = live_candles_history.copy() 
            df_for_features_live = calculate_technical_indicators(df_for_features_live)
            
            if df_for_features_live.empty:
                logging.warning("Not enough history to calculate indicators for live trading. Waiting for more data.")
                time.sleep(sleep_seconds_with_buffer)
                continue
            
            latest_data_point = df_for_features_live.iloc[[-1]] 

            normalized_latest_data = normalize_features(latest_data_point)
            
            X_live = normalized_latest_data[FEATURE_COLUMNS]
            live_prediction, live_confidence = make_predictions(model, X_live, CONFIDENCE_THRESHOLD)
            live_signal = live_prediction.iloc[0]
            
            logging.info(f"Live Signal for {TRADE_SYMBOL} on {latest_data_point.index[-1].strftime('%Y-%m-%d %H:%M')}: {live_signal} (Confidence: {live_confidence.iloc[0]:.2f})")

            current_usdt_balance = binance_manager.get_account_balance(asset=quote_asset)
            current_crypto_balance = binance_manager.get_account_balance(asset=base_asset)
            
            current_price = latest_data_point['close'].iloc[0]

            if live_signal == 1 and current_position_live == 0:
                cost_with_fee = TRADE_QUANTITY * current_price * (1 + TRANSACTION_FEE_PCT)
                if current_usdt_balance >= cost_with_fee:
                    order = binance_manager.place_market_order(TRADE_SYMBOL, TRADE_QUANTITY, SIDE_BUY)
                    if order:
                        current_position_live = 1
                        logging.info(f"✅ LIVE BUY ORDER placed: {TRADE_QUANTITY} {base_asset} @ {current_price}")
                        record_trade(TRADE_SYMBOL, "BUY", TRADE_QUANTITY, current_price)
                else:
                    logging.warning(f"BUY signal but insufficient {quote_asset} balance ({current_usdt_balance:.2f} needed {cost_with_fee:.2f}).")
            
            elif live_signal == -1 and current_position_live == 1:
                if current_crypto_balance >= TRADE_QUANTITY:
                    order = binance_manager.place_market_order(TRADE_SYMBOL, TRADE_QUANTITY, SIDE_SELL)
                    if order:
                        current_position_live = 0
                        logging.info(f"✅ LIVE SELL ORDER placed: {TRADE_QUANTITY} {base_asset} @ {current_price}")
                        record_trade(TRADE_SYMBOL, "SELL", TRADE_QUANTITY, current_price)
                else:
                    logging.warning(f"SELL signal but insufficient {base_asset} balance ({current_crypto_balance:.4f} needed {TRADE_QUANTITY}).")
            
            elif live_signal == 0 and current_position_live == 1:
                if current_crypto_balance >= TRADE_QUANTITY and binance_manager.get_account_balance(asset=quote_asset) > 0:
                    logging.info("Optional: HOLD signal while in position. Deciding whether to exit...")
                    order = binance_manager.place_market_order(TRADE_SYMBOL, TRADE_QUANTITY, SIDE_SELL)
                    if order:
                        current_position_live = 0
                        logging.info(f"✅ LIVE SELL (on HOLD) ORDER placed: {TRADE_QUANTITY} {base_asset} @ {current_price}")
                        record_trade(TRADE_SYMBOL, "SELL", TRADE_QUANTITY, current_price)
                else:
                    logging.debug("HOLD signal while in position, but not exiting or insufficient funds for sell.")

            time.sleep(sleep_seconds_with_buffer)

        except Exception as e:
            error_msg = f"An unexpected error occurred in live trading loop: {e}"
            logging.critical(error_msg, exc_info=True)
            send_email_notification("Trading Bot Live Loop Error", error_msg)
            send_telegram_notification(error_msg)
            time.sleep(60)


# Save records of Trades places
def record_trade(symbol, side, quantity, price):
    db = SessionLocal()
    trade = Trade(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        timestamp=datetime.utcnow()
    )
    db.add(trade)
    db.commit()
    db.close()


def send_email_notification(subject, message):
    try:
        host = os.getenv("EMAIL_HOST")
        port = int(os.getenv("EMAIL_PORT", 587))
        user = os.getenv("EMAIL_USER")
        password = os.getenv("EMAIL_PASS")
        to_addr = os.getenv("EMAIL_TO")
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to_addr
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to_addr], msg.as_string())
    except Exception as e:
        logging.error(f"Failed to send email notification: {e}")

def send_telegram_notification(message):
    try:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
        requests.post(url, data=data)
    except Exception as e:
        logging.error(f"Failed to send Telegram notification: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    send_email_notification("Test Subject", "Test Message")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    MODE = os.getenv("MODE", "live")  # Read MODE from .env, default to 'live'
    
    try:
        # First, ensure model is trained and saved, regardless of mode
        df_backtest_data, trained_model, _, _ = run_backtesting_pipeline(train_new_model=True) 
        
        if MODE == 'live':
            if trained_model:
                logging.info(f"Switching to LIVE trading mode for {TRADE_SYMBOL}...")
                run_live_trade_loop(trained_model)
            else:
                logging.critical("Cannot start live trading: no trained model available.")
        elif MODE == 'backtest':
            logging.info("Backtesting pipeline executed. No live trading initiated.")
        else:
            logging.error(f"Invalid MODE specified in main.py: {MODE}. Must be 'backtest' or 'live'.")

    except Exception as e:
        error_msg = f"Pipeline execution failed: {e}"
        logging.critical(error_msg, exc_info=True)
        send_email_notification("Trading Bot Critical Failure", error_msg)
        send_telegram_notification(error_msg)
        print("Pipeline execution failed. Check logs for details.")
    finally:
        logging.info("Application finished.")


