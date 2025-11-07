import logging
import os
import pandas as pd
import time
import requests
import numpy as np
from binance.client import Client
from binance.enums import *  # For order types etc.
from binance.exceptions import BinanceAPIException
import backoff

from src.config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET,
    TRADE_SYMBOL, TRADE_INTERVAL, INITIAL_CANDLES_HISTORY
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_binance_keys():
    """Load Binance credentials dynamically (supports test patching)."""
    return (
        os.getenv("BINANCE_API_KEY"),
        os.getenv("BINANCE_API_SECRET"),
        os.getenv("BINANCE_TESTNET", "False").lower() == "true"
    )

@backoff.on_exception(backoff.expo, Exception, max_tries=5, jitter=backoff.full_jitter)
class BinanceManager:
    def __init__(self):
        global BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET
        BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET = get_binance_keys()

        if not BINANCE_API_KEY or not BINANCE_API_SECRET:
            raise ValueError("🚨 API credentials missing. Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")

        api_url = (
            "https://testnet.binance.vision/api"
            if BINANCE_TESTNET else
            "https://api.binance.com/api"
        )
        logger.info(f"Using Binance API URL: {api_url}")

        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        self.client.API_URL = api_url
        self.offline_mode = False

        # ✅ Safe connection initialization
        if not self._safe_ping():
            logger.critical("⚠️ Binance API is unreachable — switching to offline mode.")
            self.offline_mode = True
        else:
            logger.info(f"✅ Connected to Binance API (Testnet={BINANCE_TESTNET}) successfully.")
            try:
                self.account_info = self.client.get_account()
                logger.info("📊 Account information loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load account info: {e}")
                self.account_info = {}

    # ----------------------------------------------------------------------
    # SAFE PING — prevents crash
    # ----------------------------------------------------------------------
    def _safe_ping(self, retries: int = 3, delay: int = 2) -> bool:
        """Try pinging Binance API safely with retry & exponential backoff."""
        for attempt in range(1, retries + 1):
            try:
                self.client.ping()
                return True
            except BinanceAPIException as e:
                logger.warning(f"Binance API ping failed (attempt {attempt}/{retries}): {e}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error during Binance ping (attempt {attempt}/{retries}): {e}")
            except Exception as e:
                logger.warning(f"Unexpected ping error (attempt {attempt}/{retries}): {e}")
            time.sleep(delay * attempt)
        return False

    # ----------------------------------------------------------------------
    # Data Retrieval
    # ----------------------------------------------------------------------
    def get_latest_ohlcv(self, symbol=TRADE_SYMBOL, interval=TRADE_INTERVAL, limit=INITIAL_CANDLES_HISTORY):
        """Fetch latest OHLCV candles. Returns mock data if offline."""
        if self.offline_mode:
            logger.warning("📉 Binance offline — returning mock OHLCV data.")
            return self._mock_ohlcv(symbol, interval, limit)

        try:
            klines = self.client.get_historical_klines(
                symbol, interval,
                f"{int(limit) * self._interval_to_minutes(interval)} minutes ago UTC"
            )

            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                            'quote_asset_volume', 'number_of_trades',
                            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

            df.set_index('open_time', inplace=True)
            df.sort_index(inplace=True)

            logger.info(f"Fetched {len(df)} {interval} candles for {symbol}. Latest close: {df['close'].iloc[-1]}")
            return df
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def _mock_ohlcv(self, symbol, interval, limit):
        """Generate mock OHLCV data for offline mode."""
        now = pd.Timestamp.utcnow()
        idx = pd.date_range(end=now, periods=limit, freq="4H")
        df = pd.DataFrame({
            "open": np.random.uniform(68000, 69000, size=limit),
            "high": np.random.uniform(69000, 69500, size=limit),
            "low": np.random.uniform(67500, 68500, size=limit),
            "close": np.random.uniform(68000, 69000, size=limit),
            "volume": np.random.uniform(10, 100, size=limit),
        }, index=idx)
        logger.info(f"🔄 Generated mock OHLCV data for {symbol} ({interval}).")
        return df

    def _interval_to_minutes(self, interval):
        """Convert Binance interval string (e.g. '1h') to minutes."""
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval.endswith('d'):
            return int(interval[:-1]) * 24 * 60
        return 1

    # ----------------------------------------------------------------------
    # Account and Orders
    # ----------------------------------------------------------------------
    def get_account_balance(self, asset='USDT'):
        """Return free balance of a given asset."""
        if self.offline_mode:
            logger.warning(f"Offline mode active — returning mock balance for {asset}.")
            return 10000.0  # Mock balance

        try:
            balance = self.client.get_asset_balance(asset=asset)
            free_balance = float(balance['free'])
            logger.info(f"Current {asset} balance: {free_balance}")
            return free_balance
        except Exception as e:
            logger.error(f"Error getting {asset} balance: {e}", exc_info=True)
            return 0.0

    def place_market_order(self, symbol, quantity, side):
        """Place a market order safely."""
        if self.offline_mode:
            logger.warning(f"Offline mode: Simulating {side} order for {symbol} (Qty: {quantity}).")
            return {
                "symbol": symbol, "side": side, "status": "FILLED",
                "price": "mock", "executedQty": quantity
            }

        try:
            if not isinstance(quantity, (int, float)) or quantity <= 0:
                logger.warning(f"Invalid quantity for market order: {quantity} — order not placed.")
                return None

            info = self.client.get_symbol_info(symbol)
            if not info:
                logger.error(f"Could not get symbol info for {symbol}")
                return None

            quantity_precision = 0
            for f in info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    quantity_precision = len(str(float(f['stepSize']))[str(float(f['stepSize'])).find('.') + 1:])
                    break

            adjusted_quantity = round(quantity, quantity_precision)
            if adjusted_quantity <= 0:
                logger.warning(f"Adjusted quantity for {symbol} is zero or negative ({adjusted_quantity}), skipping order.")
                return None

            logger.info(f"Attempting to place {side} market order for {adjusted_quantity} {symbol}...")
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=adjusted_quantity
            )
            logger.info(f"Market {side} order placed successfully: {order}")
            return order
        except Exception as e:
            logger.error(f"Error placing market {side} order for {symbol} (Qty: {quantity}): {e}", exc_info=True)
            return None

    def get_server_time(self):
        """Get Binance server time safely."""
        try:
            server_time = self.client.get_server_time()
            return server_time['serverTime']
        except Exception as e:
            logger.warning(f"Could not get Binance server time: {e}")
            return None
