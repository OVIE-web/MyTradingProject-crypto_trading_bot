# src/binance_manager.py

import logging
import os
import pandas as pd
import numpy as np
import time
import requests
from binance.client import Client
from binance.enums import ORDER_TYPE_MARKET
from binance.exceptions import BinanceAPIException

from src.config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET,
    TRADE_SYMBOL, TRADE_INTERVAL, INITIAL_CANDLES_HISTORY
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_binance_keys():
    """Load Binance credentials dynamically (supports monkeypatching for tests)."""
    api_key = os.getenv("BINANCE_API_KEY") or BINANCE_API_KEY
    api_secret = os.getenv("BINANCE_API_SECRET") or BINANCE_API_SECRET
    use_testnet = (os.getenv("BINANCE_TESTNET") or str(BINANCE_TESTNET)).lower() == "true"
    return api_key, api_secret, use_testnet


class BinanceManager:
    """Manages Binance API interactions (mocked or real)."""

    def __init__(self):
        # ✅ Always read fresh env vars directly
        api_key = os.getenv("BINANCE_API_KEY", "").strip()
        api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
        use_testnet = os.getenv("BINANCE_TESTNET", "False").lower() == "true"

        # ✅ Strong check (fixes test_binance_manager_init_no_api_keys)
        if not api_key or not api_secret:
            raise ValueError("API credentials missing. Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")

        api_url = (
            "https://testnet.binance.vision/api"
            if use_testnet else
            "https://api.binance.com/api"
        )
        logger.info(f"Using Binance API URL: {api_url}")

        # Safe client creation
        self.client = Client(api_key, api_secret)
        self.client.API_URL = api_url
        self.offline_mode = False

        if not self._safe_ping():
            logger.warning("⚠️ Binance API unreachable, switching to offline mode.")
            self.offline_mode = True

        if not self.offline_mode:
            try:
                self.account_info = self.client.get_account()
                logger.info("📊 Account information loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load account info: {e}")
                self.account_info = {}


    # ----------------------------------------------------------------------
    # Safe ping with retry
    # ----------------------------------------------------------------------
    def _safe_ping(self, retries=3, delay=1) -> bool:
        for attempt in range(1, retries + 1):
            try:
                self.client.ping()
                return True
            except (BinanceAPIException, requests.exceptions.RequestException) as e:
                logger.warning(f"Ping attempt {attempt}/{retries} failed: {e}")
                time.sleep(delay * attempt)
        return False

    # ----------------------------------------------------------------------
    # Data Retrieval
    # ----------------------------------------------------------------------
    def get_latest_ohlcv_candles(self, symbol=TRADE_SYMBOL, interval=TRADE_INTERVAL, limit=INITIAL_CANDLES_HISTORY):
        """Fetch OHLCV candles (mocked if offline)."""
        if self.offline_mode:
            return self._mock_ohlcv(symbol, interval, limit)

        try:
            klines = self.client.get_historical_klines(
                symbol, interval,
                f"{int(limit) * self._interval_to_minutes(interval)} minutes ago UTC"
            )

            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

            numeric_cols = [
                "open", "high", "low", "close", "volume",
                "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
            ]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            df.set_index("open_time", inplace=True)
            df.sort_index(inplace=True)

            logger.info(f"Fetched {len(df)} candles for {symbol} ({interval}).")
            return df

        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def _interval_to_minutes(self, interval):
        """Convert Binance interval strings like '4h' or '1d' to minutes."""
        if interval.endswith("m"):
            return int(interval[:-1])
        elif interval.endswith("h"):
            return int(interval[:-1]) * 60
        elif interval.endswith("d"):
            return int(interval[:-1]) * 24 * 60
        return 1

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
        logger.info(f"🔄 Generated mock OHLCV data for {symbol} ({interval}) in offline mode.")
        return df

    # ----------------------------------------------------------------------
    # Account & Orders
    # ----------------------------------------------------------------------
    def get_account_balance(self, asset="USDT"):
        if self.offline_mode:
            logger.info(f"Offline mode: returning mock balance for {asset}.")
            return 10000.0

        try:
            balance = self.client.get_asset_balance(asset=asset)
            return float(balance["free"])
        except Exception as e:
            logger.error(f"Error retrieving balance for {asset}: {e}", exc_info=True)
            return 0.0

    def place_market_order(self, symbol, quantity, side):
        """Place market order safely."""
        if self.offline_mode:
            logger.info(f"Offline mode: Simulating {side} order for {symbol} (Qty: {quantity}).")
            return {"symbol": symbol, "side": side, "status": "FILLED", "executedQty": quantity}

        try:
            if not isinstance(quantity, (int, float)) or quantity <= 0:
                logger.warning(f"Invalid order quantity: {quantity}")
                return None

            info = self.client.get_symbol_info(symbol)
            if not info:
                logger.error(f"No symbol info found for {symbol}")
                return None

            step_size = None
            for f in info["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    step_size = f.get("stepSize")
                    break

            # ✅ Fixed precision calculation (handles 1e-06 safely)
            if step_size:
                step_str = f"{float(step_size):e}"
                precision = abs(int(step_str.split("e")[-1]))
            else:
                precision = 6

            adjusted_qty = round(quantity, precision)
            if adjusted_qty <= 0:
                logger.warning(f"Adjusted quantity invalid ({adjusted_qty})")
                return None

            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=adjusted_qty
            )
            logger.info(f"{side} order placed successfully: {order}")
            return order
        except Exception as e:
            logger.error(f"Error placing {side} order: {e}", exc_info=True)
            return None
    
    def get_server_time(self):
        try:
            return self.client.get_server_time()["serverTime"]
        except Exception as e:
            logger.warning(f"Could not get server time: {e}")
            return None
