"""Binance service for fetching market data and placing orders"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import numpy as np
import pandas as pd
import requests
from binance.client import Client
from binance.enums import ORDER_TYPE_MARKET
from binance.exceptions import BinanceAPIException

from app.core.config import config

LOG = logging.getLogger(__name__)

BINANCE_API_URL = "https://api.binance.com/api"
BINANCE_TESTNET_API_URL = "https://testnet.binance.vision/api"


class BinanceManager:
    """
    Manages Binance API interactions.

    Supports:
    - Binance live trading
    - Binance testnet trading
    - Offline mock mode for CI/testing/local development
    """

    def __init__(self) -> None:
        self.api_key = (config.binance_api_key or "").strip()
        self.api_secret = (config.binance_api_secret or "").strip()
        self.use_testnet = bool(config.binance_testnet)

        self.api_url = BINANCE_TESTNET_API_URL if self.use_testnet else BINANCE_API_URL
        self.offline_mode = False
        self.account_info: dict[str, Any] = {}

        if not self.api_key or not self.api_secret:
            LOG.warning("Missing Binance API credentials. Switching to OFFLINE mode.")
            self.offline_mode = True
            self.client: Client | None = None
            return

        self.client = Client(self.api_key, self.api_secret)
        self.client.API_URL = self.api_url

        LOG.info("Using Binance API URL: %s", self.api_url)

        if not self._safe_ping():
            LOG.warning("Binance unreachable. Switching to OFFLINE mode.")
            self.offline_mode = True
            return

        try:
            self.account_info = self.client.get_account()
            LOG.info("Binance account loaded successfully. Testnet=%s", self.use_testnet)
        except Exception as exc:
            LOG.warning("Could not fetch Binance account info: %s", exc)

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------
    def _safe_ping(self, retries: int = 3, delay: int = 1) -> bool:
        if self.client is None:
            return False

        for attempt in range(1, retries + 1):
            try:
                self.client.ping()
                return True
            except (BinanceAPIException, requests.RequestException, Exception) as exc:
                LOG.warning("Binance ping attempt %d/%d failed: %s", attempt, retries, exc)
                time.sleep(delay * attempt)

        return False

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def get_latest_ohlcv(
        self,
        symbol: str = config.trade_symbol,
        interval: str = config.trade_interval,
        limit: int = config.initial_candles_history,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance or generate mock data if offline.
        """
        if self.offline_mode or self.client is None:
            return self._mock_ohlcv(symbol=symbol, interval=interval, limit=limit)

        try:
            lookback_minutes = limit * self._interval_to_minutes(interval)
            lookback = f"{lookback_minutes} minutes ago UTC"

            klines = self.client.get_historical_klines(
                symbol,
                interval,
                lookback,
            )

            if not klines:
                LOG.warning("No OHLCV candles returned for %s %s", symbol, interval)
                return pd.DataFrame()

            df = pd.DataFrame(
                klines,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )

            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

            numeric_cols = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
            ]

            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            df.set_index("open_time", inplace=True)
            df.sort_index(inplace=True)

            LOG.info(
                "Fetched %s candles for %s on interval %s.",
                len(df),
                symbol,
                interval,
            )

            return df

        except Exception as exc:
            LOG.error("Failed to fetch OHLCV for %s: %s", symbol, exc, exc_info=True)
            return pd.DataFrame()

    def get_latest_ohlcv_candles(
        self,
        symbol: str = config.trade_symbol,
        interval: str = config.trade_interval,
        limit: int = config.initial_candles_history,
    ) -> pd.DataFrame:
        """
        Backward-compatible alias for older code/tests.
        """
        return self.get_latest_ohlcv(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )

    def _interval_to_minutes(self, interval: str) -> int:
        if interval.endswith("m"):
            return int(interval[:-1])
        if interval.endswith("h"):
            return int(interval[:-1]) * 60
        if interval.endswith("d"):
            return int(interval[:-1]) * 24 * 60
        if interval.endswith("w"):
            return int(interval[:-1]) * 7 * 24 * 60

        LOG.warning("Unknown interval '%s'. Falling back to 1 minute.", interval)
        return 1

    def _interval_to_pandas_freq(self, interval: str) -> str:
        if interval.endswith("m"):
            return f"{int(interval[:-1])}min"
        if interval.endswith("h"):
            return f"{int(interval[:-1])}h"
        if interval.endswith("d"):
            return f"{int(interval[:-1])}D"
        if interval.endswith("w"):
            return f"{int(interval[:-1])}W"

        return "1min"

    def _mock_ohlcv(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """
        Generate mock OHLCV data for tests/local development when Binance is unavailable.
        """
        now = pd.Timestamp.utcnow()
        freq = self._interval_to_pandas_freq(interval)

        index = pd.date_range(end=now, periods=limit, freq=freq)

        close = np.random.uniform(68000, 69000, size=limit)
        open_ = close + np.random.uniform(-100, 100, size=limit)
        high = np.maximum(open_, close) + np.random.uniform(10, 250, size=limit)
        low = np.minimum(open_, close) - np.random.uniform(10, 250, size=limit)

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.uniform(10, 100, size=limit),
            },
            index=index,
        )

        df.index.name = "open_time"

        LOG.info(
            "Generated mock OHLCV for %s on interval %s. Rows=%s",
            symbol,
            interval,
            len(df),
        )

        return df

    # ------------------------------------------------------------------
    # Account & Orders
    # ------------------------------------------------------------------
    def get_account_balance(self, asset: str = "USDT") -> float:
        if self.offline_mode or self.client is None:
            return 10_000.0

        try:
            balance_raw = self.client.get_asset_balance(asset=asset)
            if balance_raw is None:
                return 0.0

            balance_raw = cast(dict[str, Any], balance_raw)
            return float(balance_raw.get("free", 0.0))

        except Exception as exc:
            LOG.error("Failed to fetch Binance account balance: %s", exc, exc_info=True)
            return 0.0

    def place_market_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
    ) -> dict[str, Any] | None:
        """
        Place a market order.

        Args:
            symbol: Trading pair, e.g. BTCUSDT.
            quantity: Quantity of base asset.
            side: BUY or SELL.

        Returns:
            Binance order response or mock order response.
        """
        normalized_side = side.upper()

        if normalized_side not in {"BUY", "SELL"}:
            LOG.error("Invalid order side: %s", side)
            return None

        if quantity <= 0:
            LOG.error("Invalid order quantity: %s", quantity)
            return None

        if self.offline_mode or self.client is None:
            LOG.info(
                "OFFLINE mode: simulated %s market order for %s quantity=%s",
                normalized_side,
                symbol,
                quantity,
            )
            return {
                "symbol": symbol,
                "side": normalized_side,
                "status": "FILLED",
                "executedQty": str(quantity),
                "price": "0",
            }

        try:
            order_raw = self.client.create_order(
                symbol=symbol,
                side=normalized_side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
            )

            return cast(dict[str, Any], order_raw)

        except Exception as exc:
            LOG.error(
                "Market order failed. symbol=%s side=%s quantity=%s error=%s",
                symbol,
                normalized_side,
                quantity,
                exc,
                exc_info=True,
            )
            return None

    def get_server_time(self) -> int | None:
        if self.offline_mode or self.client is None:
            return None

        try:
            response = self.client.get_server_time()
            return int(cast(dict[str, Any], response)["serverTime"])
        except Exception as exc:
            LOG.warning("Could not fetch Binance server time: %s", exc)
            return None


# -------------------- End of BinanceManager --------------------
