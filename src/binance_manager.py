# src/binance_manager.py
from __future__ import annotations

import logging
import os
import time
from typing import Any, cast

import numpy as np
import pandas as pd
import requests
from binance.client import Client
from binance.enums import ORDER_TYPE_MARKET
from binance.exceptions import BinanceAPIException

from src.config import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BinanceManager:
    """
    Manages Binance API interactions.

    Supports:
    - Live trading
    - Testnet trading
    - Offline (mock) mode for CI/testing
    """

    def __init__(self) -> None:
        api_key: str = os.getenv("BINANCE_API_KEY", "").strip()
        api_secret: str = os.getenv("BINANCE_API_SECRET", "").strip()
        use_testnet: bool = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

        if not api_key or not api_secret:
            raise ValueError(
                "Missing Binance API credentials. Set BINANCE_API_KEY and BINANCE_API_SECRET."
            )

        api_url = (
            "https://testnet.binance.vision/api" if use_testnet else "https://api.binance.com/api"
        )

        self.client: Client = Client(api_key, api_secret)
        self.client.API_URL = api_url
        self.offline_mode: bool = False

        logger.info("Using Binance API URL: %s", api_url)

        if not self._safe_ping():
            logger.warning("Binance unreachable — switching to OFFLINE mode.")
            self.offline_mode = True

        self.account_info: dict[str, Any] = {}
        if not self.offline_mode:
            try:
                self.account_info = self.client.get_account()
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not fetch account info: %s", exc)

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------
    def _safe_ping(self, retries: int = 3, delay: int = 1) -> bool:
        for attempt in range(1, retries + 1):
            try:
                self.client.ping()
                return True
            except (BinanceAPIException, requests.RequestException) as exc:
                logger.warning("Ping attempt %d/%d failed: %s", attempt, retries, exc)
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
        if self.offline_mode:
            return self._mock_ohlcv(symbol, interval, limit)

        try:
            lookback = f"{limit * self._interval_to_minutes(interval)} minutes ago UTC"
            klines = self.client.get_historical_klines(symbol, interval, lookback)

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

            return df

        except Exception as exc:
            logger.error("Failed to fetch OHLCV: %s", exc, exc_info=True)
            return pd.DataFrame()

    def _interval_to_minutes(self, interval: str) -> int:
        if interval.endswith("m"):
            return int(interval[:-1])
        if interval.endswith("h"):
            return int(interval[:-1]) * 60
        if interval.endswith("d"):
            return int(interval[:-1]) * 1440
        return 1

    def _mock_ohlcv(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        now = pd.Timestamp.utcnow()
        index = pd.date_range(end=now, periods=limit, freq="4H")

        df = pd.DataFrame(
            {
                "open": np.random.uniform(68000, 69000, size=limit),
                "high": np.random.uniform(69000, 69500, size=limit),
                "low": np.random.uniform(67500, 68500, size=limit),
                "close": np.random.uniform(68000, 69000, size=limit),
                "volume": np.random.uniform(10, 100, size=limit),
            },
            index=index,
        )

        logger.info("Generated mock OHLCV for %s (%s)", symbol, interval)
        return df

    # ------------------------------------------------------------------
    # Account & Orders
    # ------------------------------------------------------------------
    def get_account_balance(self, asset: str = "USDT") -> float:
        if self.offline_mode:
            return 10_000.0
        try:
            balance_raw = self.client.get_asset_balance(asset=asset)
            if balance_raw is None:
                return 0.0

            balance_raw = cast(dict[str, Any], balance_raw)
            return float(balance_raw.get("free", 0.0))
        except Exception as exc:
            logger.error("Failed to fetch  Binance account balance: %s", exc, exc_info=True)
            return 0.0

    def place_market_order(self, symbol: str, quantity: float, side: str) -> dict[str, Any] | None:
        if self.offline_mode:
            return {
                "symbol": symbol,
                "side": side,
                "status": "FILLED",
                "executedQty": quantity,
            }

        if quantity <= 0:
            return None

        try:
            order_raw = self.client.create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
            )
            return cast(dict[str, Any], order_raw)
        except Exception as exc:
            logger.error("Order failed: %s", exc, exc_info=True)
            return None

    def get_server_time(self) -> int | None:
        try:
            response = self.client.get_server_time()
            return int(cast(dict[str, Any], response)["serverTime"])
        except Exception:
            return None


# -------------------- End of BinanceManager --------------------
