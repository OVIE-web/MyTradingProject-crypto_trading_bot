"""Market data scraping utilities for OHLCV ingestion."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
from pandas import DataFrame

from app.core.config import INITIAL_CANDLES_HISTORY, TRADE_INTERVAL, TRADE_SYMBOL
from app.services.binance_service import BinanceManager
from app.tools.technical_indicators import REQUIRED_OHLCV_COLUMNS
from app.utils.validators import normalize_symbol, validate_binance_interval

LOG = logging.getLogger(__name__)
OHLCV_COLUMN_ORDER = ["open", "high", "low", "close", "volume"]


class MarketDataSource(Protocol):
    """Protocol for exchange/data clients that can provide OHLCV data."""

    def get_latest_ohlcv(self, symbol: str, interval: str, limit: int) -> DataFrame:
        """Return latest OHLCV candles for a symbol and interval."""
        ...


@dataclass(frozen=True, slots=True)
class MarketDataScrapeRequest:
    """Input for a market data scrape operation."""

    symbol: str = TRADE_SYMBOL
    interval: str = TRADE_INTERVAL
    limit: int = INITIAL_CANDLES_HISTORY
    min_rows: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        object.__setattr__(self, "interval", validate_binance_interval(self.interval))

        if self.limit <= 0:
            raise ValueError("limit must be greater than 0.")
        if self.min_rows < 0:
            raise ValueError("min_rows must be greater than or equal to 0.")


@dataclass(frozen=True, slots=True)
class MarketDataScrapeResult:
    """Result returned after a market data scrape operation."""

    symbol: str
    interval: str
    data: DataFrame
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    source: str = "binance"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        """Return the number of normalized rows."""
        return len(self.data)

    @property
    def is_empty(self) -> bool:
        """Return True when the scrape produced no rows."""
        return self.data.empty


class MarketDataScraper:
    """Fetch and normalize OHLCV market data from an injectable source."""

    def __init__(self, source: MarketDataSource | None = None) -> None:
        self._source = source

    @property
    def source(self) -> MarketDataSource:
        """Lazily create the default Binance source."""
        if self._source is None:
            self._source = BinanceManager()
        return self._source

    def fetch_ohlcv(
        self,
        request: MarketDataScrapeRequest | None = None,
    ) -> MarketDataScrapeResult:
        """Fetch, normalize, and validate latest OHLCV candles."""
        scrape_request = request or MarketDataScrapeRequest()

        LOG.info(
            "Fetching market data. symbol=%s interval=%s limit=%s",
            scrape_request.symbol,
            scrape_request.interval,
            scrape_request.limit,
        )

        source = self.source
        raw_data = source.get_latest_ohlcv(
            scrape_request.symbol,
            scrape_request.interval,
            scrape_request.limit,
        )
        normalized = normalize_ohlcv_frame(raw_data)
        validate_scraped_data(normalized, min_rows=scrape_request.min_rows)

        return MarketDataScrapeResult(
            symbol=scrape_request.symbol,
            interval=scrape_request.interval,
            data=normalized,
            source=type(source).__name__,
            metadata={"requested_limit": scrape_request.limit},
        )

    def save_to_csv(
        self,
        result: MarketDataScrapeResult,
        output_path: str | Path,
    ) -> Path:
        """Persist a scrape result as a CSV file with an explicit timestamp column."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        export_df = result.data.copy()
        timestamp_column = export_df.index.name or "open_time"
        export_df.insert(0, "timestamp", export_df.index)
        export_df.to_csv(path, index=False)

        LOG.info(
            "Saved market data CSV. path=%s rows=%s timestamp_source=%s",
            path,
            len(export_df),
            timestamp_column,
        )
        return path


def scrape_latest_ohlcv(
    request: MarketDataScrapeRequest | None = None,
    *,
    source: MarketDataSource | None = None,
) -> MarketDataScrapeResult:
    """Convenience function for one-off OHLCV scraping."""
    return MarketDataScraper(source=source).fetch_ohlcv(request)


def normalize_ohlcv_frame(data: DataFrame) -> DataFrame:
    """Return a normalized OHLCV DataFrame with numeric columns and sorted timestamps."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("market data must be a pandas DataFrame.")

    df = data.copy()
    if df.empty:
        return _empty_ohlcv_frame()

    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce")
        df = df.set_index("open_time")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    missing = sorted(REQUIRED_OHLCV_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    required_columns = OHLCV_COLUMN_ORDER
    df = df.loc[df.index.notna()].copy()
    df[required_columns] = df[required_columns].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=required_columns)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df.index.name = df.index.name or "open_time"

    extra_columns = [column for column in df.columns if column not in required_columns]
    return cast(DataFrame, df.loc[:, required_columns + extra_columns].copy())


def validate_scraped_data(data: DataFrame, *, min_rows: int = 1) -> None:
    """Validate normalized scraped data before downstream processing."""
    if min_rows < 0:
        raise ValueError("min_rows must be greater than or equal to 0.")

    missing = sorted(REQUIRED_OHLCV_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    if len(data) < min_rows:
        raise ValueError(f"Expected at least {min_rows} rows of market data, got {len(data)}.")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("market data index must be a DatetimeIndex.")


def _empty_ohlcv_frame() -> DataFrame:
    df = pd.DataFrame(columns=OHLCV_COLUMN_ORDER)
    df.index = pd.DatetimeIndex([], name="open_time")
    return df
