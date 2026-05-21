from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandas import DataFrame

from app.scrapers.market_data_scraper import (
    MarketDataScraper,
    MarketDataScrapeRequest,
    normalize_ohlcv_frame,
    scrape_latest_ohlcv,
    validate_scraped_data,
)


@dataclass
class FakeMarketDataSource:
    data: DataFrame
    calls: list[tuple[str, str, int]]

    def get_latest_ohlcv(self, symbol: str, interval: str, limit: int) -> DataFrame:
        self.calls.append((symbol, interval, limit))
        return self.data


@pytest.fixture
def raw_ohlcv_frame() -> DataFrame:
    return pd.DataFrame(
        {
            "open_time": [
                "2026-05-18 08:00:00",
                "2026-05-18 04:00:00",
                "2026-05-18 04:00:00",
            ],
            "open": ["105.0", "100.0", "101.0"],
            "high": ["110.0", "106.0", "107.0"],
            "low": ["99.0", "95.0", "96.0"],
            "close": ["108.0", "102.0", "103.0"],
            "volume": ["12.5", "10.0", "11.0"],
            "number_of_trades": [20, 10, 11],
        }
    )


def test_scrape_request_normalizes_symbol_and_validates_interval() -> None:
    request = MarketDataScrapeRequest(symbol="btc/usdt", interval="4h", limit=10)

    assert request.symbol == "BTCUSDT"
    assert request.interval == "4h"
    assert request.limit == 10


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"limit": 0}, "limit must be greater than 0"),
        ({"min_rows": -1}, "min_rows must be greater than or equal to 0"),
        ({"interval": "bad"}, "interval must be one of"),
    ],
)
def test_scrape_request_rejects_invalid_values(
    kwargs: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        MarketDataScrapeRequest(**kwargs)


def test_normalize_ohlcv_frame_sorts_deduplicates_and_coerces_numeric_values(
    raw_ohlcv_frame: DataFrame,
) -> None:
    normalized = normalize_ohlcv_frame(raw_ohlcv_frame)

    assert list(normalized.columns[:5]) == ["open", "high", "low", "close", "volume"]
    assert normalized.index.name == "open_time"
    assert normalized.index.is_monotonic_increasing
    assert len(normalized) == 2
    assert normalized.iloc[0]["open"] == 101.0
    assert normalized.iloc[0]["number_of_trades"] == 11


def test_normalize_ohlcv_frame_accepts_timestamp_column() -> None:
    raw_data = pd.DataFrame(
        {
            "timestamp": ["2026-05-18T00:00:00Z"],
            "open": [100],
            "high": [105],
            "low": [95],
            "close": [102],
            "volume": [10],
        }
    )

    normalized = normalize_ohlcv_frame(raw_data)

    assert isinstance(normalized.index, pd.DatetimeIndex)
    assert normalized.index.name == "timestamp"


def test_normalize_ohlcv_frame_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing required OHLCV columns"):
        normalize_ohlcv_frame(pd.DataFrame({"open_time": ["2026-05-18"], "close": [100]}))


def test_validate_scraped_data_rejects_too_few_rows(raw_ohlcv_frame: DataFrame) -> None:
    normalized = normalize_ohlcv_frame(raw_ohlcv_frame)

    with pytest.raises(ValueError, match="Expected at least 3 rows"):
        validate_scraped_data(normalized, min_rows=3)


def test_fetch_ohlcv_returns_normalized_result(raw_ohlcv_frame: DataFrame) -> None:
    source = FakeMarketDataSource(data=raw_ohlcv_frame, calls=[])
    request = MarketDataScrapeRequest(symbol="eth-usdt", interval="1h", limit=3, min_rows=2)

    result = MarketDataScraper(source=source).fetch_ohlcv(request)

    assert source.calls == [("ETHUSDT", "1h", 3)]
    assert result.symbol == "ETHUSDT"
    assert result.interval == "1h"
    assert result.rows == 2
    assert result.source == "FakeMarketDataSource"
    assert result.metadata == {"requested_limit": 3}


def test_scrape_latest_ohlcv_uses_injected_source(raw_ohlcv_frame: DataFrame) -> None:
    source = FakeMarketDataSource(data=raw_ohlcv_frame, calls=[])

    result = scrape_latest_ohlcv(
        MarketDataScrapeRequest(symbol="btcusdt", interval="4h", limit=3),
        source=source,
    )

    assert result.symbol == "BTCUSDT"
    assert source.calls == [("BTCUSDT", "4h", 3)]


def test_fetch_ohlcv_rejects_empty_data_when_min_rows_required() -> None:
    source = FakeMarketDataSource(data=pd.DataFrame(), calls=[])

    with pytest.raises(ValueError, match="Expected at least 1 rows"):
        MarketDataScraper(source=source).fetch_ohlcv(MarketDataScrapeRequest(limit=5))


def test_save_to_csv_writes_timestamped_market_data(
    raw_ohlcv_frame: DataFrame,
    tmp_path: Path,
) -> None:
    source = FakeMarketDataSource(data=raw_ohlcv_frame, calls=[])
    scraper = MarketDataScraper(source=source)
    result = scraper.fetch_ohlcv(MarketDataScrapeRequest(limit=3, min_rows=2))

    output_path = scraper.save_to_csv(result, tmp_path / "market" / "ohlcv.csv")

    saved = pd.read_csv(output_path)
    assert output_path.exists()
    assert saved.columns[0] == "timestamp"
    assert len(saved) == 2
