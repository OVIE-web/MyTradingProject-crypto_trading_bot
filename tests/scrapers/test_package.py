from __future__ import annotations

import app.scrapers as scrapers
from app.scrapers.market_data_scraper import (
    MarketDataScraper,
    MarketDataScrapeRequest,
    MarketDataScrapeResult,
    MarketDataSource,
    normalize_ohlcv_frame,
    scrape_latest_ohlcv,
    validate_scraped_data,
)


def test_scrapers_package_exports_public_api() -> None:
    assert scrapers.__all__ == [
        "MarketDataScrapeRequest",
        "MarketDataScrapeResult",
        "MarketDataScraper",
        "MarketDataSource",
        "normalize_ohlcv_frame",
        "scrape_latest_ohlcv",
        "validate_scraped_data",
    ]


def test_scrapers_package_exports_match_source_module() -> None:
    assert scrapers.MarketDataScrapeRequest is MarketDataScrapeRequest
    assert scrapers.MarketDataScrapeResult is MarketDataScrapeResult
    assert scrapers.MarketDataScraper is MarketDataScraper
    assert scrapers.MarketDataSource is MarketDataSource
    assert scrapers.normalize_ohlcv_frame is normalize_ohlcv_frame
    assert scrapers.scrape_latest_ohlcv is scrape_latest_ohlcv
    assert scrapers.validate_scraped_data is validate_scraped_data
