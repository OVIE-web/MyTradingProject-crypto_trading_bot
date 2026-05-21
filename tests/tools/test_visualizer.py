from __future__ import annotations

from unittest.mock import patch

import pandas as pd
from plotly import graph_objects as go

from app.tools.visualizer import visualize_trading_results


def make_indicator_frame(periods: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    close = pd.Series(range(102, 102 + periods), index=dates)
    return pd.DataFrame(
        {
            "open": close - 2,
            "high": close + 3,
            "low": close - 7,
            "close": close,
            "volume": range(1_000, 1_000 + periods),
            "bb_upper": close + 10,
            "bb_lower": close - 10,
            "bb_mid": close,
            "sma_20": close - 1,
            "sma_50": close - 3,
            "rsi": [50 + i for i in range(periods)],
            "atr": [2 + (i * 0.1) for i in range(periods)],
        },
        index=dates,
    )


def make_trades_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=2, freq="D")
    return pd.DataFrame(
        {
            "type": ["buy", "sell"],
            "price": [103.0, 107.0],
            "shares": [1.0, 1.0],
            "fee": [0.10, 0.10],
        },
        index=dates,
    )


def make_portfolio_frame(periods: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {"total_value": [10_000 + (i * 100) for i in range(periods)]},
        index=dates,
    )


def test_visualize_empty_dataframe_returns_none() -> None:
    result = visualize_trading_results(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    assert result is None


def test_visualize_basic_structure_with_trades_and_portfolio() -> None:
    with patch.object(go.Figure, "show") as show:
        result = visualize_trading_results(
            make_indicator_frame(),
            make_trades_frame(),
            make_portfolio_frame(),
        )

    assert isinstance(result, go.Figure)
    show.assert_called_once_with()

    trace_names = {trace.name for trace in result.data}
    assert {
        "OHLC",
        "BB Upper",
        "BB Lower",
        "BB Mid",
        "SMA 20",
        "SMA 50",
        "Buy Signals",
        "Sell Signals",
        "Volume",
        "RSI",
        "ATR",
        "Account Value",
    }.issubset(trace_names)
    assert result.layout.title.text == "Trading Strategy Performance"
    assert result.layout.annotations


def test_visualize_allows_missing_optional_indicator_columns() -> None:
    indicators = make_indicator_frame().drop(
        columns=["bb_upper", "bb_lower", "bb_mid", "sma_20", "sma_50"]
    )

    with patch.object(go.Figure, "show"):
        result = visualize_trading_results(indicators, pd.DataFrame(), pd.DataFrame())

    assert isinstance(result, go.Figure)
    trace_names = {trace.name for trace in result.data}
    assert "OHLC" in trace_names
    assert "Volume" in trace_names
    assert "RSI" in trace_names
    assert "ATR" in trace_names
    assert "BB Upper" not in trace_names


def test_visualize_custom_rsi_thresholds() -> None:
    with patch.object(go.Figure, "show"):
        result = visualize_trading_results(
            make_indicator_frame(),
            pd.DataFrame(),
            pd.DataFrame(),
            rsi_lower_thresh=25,
            rsi_upper_thresh=75,
        )

    assert isinstance(result, go.Figure)
    assert result.layout.yaxis3.range == (0, 100)
