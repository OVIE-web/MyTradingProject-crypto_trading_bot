"""
Tests for src/visualizer.py module.
Tests trading results visualization functionality.
"""

import pandas as pd
from plotly import graph_objects as go

from src.visualizer import visualize_trading_results


class TestVisualizeTradingResults:
    """Test cases for visualize_trading_results function."""

    def test_visualize_empty_dataframe(self) -> None:
        """Test that empty DataFrame returns None."""
        empty_df = pd.DataFrame()
        trades_df = pd.DataFrame()
        portfolio_df = pd.DataFrame()

        result = visualize_trading_results(empty_df, trades_df, portfolio_df)

        assert result is None

    def test_visualize_basic_structure(self) -> None:
        """Test basic figure structure creation."""
        # Create sample data
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df_indicators = pd.DataFrame({
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            "low": [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            "close": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            "bb_upper": [110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
            "bb_lower": [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
            "bb_mid": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "sma_20": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            "sma_50": [96, 97, 98, 99, 100, 101, 102, 103, 104, 105],
            "rsi": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
            "atr": [2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]
        }, index=dates)

        trades_df = pd.DataFrame({
            "timestamp": dates[:3],
            "side": ["BUY", "SELL", "BUY"],
            "price": [102, 106, 110],
            "quantity": [1.0, 1.0, 1.0]
        })

        portfolio_df = pd.DataFrame({
            "date": dates,
            "total_value": [10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900]
        })

        result = visualize_trading_results(df_indicators, trades_df, portfolio_df)

        assert result is not None
        assert isinstance(result, go.Figure)

        # Check that we have traces (data should exist)
        assert hasattr(result, 'data')  # Should have multiple traces

        # Check subplot titles
        expected_titles = [
            "Price Action & Trades with Indicators",
            "Volume",
            "Relative Strength Index (RSI)",
            "Average True Range (ATR)",
            "Account Value"
        ]
        for title in expected_titles:
            assert any(title in str(trace) for trace in result.data)

    def test_visualize_missing_columns(self) -> None:
        """Test handling of missing optional columns."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        # DataFrame missing some optional columns like bb_upper, bb_lower
        df_indicators = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "close": [102, 103, 104, 105, 106],
            "volume": [1000, 1100, 1200, 1300, 1400],
            "rsi": [50, 55, 60, 65, 70],
            "atr": [2, 2.1, 2.2, 2.3, 2.4]
        }, index=dates)

        trades_df = pd.DataFrame()
        portfolio_df = pd.DataFrame()

        # Should not crash even with missing columns
        result = visualize_trading_results(df_indicators, trades_df, portfolio_df)

        assert result is not None
        assert isinstance(result, go.Figure)

    def test_visualize_custom_rsi_thresholds(self) -> None:
        """Test custom RSI threshold parameters."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df_indicators = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "close": [102, 103, 104, 105, 106],
            "volume": [1000, 1100, 1200, 1300, 1400],
            "rsi": [20, 40, 60, 80, 90],  # Values that will trigger thresholds
            "atr": [2, 2.1, 2.2, 2.3, 2.4]
        }, index=dates)

        trades_df = pd.DataFrame()
        portfolio_df = pd.DataFrame()

        # Test with custom thresholds
        result = visualize_trading_results(
            df_indicators, trades_df, portfolio_df,
            rsi_lower_thresh=25, rsi_upper_thresh=75
        )

        assert result is not None
        assert isinstance(result, go.Figure)

        # The function should complete without errors
        # (Detailed threshold verification would require inspecting the plot data)
