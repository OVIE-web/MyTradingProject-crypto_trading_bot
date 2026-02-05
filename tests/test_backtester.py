# tests/test_backtester.py

import numpy as np
import pandas as pd
import pytest

import src.config
from src.backtester import backtest_strategy

src.config.TRANSACTION_FEE_PCT = 0


@pytest.fixture
def simple_price_data() -> pd.DataFrame:
    """Provides simple OHLCV data for backtesting."""
    dates = pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05"])
    data = {
        "open": [100, 105, 102, 108, 110],
        "high": [106, 108, 109, 111, 112],
        "low": [99, 102, 100, 105, 107],
        "close": [105, 103, 108, 110, 109],
        "volume": [1000, 1200, 1100, 1300, 1050],
    }
    df = pd.DataFrame(data, index=dates)
    return df


def test_backtest_strategy_no_trades(simple_price_data: pd.DataFrame) -> None:
    """Test that no trades occur when all predictions are hold (0)."""
    # Predictions lead to no trades
    predictions = pd.Series([0, 0, 0, 0, 0], index=simple_price_data.index)  # All hold
    initial_balance = 1000
    trades_df, daily_portfolio_df = backtest_strategy(
        simple_price_data, predictions, initial_balance=initial_balance
    )

    # Verify no trades occurred
    assert trades_df.empty, "Expected no trades when all predictions are hold"
    assert not daily_portfolio_df.empty, "Daily portfolio data should not be empty"
    # Verify balance remains unchanged
    final_value = daily_portfolio_df.iloc[-1]["total_value"]
    assert np.isclose(final_value, initial_balance, rtol=1e-10), (
        f"Expected final value {final_value} to equal initial balance {initial_balance}"
    )


def test_backtest_strategy_single_trade_cycle(simple_price_data: pd.DataFrame) -> None:
    # Buy on Day 1, Sell on Day 2
    predictions = pd.Series([1, -1, 0, 0, 0], index=simple_price_data.index)
    initial_balance = 1000
    fee_pct = 0.001  # 0.1%
    trades_df, daily_portfolio_df = backtest_strategy(
        simple_price_data, predictions, initial_balance, fee_pct
    )

    assert not trades_df.empty
    assert len(trades_df) == 2  # One buy, one sell
    assert trades_df.iloc[0]["type"] == "buy"
    assert trades_df.iloc[1]["type"] == "sell"

    # Verify quantities and fees
    buy_price = 105
    sell_price = 103
    expected_shares = initial_balance / (buy_price * (1 + fee_pct))
    assert np.isclose(trades_df.iloc[0]["shares"], expected_shares)

    expected_final_balance = (
        initial_balance / (1 + fee_pct) / buy_price * sell_price * (1 - fee_pct)
    )
    assert np.isclose(
        daily_portfolio_df["total_value"].iloc[-1], expected_final_balance, rtol=1e-05
    )
    assert trades_df.iloc[1]["trade_return"] < 0  # Loss trade


def test_backtest_strategy_profitable_trade(simple_price_data: pd.DataFrame) -> None:
    """Test a profitable trading scenario (buy low, sell high)."""
    predictions = pd.Series([0, 1, 0, -1, 0], index=simple_price_data.index)
    initial_balance = 1000
    fee_pct = 0.001

    trades_df, daily_portfolio_df = backtest_strategy(
        simple_price_data, predictions, initial_balance=initial_balance, transaction_fee_pct=fee_pct
    )

    assert not trades_df.empty, "Expected trades to be executed"
    assert len(trades_df) == 2, "Expected one buy and one sell trade"

    buy_trade = trades_df.iloc[0]
    sell_trade = trades_df.iloc[1]
    assert buy_trade["type"] == "buy", "First trade should be buy"
    assert sell_trade["type"] == "sell", "Second trade should be sell"

    actual_return = sell_trade["trade_return"]
    assert actual_return > 0, f"Expected positive return, got {actual_return}"

    final_value = daily_portfolio_df["total_value"].iloc[-1]
    assert final_value > initial_balance, (
        f"Expected final value {final_value} to be greater than initial balance {initial_balance}"
    )


def test_backtest_strategy_insufficient_funds(simple_price_data: pd.DataFrame) -> None:
    """Test behavior when balance is too low to execute trades."""
    # Attempt to buy with extremely low balance
    predictions = pd.Series([1, 0, 0, 0, 0], index=simple_price_data.index)
    initial_balance = 0.01  # Too low to buy any meaningful amount

    trades_df, daily_portfolio_df = backtest_strategy(
        simple_price_data, predictions, initial_balance=initial_balance
    )

    # Verify no trades were executed
    assert trades_df.empty, "Expected no trades due to insufficient balance"

    # Verify balance remains unchanged
    final_value = daily_portfolio_df["total_value"].iloc[-1]
    assert np.isclose(final_value, initial_balance, rtol=1e-10), (
        f"Expected final value {final_value} to equal initial balance {initial_balance}"
    )

    # Verify portfolio tracking is correct
    assert all(daily_portfolio_df["cash"] == initial_balance), "Cash balance should remain constant"
    assert all(daily_portfolio_df["shares"] == 0), "No shares should be held"
