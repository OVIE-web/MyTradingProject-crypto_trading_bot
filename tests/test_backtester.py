# tests/test_backtester.py

import pytest
import pandas as pd
import numpy as np
from src.backtester import backtest_strategy

@pytest.fixture
def simple_price_data():
    """Provides simple OHLCV data for backtesting."""
    dates = pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'])
    data = {
        'open': [100, 105, 102, 108, 110],
        'high': [106, 108, 109, 111, 112],
        'low': [99, 102, 100, 105, 107],
        'close': [105, 103, 108, 110, 109],
        'volume': [1000, 1200, 1100, 1300, 1050]
    }
    df = pd.DataFrame(data, index=dates)
    return df

def test_backtest_strategy_no_trades(simple_price_data):
    # Predictions lead to no trades
    predictions = pd.Series([0, 0, 0, 0, 0], index=simple_price_data.index)  # All hold
    trades_df, daily_portfolio_df = backtest_strategy(simple_price_data, predictions, initial_balance=1000)

    assert trades_df.empty
    assert not daily_portfolio_df.empty
    assert daily_portfolio_df['total_value'].iloc[-1] == 1000  # Balance should not change

def test_backtest_strategy_single_trade_cycle(simple_price_data):
    # Buy on Day 1, Sell on Day 2
    predictions = pd.Series([1, -1, 0, 0, 0], index=simple_price_data.index)
    initial_balance = 1000
    fee_pct = 0.001  # 0.1%
    trades_df, daily_portfolio_df = backtest_strategy(simple_price_data, predictions, initial_balance, fee_pct)

    assert not trades_df.empty
    assert len(trades_df) == 2  # One buy, one sell
    assert trades_df.iloc[0]['type'] == 'buy'
    assert trades_df.iloc[1]['type'] == 'sell'

    # Verify quantities and fees
    buy_price = 105
    sell_price = 103
    expected_shares = initial_balance / (buy_price * (1 + fee_pct))
    assert np.isclose(trades_df.iloc[0]['shares'], expected_shares)

    expected_final_balance = initial_balance / (1 + fee_pct) / buy_price * sell_price * (1 - fee_pct)
    assert np.isclose(daily_portfolio_df['total_value'].iloc[-1], expected_final_balance, rtol=1e-05)
    assert trades_df.iloc[1]['trade_return'] < 0  # Loss trade

def test_backtest_strategy_profitable_trade(simple_price_data):
    # Buy on Day 1, Sell on Day 3
    predictions = pd.Series([1, 0, -1, 0, 0], index=simple_price_data.index)
    initial_balance = 1000
    fee_pct = 0.001
    trades_df, daily_portfolio_df = backtest_strategy(simple_price_data, predictions, initial_balance, fee_pct)

    assert not trades_df.empty
    assert len(trades_df) == 2
    assert trades_df.iloc[1]['trade_return'] > 0  # Profitable trade
    assert daily_portfolio_df['total_value'].iloc[-1] > initial_balance

def test_backtest_strategy_insufficient_funds(simple_price_data):
    # High price, very low balance
    predictions = pd.Series([1, 0, 0, 0, 0], index=simple_price_data.index)
    trades_df, daily_portfolio_df = backtest_strategy(simple_price_data, predictions, initial_balance=0.01)

    assert trades_df.empty  # No trades due to insufficient balance
    assert daily_portfolio_df['total_value'].iloc[-1] == 0.01
