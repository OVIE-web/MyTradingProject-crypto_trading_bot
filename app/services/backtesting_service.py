# app/services/backtesting_service.py
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.core.config import INITIAL_BALANCE, TRANSACTION_FEE_PCT

LOG = logging.getLogger(__name__)


def backtest_strategy(
    df_original: pd.DataFrame,
    predictions: pd.Series,
    initial_balance: float = INITIAL_BALANCE,
    transaction_fee_pct: float = TRANSACTION_FEE_PCT,
    min_trade_value: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtest a simple long-only trading strategy using model predictions.

    Signal meaning:
        1  = buy
        0  = hold / exit
        -1 = sell

    Args:
        df_original: Historical OHLCV/features DataFrame. Must include a 'close' column.
        predictions: Prediction Series aligned with df_original index.
        initial_balance: Starting cash balance.
        transaction_fee_pct: Trading fee percentage, e.g. 0.001 for 0.1%.
        min_trade_value: Minimum trade value required before entering a position.

    Returns:
        trades_df: Executed trade log.
        daily_portfolio_df: Daily portfolio value snapshots.
    """
    if df_original.empty or predictions.empty:
        LOG.warning("Empty DataFrame or predictions provided. No backtest performed.")
        return pd.DataFrame(), pd.DataFrame()

    if not df_original.index.equals(predictions.index):
        LOG.error("Index mismatch between df_original and predictions.")
        raise ValueError("df_original and predictions must have identical indices.")

    if "close" not in df_original.columns:
        LOG.error("Missing 'close' column in df_original.")
        raise ValueError("df_original must contain a 'close' column.")

    results = df_original.copy()
    results["prediction"] = predictions

    balance = float(initial_balance)
    position = 0
    shares = 0.0

    trades: list[dict[str, Any]] = []
    daily_portfolio_values: list[dict[str, Any]] = []

    LOG.info("Starting backtest with initial balance: $%.2f", initial_balance)

    try:
        for i in range(len(results)):
            current_date = results.index[i]
            current_price = float(results.iloc[i]["close"])
            signal = int(results.iloc[i]["prediction"])

            shares_value = shares * current_price if position == 1 else 0.0
            current_portfolio_value = balance + shares_value

            daily_portfolio_values.append(
                {
                    "date": current_date,
                    "total_value": current_portfolio_value,
                    "cash": balance,
                    "shares": shares,
                    "shares_value": shares_value,
                    "position": position,
                }
            )

            # BUY: enter long position only if no current position.
            if position == 0 and signal == 1:
                cost_per_unit_with_fee = current_price * (1 + transaction_fee_pct)
                potential_shares = balance / cost_per_unit_with_fee
                trade_value = potential_shares * current_price

                if trade_value >= min_trade_value:
                    cost = trade_value
                    fee = cost * transaction_fee_pct

                    balance -= cost + fee
                    shares = potential_shares
                    position = 1

                    trades.append(
                        {
                            "date": current_date,
                            "type": "buy",
                            "price": current_price,
                            "shares": shares,
                            "fee": fee,
                            "balance": balance,
                            "portfolio_value_after_trade": balance + shares * current_price,
                            "trade_return": 0.0,
                        }
                    )
                else:
                    LOG.debug(
                        "BUY signal skipped at %s. Trade value %.2f below minimum %.2f.",
                        current_date,
                        trade_value,
                        min_trade_value,
                    )

            # SELL / EXIT: close position when signal is sell or neutral.
            elif position == 1 and signal in (-1, 0):
                value = shares * current_price
                fee = value * transaction_fee_pct
                balance += value - fee

                last_buy = next(
                    (trade for trade in reversed(trades) if trade["type"] == "buy"),
                    None,
                )

                trade_return = 0.0
                if last_buy:
                    buy_price = float(last_buy["price"])
                    trade_return = (
                        (current_price * (1 - transaction_fee_pct))
                        / (buy_price * (1 + transaction_fee_pct))
                    ) - 1

                trades.append(
                    {
                        "date": current_date,
                        "type": "sell",
                        "price": current_price,
                        "shares": shares,
                        "fee": fee,
                        "balance": balance,
                        "portfolio_value_after_trade": balance,
                        "trade_return": trade_return,
                    }
                )

                shares = 0.0
                position = 0

        # Close any open position at the final close price.
        if position == 1 and shares > 0:
            final_date = results.index[-1]
            final_price = float(results.iloc[-1]["close"])

            value = shares * final_price
            fee = value * transaction_fee_pct
            balance += value - fee

            last_buy = next(
                (trade for trade in reversed(trades) if trade["type"] == "buy"),
                None,
            )

            trade_return = 0.0
            if last_buy:
                buy_price = float(last_buy["price"])
                trade_return = (
                    (final_price * (1 - transaction_fee_pct))
                    / (buy_price * (1 + transaction_fee_pct))
                ) - 1

            trades.append(
                {
                    "date": final_date,
                    "type": "sell",
                    "price": final_price,
                    "shares": shares,
                    "fee": fee,
                    "balance": balance,
                    "portfolio_value_after_trade": balance,
                    "trade_return": trade_return,
                }
            )

            shares = 0.0
            position = 0

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df.set_index("date", inplace=True)

        daily_portfolio_df = pd.DataFrame(daily_portfolio_values)
        if not daily_portfolio_df.empty:
            daily_portfolio_df.set_index("date", inplace=True)

        final_balance = balance
        total_return_pct = (
            ((final_balance - initial_balance) / initial_balance) * 100 if initial_balance else 0.0
        )

        completed_trades = (
            len(trades_df[trades_df["type"] == "sell"])
            if not trades_df.empty and "type" in trades_df.columns
            else 0
        )

        LOG.info("Backtest complete.")
        LOG.info("Initial Balance: $%.2f", initial_balance)
        LOG.info("Final Balance: $%.2f", final_balance)
        LOG.info("Total Return: %.2f%%", total_return_pct)
        LOG.info("Completed Trades: %s", completed_trades)

        return trades_df, daily_portfolio_df

    except Exception:
        LOG.exception("Unhandled error during backtesting.")
        raise
