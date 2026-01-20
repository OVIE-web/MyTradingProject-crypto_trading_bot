# src/backtester.py
from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

from src.config import INITIAL_BALANCE, TRANSACTION_FEE_PCT

logger = logging.getLogger(__name__)


def backtest_strategy(
    df_original: pd.DataFrame,
    predictions: pd.Series,
    initial_balance: float = INITIAL_BALANCE,
    transaction_fee_pct: float = TRANSACTION_FEE_PCT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtests the trading strategy based on generated predictions.
    Assumes df_original and predictions are aligned by index.
    """
    if df_original.empty or predictions.empty:
        logger.warning(
            "Empty DataFrame or predictions provided to backtest_strategy. No backtest performed."
        )
        return pd.DataFrame(), pd.DataFrame()

    if not df_original.index.equals(predictions.index):
        logger.error("Index mismatch between df_original and predictions in backtest_strategy.")
        raise ValueError("df_original and predictions must have identical indices.")

    if "close" not in df_original.columns:
        logger.error("Missing 'close' column in df_original for backtesting.")
        raise ValueError("df_original must contain a 'close' column.")

    results = df_original.copy()
    results["prediction"] = predictions

    balance: float = float(initial_balance)
    position: int = 0
    shares: float = 0.0
    trades: list[dict[str, float | str | pd.Timestamp]] = []
    daily_portfolio_values: list[dict[str, float | int | pd.Timestamp]] = []

    try:
        logger.info("Starting backtest with initial balance: $%.2f", initial_balance)

        for i in range(len(results)):
            current_date = results.index[i]
            current_price: float = float(results.iloc[i]["close"])
            signal: int = int(results.iloc[i]["prediction"])

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

            # BUY
            if position == 0 and signal == 1:
                cost_per_unit = current_price * (1 + transaction_fee_pct)
                potential_shares = balance / cost_per_unit

                if potential_shares * current_price >= 10.0:
                    cost = potential_shares * current_price
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
                        }
                    )

            # SELL
            elif position == 1 and signal in (-1, 0):
                value = shares * current_price
                fee = value * transaction_fee_pct
                balance += value - fee

                last_buy = next((t for t in reversed(trades) if t["type"] == "buy"), None)
                trade_return = 0.0

                if last_buy:
                    buy_price = float(last_buy["price"])
                    trade_return = (current_price * (1 - transaction_fee_pct)) / (
                        buy_price * (1 + transaction_fee_pct)
                    ) - 1

                trades.append(
                    {
                        "date": current_date,
                        "type": "sell",
                        "price": current_price,
                        "shares": shares,
                        "fee": fee,
                        "balance": balance,
                        "trade_return": trade_return,
                    }
                )

                shares = 0.0
                position = 0

        trades_df = pd.DataFrame(trades).set_index("date") if trades else pd.DataFrame()

        daily_portfolio_df = (
            pd.DataFrame(daily_portfolio_values).set_index("date")
            if daily_portfolio_values
            else pd.DataFrame()
        )

        logger.info("Backtest complete | Final Balance: $%.2f", balance)
        return trades_df, daily_portfolio_df

    except Exception:
        logger.exception("Unhandled error during backtesting")
        raise
