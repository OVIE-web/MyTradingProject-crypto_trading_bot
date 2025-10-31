# src/backtester.py
import pandas as pd
import logging

from src.config import INITIAL_BALANCE, TRANSACTION_FEE_PCT

def backtest_strategy(df_original, predictions, initial_balance=INITIAL_BALANCE, transaction_fee_pct=TRANSACTION_FEE_PCT):
    """
    Backtests the trading strategy based on generated predictions.
    Assumes df_original and predictions are aligned by index.
    
    Args:
        df_original (pd.DataFrame): Original price data (OHLCV) for the backtest period.
                                    Must include 'close' price.
        predictions (pd.Series): Model predictions (-1: sell, 0: hold, 1: buy).
                                 Must have the same index as df_original.
        initial_balance (float): Starting capital for the backtest.
        transaction_fee_pct (float): Transaction fee (e.g., 0.001 for 0.1%).
    
    Returns:
        tuple: (pd.DataFrame: DataFrame with detailed trade log,
                pd.DataFrame: DataFrame with daily portfolio value snapshots)
    """
    if df_original.empty or predictions.empty:
        logging.warning("Empty DataFrame or predictions provided to backtest_strategy. No backtest performed.")
        return pd.DataFrame(), pd.DataFrame()

    if not df_original.index.equals(predictions.index):
        logging.error("Index mismatch between df_original and predictions in backtest_strategy.")
        raise ValueError("df_original and predictions must have identical indices for backtesting.")
    
    if 'close' not in df_original.columns:
        logging.error("Missing 'close' column in df_original for backtesting.")
        raise ValueError("df_original must contain a 'close' price column.")

    # Prepare a DataFrame for daily tracking, merging predictions
    results = df_original.copy()
    results['prediction'] = predictions

    # Initialize trading variables
    balance = float(initial_balance)
    position = 0
    shares = 0.0
    trades = []

    daily_portfolio_values = []

    try:
        logging.info(f"Starting backtest with initial balance: ${initial_balance:.2f}")
        for i in range(len(results)):
            current_date = results.index[i]
            current_price = results.iloc[i]['close']
            signal = results.iloc[i]['prediction']

            # Always record portfolio value and composition for the day
            shares_value = shares * current_price if position == 1 else 0
            current_portfolio_value = balance + shares_value
            daily_portfolio_values.append({
                'date': current_date,
                'total_value': current_portfolio_value,
                'cash': balance,
                'shares': shares,
                'shares_value': shares_value,
                'position': position
            })

            # --- Trading Logic ---
            # Buy Signal: No position and signal is 1 (buy)
            if position == 0 and signal == 1:
                cost_per_unit_with_fee = current_price * (1 + transaction_fee_pct)
                potential_shares = balance / cost_per_unit_with_fee

                # Check if enough balance for a meaningful trade (at least $10 worth)
                MIN_TRADE_VALUE = 10.0  # Minimum trade value in quote currency
                if potential_shares * current_price >= MIN_TRADE_VALUE:
                    shares_to_buy = potential_shares
                    cost = shares_to_buy * current_price
                    fee = cost * transaction_fee_pct
                    
                    balance -= (cost + fee)
                    shares = shares_to_buy
                    position = 1
                    trades.append({
                        'date': current_date,
                        'type': 'buy',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'fee': fee,
                        'balance': balance,
                        'portfolio_value_after_trade': balance + (shares * current_price)
                    })
                    logging.debug(f"BUY executed on {current_date.strftime('%Y-%m-%d')}: @ ${current_price:.2f}, Shares: {shares_to_buy:.4f}, Rem. Balance: ${balance:.2f}")
                else:
                    logging.debug(f"BUY signal on {current_date.strftime('%Y-%m-%d')} but not enough balance or too small amount to buy.")

            # Sell Signal: Holding a position and signal is -1 (sell) or 0 (hold, implying exit)
            elif position == 1 and (signal == -1 or signal == 0):
                if shares > 0:
                    value = shares * current_price
                    fee = value * transaction_fee_pct
                    balance += (value - fee)
                    
                    # Find the last buy trade to calculate return
                    last_buy = next((t for t in reversed(trades) if t['type'] == 'buy'), None)
                    if last_buy:
                        # Calculate return accounting for fees on both buy and sell
                        buy_with_fee = last_buy['price'] * (1 + transaction_fee_pct)
                        sell_with_fee = current_price * (1 - transaction_fee_pct)
                        trade_return = (sell_with_fee / buy_with_fee) - 1
                    else:
                        trade_return = 0
                    
                    trades.append({
                        'date': current_date,
                        'type': 'sell',
                        'trade_return': trade_return,
                        'price': current_price,
                        'shares': shares,
                        'fee': fee,
                        'balance': balance,
                        'portfolio_value_after_trade': balance
                    })
                    logging.debug(f"SELL executed on {current_date.strftime('%Y-%m-%d')}: @ ${current_price:.2f}, Shares: {shares:.4f}, New Balance: ${balance:.2f}")
                    shares = 0.0
                    position = 0
                else:
                    logging.debug(f"SELL signal on {current_date.strftime('%Y-%m-%d')} but no shares to sell (position already closed).")
            else:
                logging.debug(f"HOLD signal on {current_date.strftime('%Y-%m-%d')}.")


        # Close any remaining position at the end of the backtest
        if position == 1 and shares > 0:
            final_price = results.iloc[-1]['close']
            value = shares * final_price
            fee = value * transaction_fee_pct
            balance += (value - fee)
            
            # Find the last buy trade to calculate return for final trade
            last_buy = next((t for t in reversed(trades) if t['type'] == 'buy'), None)
            if last_buy:
                # Calculate return accounting for fees on both buy and sell
                buy_with_fee = last_buy['price'] * (1 + transaction_fee_pct)
                sell_with_fee = final_price * (1 - transaction_fee_pct)
                trade_return = (sell_with_fee / buy_with_fee) - 1
            else:
                trade_return = 0
            
            trades.append({
                'date': results.index[-1],
                'type': 'sell',
                'price': final_price,
                'shares': shares,
                'fee': fee,
                'balance': balance,
                'portfolio_value_after_trade': balance,
                'trade_return': trade_return
            })
            logging.debug(f"FINAL SELL executed on {results.index[-1].strftime('%Y-%m-%d')}: @ ${final_price:.2f}, Shares: {shares:.4f}, Final Balance: ${balance:.2f}")

        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df.set_index('date', inplace=True)
            # Don't overwrite trade_return as it's already calculated for each trade
        else:
            logging.info("No trades were executed during the backtesting period.")

        # Convert daily portfolio values to DataFrame
        daily_portfolio_df = pd.DataFrame(daily_portfolio_values)
        if not daily_portfolio_df.empty:
            daily_portfolio_df.set_index('date', inplace=True)
        else:
            logging.warning("No daily portfolio values generated. Initializing with initial balance.")
            daily_portfolio_df = pd.DataFrame([{'date': df_original.index[0], 'total_value': initial_balance}]).set_index('date')
            
        # --- Print Performance Metrics ---
        final_balance_overall = daily_portfolio_df.iloc[-1]['total_value'] if not daily_portfolio_df.empty else initial_balance
        total_return_overall = (final_balance_overall - initial_balance) / initial_balance * 100 if initial_balance != 0 else 0
        
        logging.info(f'\n=== Backtesting Performance Summary ===')
        logging.info(f'Initial Balance: ${initial_balance:.2f}')
        logging.info(f'Final Balance: ${final_balance_overall:.2f}')
        logging.info(f'Total Return: {total_return_overall:.2f}%')
        
        num_trades = len(trades_df) // 2 if not trades_df.empty and len(trades_df) >= 2 else 0
        logging.info(f'Number of Completed Round-trip Trades: {num_trades}')

        if num_trades > 0:
            profitable_trades_count = 0
            for i in range(0, len(trades_df) - 1, 2):
                if trades_df.iloc[i]['type'] == 'buy' and trades_df.iloc[i+1]['type'] == 'sell':
                    buy_cost = trades_df.iloc[i]['shares'] * trades_df.iloc[i]['price'] + trades_df.iloc[i]['fee']
                    sell_proceeds = trades_df.iloc[i+1]['shares'] * trades_df.iloc[i+1]['price'] - trades_df.iloc[i+1]['fee']
                    
                    trade_profit_loss = sell_proceeds - buy_cost # <--- FIXED THIS LINE
                    
                    if trade_profit_loss > 0:
                        profitable_trades_count += 1
            
            win_rate = (profitable_trades_count / num_trades) * 100 if num_trades > 0 else 0
            logging.info(f'Win Rate (closed trades): {win_rate:.2f}%')
            
            sell_trades = trades_df[trades_df['type'] == 'sell']
            if not sell_trades.empty:
                # The 'trade_return' here is the change in total balance *after* a sell, not necessarily a round-trip % return
                avg_return = sell_trades["trade_return"].mean()*100
                best_return = sell_trades["trade_return"].max()*100
                worst_return = sell_trades["trade_return"].min()*100
                logging.info(f'Average Return per Sell Action: {avg_return:.2f}%')
                logging.info(f'Best Sell Action Return: {best_return:.2f}%')
                logging.info(f'Worst Sell Action Return: {worst_return:.2f}%')
            else:
                logging.info('No completed sell actions to calculate individual trade returns.')
        else:
            logging.info('No completed trades to calculate win rate or average returns.')
        
        return trades_df, daily_portfolio_df
    
    except Exception as e:
        logging.error(f'Error during backtesting: {str(e)}')
        raise