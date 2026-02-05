# src/visualizer.py

import logging

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_trading_results(
    df_with_indicators: pd.DataFrame,
    trades_df: pd.DataFrame,
    daily_portfolio_df: pd.DataFrame,
    rsi_lower_thresh: float = 30,
    rsi_upper_thresh: float = 70,
) -> go.Figure | None:
    """
    Creates interactive candlestick chart with trading signals, volume,
    account value, and technical indicators.

    Args:
        df_with_indicators (pd.DataFrame): DataFrame containing OHLCV and all calculated technical indicators.
                                           Must have 'open', 'high', 'low', 'close', 'volume',
                                           'bb_upper', 'bb_lower', 'bb_mid', 'sma_20', 'sma_50',
                                           'rsi', 'atr'.
        trades_df (pd.DataFrame): DataFrame with trade information (buy/sell points).
        daily_portfolio_df (pd.DataFrame): DataFrame with daily total portfolio value.
        rsi_lower_thresh (float): The lower RSI threshold for plotting.
        rsi_upper_thresh (float): The upper RSI threshold for plotting.
    """
    try:
        if df_with_indicators.empty:
            logging.warning("DataFrame with indicators is empty, cannot create visualization.")
            return None

        # Define the number of rows for subplots: Price+Trades, Volume, RSI, ATR, Account Value
        fig = make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,  # Adjust spacing for more subplots
            subplot_titles=(
                "Price Action & Trades with Indicators",
                "Volume",
                "Relative Strength Index (RSI)",
                "Average True Range (ATR)",
                "Account Value",
            ),
            row_heights=[0.5, 0.15, 0.15, 0.1, 0.1],
        )  # Allocate space for each subplot

        # --- Row 1: Price Action & Trades with Indicators ---

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df_with_indicators.index,
                open=df_with_indicators["open"],
                high=df_with_indicators["high"],
                low=df_with_indicators["low"],
                close=df_with_indicators["close"],
                name="OHLC",
                increasing_line_color="green",  # Customizing candle colors for clarity
                decreasing_line_color="red",
            ),
            row=1,
            col=1,
        )

        # Bollinger Bands
        if "bb_upper" in df_with_indicators.columns and "bb_lower" in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators["bb_upper"],
                    line=dict(color="red", width=1),
                    name="BB Upper",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators["bb_lower"],
                    line=dict(color="green", width=1),
                    name="BB Lower",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
        if "bb_mid" in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators["bb_mid"],
                    line=dict(color="blue", width=1),
                    name="BB Mid",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # Simple Moving Averages
        if "sma_20" in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators["sma_20"],
                    line=dict(color="orange", width=1),
                    name="SMA 20",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
        if "sma_50" in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators["sma_50"],
                    line=dict(color="purple", width=1),
                    name="SMA 50",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # Buy and Sell Signals (on price chart)
        if not trades_df.empty:
            buys = trades_df[trades_df["type"] == "buy"]
            if not buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buys.index,
                        y=buys["price"],
                        mode="markers",
                        name="Buy Signals",
                        marker=dict(color="green", size=10, symbol="triangle-up"),
                    ),
                    row=1,
                    col=1,
                )

            sells = trades_df[trades_df["type"] == "sell"]
            if not sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sells.index,
                        y=sells["price"],
                        mode="markers",
                        name="Sell Signals",
                        marker=dict(color="red", size=10, symbol="triangle-down"),
                    ),
                    row=1,
                    col=1,
                )
        else:
            logging.info("No trades data to plot buy/sell signals.")

        # --- Row 2: Volume Chart ---
        fig.add_trace(
            go.Bar(
                x=df_with_indicators.index,
                y=df_with_indicators["volume"],
                name="Volume",
                marker_color="blue",
                opacity=0.3,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # --- Row 3: Relative Strength Index (RSI) ---
        if "rsi" in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators["rsi"],
                    line=dict(color="darkblue", width=1.5),
                    name="RSI",
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
            # Add RSI overbought/oversold lines
            fig.add_hline(
                y=rsi_upper_thresh,
                line_dash="dot",
                line_color="red",
                row="3",
                col="1",
                name="RSI Overbought",
            )
            fig.add_hline(
                y=rsi_lower_thresh,
                line_dash="dot",
                line_color="green",
                row="3",
                col="1",
                name="RSI Oversold",
            )
            fig.update_yaxes(range=[0, 100], row="3", col="1")  # Standard RSI range
        # --- Row 4: Average True Range (ATR) ---
        if "atr" in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators["atr"],
                    line=dict(color="darkgreen", width=1.5),
                    name="ATR",
                    showlegend=False,
                ),
                row=4,
                col=1,
            )

        # --- Row 5: Account Value ---
        if not daily_portfolio_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=daily_portfolio_df.index,
                    y=daily_portfolio_df["total_value"],
                    name="Account Value",
                    line=dict(color="purple", width=2),
                ),
                row=5,
                col=1,
            )

        # --- Update Layout and Axes ---
        fig.update_layout(
            title="Trading Strategy Performance",
            xaxis5_title="Date",  # The last x-axis is for the bottommost subplot
            yaxis_title="Price",
            yaxis2_title="Volume",
            yaxis3_title="RSI",
            yaxis4_title="ATR",
            yaxis5_title="Value ($)",
            width=1200,
            height=1200,  # Increased height to accommodate more subplots
            showlegend=True,
            hovermode="x unified",  # Shows all data for a given X coordinate
            # Hide rangeslider for all subplots except the bottom one
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
            xaxis3_rangeslider_visible=False,
            xaxis4_rangeslider_visible=False,
        )

        # Add range slider to the bottommost x-axis
        fig.update_xaxes(rangeslider_visible=True, row=5, col=1)  # Rangeslider on the last x-axis

        # Add performance metrics as annotations
        if not daily_portfolio_df.empty:
            initial_balance = daily_portfolio_df.iloc[0]["total_value"]
            final_balance = daily_portfolio_df.iloc[-1]["total_value"]
            total_return = (
                ((final_balance - initial_balance) / initial_balance) * 100
                if initial_balance != 0
                else 0
            )

            num_trades = len(trades_df) // 2 if not trades_df.empty and len(trades_df) >= 2 else 0
            profitable_trades_count = 0
            if num_trades > 0:
                for i in range(0, len(trades_df) - 1, 2):
                    if (
                        trades_df.iloc[i]["type"] == "buy"
                        and trades_df.iloc[i + 1]["type"] == "sell"
                    ):
                        buy_cost = (
                            trades_df.iloc[i]["shares"] * trades_df.iloc[i]["price"]
                            + trades_df.iloc[i]["fee"]
                        )
                        sell_proceeds = (
                            trades_df.iloc[i + 1]["shares"] * trades_df.iloc[i + 1]["price"]
                            - trades_df.iloc[i + 1]["fee"]
                        )
                        if sell_proceeds > buy_cost:
                            profitable_trades_count += 1
                win_rate = (profitable_trades_count / num_trades) * 100
            else:
                win_rate = 0

            metrics_text = (
                f"Initial Balance: ${initial_balance:,.2f}<br>"
                f"Final Balance: ${final_balance:,.2f}<br>"
                f"Total Return: {total_return:.2f}%<br>"
                f"Number of Trades: {num_trades}<br>"
                f"Win Rate: {win_rate:.2f}%"
            )

            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.0,  # Position top right
                xanchor="right",
                yanchor="top",  # Anchor the text to the top right of the box
                text=metrics_text,
                showarrow=False,
                font=dict(size=12),
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
            )

        fig.show()
        logging.info("Visualization displayed.")
        return fig

    except Exception as e:
        logging.error(f"Error in visualization: {str(e)}")
        raise
