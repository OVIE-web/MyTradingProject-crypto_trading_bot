# src/binance_manager.py

import logging
from binance.client import Client
from binance.enums import * # Import enums for order types, etc.
import pandas as pd


from src.config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET, BINANCE_API_URL,
    BINANCE_TESTNET_API_URL, TRADE_SYMBOL, TRADE_INTERVAL, INITIAL_CANDLES_HISTORY
)

# No logging.basicConfig here, as it's handled in main.py

class BinanceManager:
    def __init__(self):
        """Initializes the Binance API client."""
        # Validate credentials first
        if (not BINANCE_API_KEY or BINANCE_API_KEY.strip() in ["your_binance_api_key", ""]) \
           or (not BINANCE_API_SECRET or BINANCE_API_SECRET.strip() in ["your_binance_api_secret", ""]):
            raise ValueError("API credentials missing")

        api_url = BINANCE_TESTNET_API_URL if BINANCE_TESTNET else BINANCE_API_URL
        logging.info(f"Using Binance API URL: {api_url}")

        # Create client only after validation
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        self.client.API_URL = api_url  # Override the API URL for testnet if needed

        try:
            # Test connection
            self.client.ping()
            logging.info(f"Connected to Binance API (Testnet: {BINANCE_TESTNET}) successfully.")
            self.account_info = self.client.get_account()
            logging.info(f"Account loaded for Testnet: {BINANCE_TESTNET}")
        except Exception as e:
            logging.critical(f"Failed to connect to Binance API: {e}", exc_info=True)
            raise


    def get_latest_ohlcv_candles(self, symbol=TRADE_SYMBOL, interval=TRADE_INTERVAL, limit=INITIAL_CANDLES_HISTORY):
        """
        Fetches the latest OHLCV candles from Binance.
        Returns a pandas DataFrame.
        """
        try:
            # interval is like '1h', '1d', '1m', etc.
            # klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            # Use historical_klines for fetching past data
            klines = self.client.get_historical_klines(symbol, interval, "{} minutes ago UTC".format(int(limit) * self._interval_to_minutes(interval)))
            
            # Format into pandas DataFrame
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Convert to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                            'quote_asset_volume', 'number_of_trades',
                            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.set_index('open_time', inplace=True)
            df.sort_index(inplace=True)
            
            logging.info(f"Fetched {len(df)} {interval} candles for {symbol}. Latest close: {df['close'].iloc[-1]}")
            return df
        except Exception as e:
            logging.error(f"Error fetching candles for {symbol}: {e}", exc_info=True)
            return pd.DataFrame() # Return empty DataFrame on error

    def _interval_to_minutes(self, interval):
        """Helper to convert Binance interval string to minutes."""
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval.endswith('d'):
            return int(interval[:-1]) * 24 * 60
        # Add more intervals as needed
        return 1 # Default to 1 minute if unknown

    def get_account_balance(self, asset='USDT'):
        """Gets the free balance of a specific asset."""
        try:
            balance = self.client.get_asset_balance(asset=asset)
            free_balance = float(balance['free'])
            logging.info(f"Current {asset} balance: {free_balance}")
            return free_balance
        except Exception as e:
            logging.error(f"Error getting {asset} balance: {e}", exc_info=True)
            return 0.0

    def place_market_order(self, symbol, quantity, side):
        """
        Places a market order (BUY or SELL).
        
        Args:
            symbol (str): Trading pair (e.g., 'BTCUSDT')
            quantity (float): Amount of base asset (e.g., BTC amount for BTCUSDT)
            side (str): 'BUY' or 'SELL'
        """
        try:
            if not isinstance(quantity, (int, float)) or quantity <= 0:
                logging.error(f"Invalid quantity for market order: {quantity}")
                return None

            # Get symbol info to determine precision
            info = self.client.get_symbol_info(symbol)
            if not info:
                logging.error(f"Could not get symbol info for {symbol}")
                return None
            
            # Find quantity precision (stepSize)
            quantity_precision = 0
            for f in info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    quantity_precision = len(str(float(f['stepSize']))[str(float(f['stepSize'])).find('.')+1:])
                    break
            
            # Adjust quantity to fit precision
            adjusted_quantity = round(quantity, quantity_precision)
            
            if adjusted_quantity <= 0:
                logging.warning(f"Adjusted quantity for {symbol} is zero or negative ({adjusted_quantity}), skipping order.")
                return None

            logging.info(f"Attempting to place {side} market order for {adjusted_quantity} {symbol}...")
            
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=adjusted_quantity
            )
            logging.info(f"Market {side} order placed successfully: {order}")
            return order
        except Exception as e:
            logging.error(f"Error placing market {side} order for {symbol} (Qty: {quantity}): {e}", exc_info=True)
            return None

    def get_server_time(self):
        """Gets Binance server time for synchronization."""
        try:
            server_time = self.client.get_server_time()
            return server_time['serverTime']
        except Exception as e:
            logging.warning(f"Could not get Binance server time: {e}")
            return None