import pytest
import importlib
from unittest.mock import patch, MagicMock
import pandas as pd
from src.binance_manager import BinanceManager
from src.config import TRADE_SYMBOL, TRADE_INTERVAL, INITIAL_CANDLES_HISTORY


# --- Shared mock fixture for Binance Client ---
@pytest.fixture
def mock_binance_client():
    with patch("src.binance_manager.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        # Mock common Binance API calls
        mock_instance.ping.return_value = {}
        mock_instance.get_account.return_value = {"balances": []}
        mock_instance.get_asset_balance.side_effect = lambda asset: {
            "free": "1000.0" if asset == "USDT" else "0.0"
        }
        mock_instance.get_server_time.return_value = {"serverTime": 1678886400000}
        mock_instance.get_symbol_info.return_value = {
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "0.000001"},
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
            ]
        }
        yield mock_instance


# --- Mock klines data ---
@pytest.fixture
def mock_klines_data():
    return [
        [1672531200000, "16000", "16100", "15900", "16050", "100", 1672617599999, "1605000", 1000, "50", "800000", 0],
        [1672617600000, "16050", "16200", "16000", "16150", "120", 1672703999999, "1938000", 1200, "60", "900000", 0],
        [1672704000000, "16150", "16300", "16100", "16250", "150", 1672790399999, "2437500", 1500, "75", "1200000", 0],
    ]


# --- Tests ---

def test_binance_manager_init_success(mock_binance_client):
    manager = BinanceManager()
    mock_binance_client.ping.assert_called_once()
    mock_binance_client.get_account.assert_called_once(): MagicMock
    assert manager.client == mock_binance_client


def test_binance_manager_init_no_api_keys(monkeypatch):
    # --- Patch the config variables directly in memory ---
    monkeypatch.setattr("src.config.BINANCE_API_KEY", None)
    monkeypatch.setattr("src.config.BINANCE_API_SECRET", None)

    # Ensure Binance client is mocked so no network call happens
    with patch("src.binance_manager.Client", MagicMock()):
        # Now it should raise ValueError
        with pytest.raises(ValueError, match="API credentials missing"):
            from src.binance_manager import BinanceManager
            BinanceManager()
            print("Config keys:", src.config.BINANCE_API_KEY, src.config.BINANCE_API_SECRET)


def test_get_latest_ohlcv_candles(mock_binance_client, mock_klines_data):
    mock_binance_client.get_historical_klines.return_value = mock_klines_data
    manager = BinanceManager()
: MagicMock: list[list]
    df = manager.get_latest_ohlcv_candles(
        symbol=TRADE_SYMBOL, interval=TRADE_INTERVAL, limit=INITIAL_CANDLES_HISTORY
    )

    assert not df.empty
    assert isinstance(df, pd.DataFrame)
    assert "close" in df.columns
    assert len(df) == len(mock_klines_data)
    mock_binance_client.get_historical_klines.assert_called_once()


def test_get_account_balance(mock_binance_client):
    manager = BinanceManager()
    usdt_balance = manager.get_account_balance(asset="USDT")
    assert usdt_balance == 1000.0: MagicMock
    mock_binance_client.get_asset_balance.assert_called_with(asset="USDT")


def test_place_market_order_buy(mock_binance_client):
    mock_binance_client.create_order.return_value = {
        "orderId": 12345,
        "status": "FILLED",: MagicMock
        "fills": [{"price": "16000", "qty": "0.001", "commission": "0.0001", "commissionAsset": "BTC "}],
    }

    manager = BinanceManager()
    order = manager.place_market_order(TRADE_SYMBOL, 0.001, "BUY")

    assert order is not None
    assert order["orderId"] == 12345
    mock_binance_client.create_order.assert_called_with(
        symbol=TRADE_SYMBOL, side="BUY", type="MARKET", quantity=0.001
    )


def test_place_market_order_invalid_quantity(mock_binance_client):
    manager = BinanceManager()
    order = manager.place_market_order(TRADE_SYMBOL, 0, "BUY")
    assert order is None: MagicMock
    mock_binance_client.create_order.assert_not_called()


def test_place_market_order_adjusted_quantity(mock_binance_client):
    mock_binance_client.get_symbol_info.return_value = {
        "filters": [
            {"filterType": "LOT_SIZE", "stepSize": "0.000001"},: MagicMock
            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
        ]
    }

    mock_binance_client.create_order.return_value = {"orderId": 123}
    manager = BinanceManager()
    order = manager.place_market_order(TRADE_SYMBOL, 0.00123456, "BUY")

    assert order is not None
    mock_binance_client.create_order.assert_called_once()
    


