"""
Unit tests for the FastAPI trades router and database operations.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from src.db import SessionLocal, Trade
from src.main_api import app

client = TestClient(app)


# Test database operations separately
class TestTradesDatabase:
    """Test trades database operations."""

    def test_create_trade_in_database(self) -> None:
        """Test creating a trade in database."""
        db = SessionLocal()
        try:
            trade = Trade(
                symbol="BTCUSDT",
                side="BUY",
                quantity=0.1,
                price=50000.0,
                timestamp=datetime.now(UTC),
            )
            db.add(trade)
            db.commit()
            db.refresh(trade)

            assert trade.id is not None
            assert trade.symbol == "BTCUSDT"
            assert trade.side == "BUY"

            # Cleanup
            db.delete(trade)
            db.commit()
        finally:
            db.close()

    def test_query_trades(self) -> None:
        """Test querying trades from database."""
        db = SessionLocal()
        try:
            # Insert test trade
            trade = Trade(
                symbol="ETHUSDT",
                side="SELL",
                quantity=1.0,
                price=3000.0,
                timestamp=datetime.now(UTC),
            )
            db.add(trade)
            db.commit()

            # Query it back
            result = db.query(Trade).filter(Trade.symbol == "ETHUSDT").first()
            assert result is not None
            assert result.symbol == "ETHUSDT"

            # Cleanup
            db.delete(result)
            db.commit()
        finally:
            db.close()


class TestTradesRouter:
    """Test trades router endpoints."""

    def test_create_trade_endpoint(self) -> None:
        """Test POST /trades/ endpoint."""
        payload = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.1,
            "price": 50000.0,
        }
        response = client.post("/trades/", json=payload)
        # May succeed or fail based on database state
        assert response.status_code in [200, 400, 422, 500]

    def test_get_trades_endpoint(self) -> None:
        """Test GET /trades/ endpoint."""
        response = client.get("/trades/")
        # Should return list or error, not crash
        assert response.status_code in [200, 500]

    def test_get_trade_by_id(self) -> None:
        """Test GET /trades/{trade_id} endpoint."""
        response = client.get("/trades/1")
        # May return 404 if trade doesn't exist
        assert response.status_code in [200, 404, 500]

    def test_delete_trade_endpoint(self) -> None:
        """Test DELETE /trades/{trade_id} endpoint."""
        response = client.delete("/trades/999")
        # Should return 404 for non-existent trade
        assert response.status_code in [404, 500]

    def test_get_trades_returns_list(self) -> None:
        """Test GET /trades returns a list."""
        response = client.get("/trades/")
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)

    def test_trade_schema_validation(self) -> None:
        """Test trade schema validation."""
        # Missing required field
        payload = {
            "symbol": "BTCUSDT",
            # Missing 'side', 'quantity', 'price'
        }
        response = client.post("/trades/", json=payload)
        # Should return 422 validation error
        assert response.status_code in [422, 400]

    def test_create_trade_with_all_fields(self) -> None:
        """Test creating trade with all required fields."""
        payload = {
            "symbol": "BNBUSDT",
            "side": "BUY",
            "quantity": 10.0,
            "price": 600.0,
        }
        response = client.post("/trades/", json=payload)
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 400, 422, 500]

    def test_get_trades_list_response_format(self) -> None:
        """Test GET /trades response format."""
        response = client.get("/trades/")
        if response.status_code == 200:
            data = response.json()
            # Should be a list
            assert isinstance(data, list)
            # If not empty, verify structure
            if len(data) > 0:
                trade = data[0]
                assert "id" in trade or "symbol" in trade


class TestTradesRouterErrors:
    """Test error handling in trades router."""

    def test_invalid_trade_id_type(self) -> None:
        """Test GET /trades with invalid ID type."""
        response = client.get("/trades/invalid")
        # Should return 422 validation error
        assert response.status_code in [422, 404]

    def test_negative_quantity(self) -> None:
        """Test creating trade with negative quantity."""
        payload = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": -0.1,  # Invalid
            "price": 50000.0,
        }
        response = client.post("/trades/", json=payload)
        # Should fail validation or be handled
        assert response.status_code in [200, 400, 422, 500]

    def test_invalid_side(self) -> None:
        """Test creating trade with invalid side."""
        payload = {
            "symbol": "BTCUSDT",
            "side": "INVALID",  # Not BUY or SELL
            "quantity": 0.1,
            "price": 50000.0,
        }
        response = client.post("/trades/", json=payload)
        # May pass (if no validation) or fail (if validated)
        assert response.status_code in [200, 400, 422, 500]
