from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes import trades
from app.main import app
from app.models.trade import Trade

client = TestClient(app)


def make_trade(*, trade_id: int = 1, symbol: str = "BTCUSDT", side: str = "BUY") -> Trade:
    trade = Trade(
        id=trade_id,
        symbol=symbol,
        side=side,
        quantity=Decimal("0.1"),
        price=Decimal("50000"),
        confidence=Decimal("0.85"),
        status="FILLED",
        order_id=f"order-{trade_id}",
        timestamp=datetime(2026, 5, 16, tzinfo=UTC),
    )
    return trade


class FakeQuery:
    def __init__(self, rows: list[Trade]) -> None:
        self.rows = rows

    def all(self) -> list[Trade]:
        return self.rows

    def filter(self, *args: Any, **kwargs: Any) -> FakeQuery:
        return self

    def first(self) -> Trade | None:
        return self.rows[0] if self.rows else None


class FakeSession:
    def __init__(self, rows: list[Trade] | None = None) -> None:
        self.rows = rows or []
        self.added: list[Trade] = []
        self.deleted: list[Trade] = []
        self.commit_count = 0
        self.closed = False

    def add(self, trade: Trade) -> None:
        self.added.append(trade)
        self.rows.append(trade)

    def commit(self) -> None:
        self.commit_count += 1

    def refresh(self, trade: Trade) -> None:
        trade.id = trade.id or 1

    def query(self, model: type[Trade]) -> FakeQuery:
        assert model is Trade
        return FakeQuery(self.rows)

    def delete(self, trade: Trade) -> None:
        self.deleted.append(trade)
        if trade in self.rows:
            self.rows.remove(trade)

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def override_db(fake_session: FakeSession) -> None:
    def _get_db() -> Iterator[FakeSession]:
        yield fake_session

    app.dependency_overrides[trades.get_db] = _get_db


def test_get_db_closes_session(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = FakeSession()
    monkeypatch.setattr(trades, "SessionLocal", lambda: fake_session)

    db_generator = trades.get_db()
    yielded = next(db_generator)

    assert yielded is fake_session

    with pytest.raises(StopIteration):
        next(db_generator)

    assert fake_session.closed is True


def test_create_trade_endpoint_persists_trade_with_fake_session() -> None:
    fake_session = FakeSession()
    override_db(fake_session)
    payload = {
        "symbol": " btc/usdt ",
        "side": "buy",
        "quantity": "0.1",
        "price": "50000",
    }

    response = client.post("/trades/", json=payload)

    assert response.status_code == 200
    assert response.json()["symbol"] == "BTCUSDT"
    assert response.json()["side"] == "BUY"
    assert fake_session.commit_count == 1
    assert len(fake_session.added) == 1


def test_create_trade_endpoint_validates_payload() -> None:
    fake_session = FakeSession()
    override_db(fake_session)

    response = client.post("/trades/", json={"symbol": "BTCUSDT"})

    assert response.status_code == 422
    assert fake_session.added == []


def test_get_trades_endpoint_returns_list() -> None:
    override_db(FakeSession(rows=[make_trade(trade_id=1), make_trade(trade_id=2, side="SELL")]))

    response = client.get("/trades/")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert [trade["id"] for trade in data] == [1, 2]


def test_get_trade_by_id_returns_trade() -> None:
    override_db(FakeSession(rows=[make_trade(trade_id=7, symbol="ETHUSDT")]))

    response = client.get("/trades/7")

    assert response.status_code == 200
    assert response.json()["id"] == 7
    assert response.json()["symbol"] == "ETHUSDT"


def test_get_trade_by_id_returns_404_when_missing() -> None:
    override_db(FakeSession(rows=[]))

    response = client.get("/trades/999")

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Trade not found"


def test_delete_trade_endpoint_deletes_existing_trade() -> None:
    trade = make_trade(trade_id=3)
    fake_session = FakeSession(rows=[trade])
    override_db(fake_session)

    response = client.delete("/trades/3")

    assert response.status_code == 200
    assert response.json() == {"message": "Trade 3 deleted successfully."}
    assert fake_session.deleted == [trade]
    assert fake_session.commit_count == 1


def test_delete_trade_endpoint_returns_404_when_missing() -> None:
    override_db(FakeSession(rows=[]))

    response = client.delete("/trades/999")

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Trade not found"


def test_invalid_trade_id_type_returns_validation_error() -> None:
    override_db(FakeSession())

    response = client.get("/trades/invalid")

    assert response.status_code == 422


def test_invalid_trade_payload_values_return_validation_error() -> None:
    fake_session = FakeSession()
    override_db(fake_session)
    payload = {
        "symbol": "BTCUSDT",
        "side": "HOLD",
        "quantity": "-0.1",
        "price": "50000",
    }

    response = client.post("/trades/", json=payload)

    assert response.status_code == 422
    assert fake_session.added == []
