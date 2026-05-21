from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from app.domain.risk_management import RiskDecision, RiskDecisionStatus
from app.domain.signals import TradeAction
from app.guardrails.trade_limits import TradeLimitConfig, TradeLimitState
from app.models.trade import Trade
from app.services.trade_execution_service import (
    TradeExecutionRequest,
    TradeExecutionService,
    execute_market_trade,
)


def make_risk_decision(
    *,
    status: RiskDecisionStatus = RiskDecisionStatus.APPROVED,
    action: TradeAction = TradeAction.BUY,
    quantity: float = 0.01,
    notional_value: float = 680.0,
    reason: str = "buy_approved",
) -> RiskDecision:
    return RiskDecision(
        status=status,
        action=action,
        quantity=quantity,
        notional_value=notional_value,
        reason=reason,
        metadata={"confidence": 0.82},
    )


def make_limit_config() -> TradeLimitConfig:
    return TradeLimitConfig(
        allowed_symbols=("BTCUSDT",),
        max_order_notional=10_000.0,
        max_daily_notional=25_000.0,
        min_seconds_between_trades=0,
    )


def test_execute_market_trade_rejects_unapproved_risk_decision() -> None:
    exchange = MagicMock()
    service = TradeExecutionService(exchange=exchange)
    request = TradeExecutionRequest(
        symbol="BTCUSDT",
        price=68_000.0,
        risk_decision=make_risk_decision(
            status=RiskDecisionStatus.REJECTED,
            quantity=0.0,
            notional_value=0.0,
            reason="daily_loss_limit_reached",
        ),
    )

    result = service.execute_market_trade(request)

    assert result.success is False
    assert result.status == "REJECTED"
    assert result.reason == "risk_decision_not_approved"
    exchange.place_market_order.assert_not_called()


def test_execute_market_trade_blocks_when_trade_limits_fail() -> None:
    exchange = MagicMock()
    service = TradeExecutionService(exchange=exchange)
    request = TradeExecutionRequest(
        symbol="BTCUSDT",
        price=68_000.0,
        risk_decision=make_risk_decision(quantity=1.0, notional_value=68_000.0),
        limit_config=TradeLimitConfig(
            allowed_symbols=("BTCUSDT",),
            max_order_notional=100.0,
            min_seconds_between_trades=0,
        ),
    )

    result = service.execute_market_trade(request)

    assert result.success is False
    assert result.status == "BLOCKED"
    assert result.reason == "max_order_notional_exceeded"
    assert result.limit_decision is not None
    exchange.place_market_order.assert_not_called()


def test_execute_market_trade_places_order_and_returns_exchange_result() -> None:
    exchange = MagicMock()
    exchange.place_market_order.return_value = {
        "status": "FILLED",
        "executedQty": "0.02",
        "price": "68100",
        "orderId": "order-123",
    }
    request = TradeExecutionRequest(
        symbol="BTCUSDT",
        price=68_000.0,
        risk_decision=make_risk_decision(quantity=0.02, notional_value=1_360.0),
        limit_state=TradeLimitState(),
        limit_config=make_limit_config(),
        metadata={"source": "test"},
    )

    result = execute_market_trade(request, exchange=exchange)

    assert result.success is True
    assert result.status == "FILLED"
    assert result.reason == "trade_executed"
    assert result.quantity == 0.02
    assert result.price == 68_100.0
    assert result.trade_id is None
    exchange.place_market_order.assert_called_once_with(
        symbol="BTCUSDT",
        quantity=0.02,
        side="BUY",
    )


def test_execute_market_trade_returns_failed_when_exchange_rejects_order() -> None:
    exchange = MagicMock()
    exchange.place_market_order.return_value = None
    service = TradeExecutionService(exchange=exchange)
    request = TradeExecutionRequest(
        symbol="BTCUSDT",
        price=68_000.0,
        risk_decision=make_risk_decision(),
        limit_config=make_limit_config(),
    )

    result = service.execute_market_trade(request)

    assert result.success is False
    assert result.status == "FAILED"
    assert result.reason == "exchange_order_failed"


def test_execute_market_trade_persists_trade_when_db_session_is_available() -> None:
    exchange = MagicMock()
    exchange.place_market_order.return_value = {
        "status": "FILLED",
        "executedQty": "0.01",
        "price": "68050",
        "orderId": "order-456",
        "commission": "0.25",
        "commissionAsset": "USDT",
    }
    db = MagicMock()

    def refresh_trade(trade: Trade) -> None:
        trade.id = 42

    db.refresh.side_effect = refresh_trade
    service = TradeExecutionService(exchange=exchange, db=db)
    request = TradeExecutionRequest(
        symbol="BTCUSDT",
        price=68_000.0,
        risk_decision=make_risk_decision(),
        limit_config=make_limit_config(),
        metadata={"confidence": 0.82},
    )

    result = service.execute_market_trade(request)

    assert result.trade_id == 42
    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.refresh.assert_called_once()

    persisted_trade = db.add.call_args.args[0]
    assert isinstance(persisted_trade, Trade)
    assert persisted_trade.symbol == "BTCUSDT"
    assert persisted_trade.side == "BUY"
    assert persisted_trade.quantity == Decimal("0.01")
    assert persisted_trade.price == Decimal("68050")
    assert persisted_trade.confidence == Decimal("0.82")
    assert persisted_trade.order_id == "order-456"


def test_execute_market_trade_rolls_back_when_persistence_fails() -> None:
    exchange = MagicMock()
    exchange.place_market_order.return_value = {
        "status": "FILLED",
        "executedQty": "0.01",
        "price": "68050",
    }
    db = MagicMock()
    db.commit.side_effect = RuntimeError("database unavailable")
    service = TradeExecutionService(exchange=exchange, db=db)
    request = TradeExecutionRequest(
        symbol="BTCUSDT",
        price=68_000.0,
        risk_decision=make_risk_decision(),
        limit_config=make_limit_config(),
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        service.execute_market_trade(request)

    db.rollback.assert_called_once()
