from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session

from app.domain.risk_management import RiskDecision
from app.domain.signals import TradeAction
from app.guardrails.trade_limits import (
    TradeLimitConfig,
    TradeLimitDecision,
    TradeLimitState,
    check_risk_decision_limits,
)
from app.models.trade import Trade
from app.services.binance_service import BinanceManager

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TradeExecutionRequest:
    """Input required to execute an approved trade."""

    symbol: str
    price: float
    risk_decision: RiskDecision
    limit_state: TradeLimitState | None = None
    limit_config: TradeLimitConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TradeExecutionResult:
    """Result returned after attempting trade execution."""

    success: bool
    action: TradeAction
    symbol: str
    quantity: float
    price: float
    status: str
    reason: str
    order_response: dict[str, Any] | None = None
    trade_id: int | None = None
    limit_decision: TradeLimitDecision | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TradeExecutionService:
    """Coordinate guardrails, exchange execution, and trade persistence."""

    def __init__(
        self,
        *,
        exchange: BinanceManager | None = None,
        db: Session | None = None,
    ) -> None:
        self._exchange = exchange
        self._db = db

    @property
    def exchange(self) -> BinanceManager:
        """Lazily create the exchange client to keep tests/imports lightweight."""
        if self._exchange is None:
            self._exchange = BinanceManager()
        return self._exchange

    def execute_market_trade(self, request: TradeExecutionRequest) -> TradeExecutionResult:
        """Execute a market trade after risk and limit checks."""
        risk_decision = request.risk_decision

        if not risk_decision.approved:
            return TradeExecutionResult(
                success=False,
                action=risk_decision.action,
                symbol=request.symbol,
                quantity=0.0,
                price=request.price,
                status="REJECTED",
                reason="risk_decision_not_approved",
                metadata={"risk_reason": risk_decision.reason},
            )

        limit_decision = check_risk_decision_limits(
            symbol=request.symbol,
            price=request.price,
            risk_decision=risk_decision,
            state=request.limit_state,
            config=request.limit_config,
        )

        if not limit_decision.allowed:
            return TradeExecutionResult(
                success=False,
                action=risk_decision.action,
                symbol=request.symbol,
                quantity=risk_decision.quantity,
                price=request.price,
                status=limit_decision.status.value,
                reason=limit_decision.reason,
                limit_decision=limit_decision,
                metadata=limit_decision.metadata,
            )

        order_response = self.exchange.place_market_order(
            symbol=request.symbol,
            quantity=risk_decision.quantity,
            side=risk_decision.action.value,
        )

        if order_response is None:
            return TradeExecutionResult(
                success=False,
                action=risk_decision.action,
                symbol=request.symbol,
                quantity=risk_decision.quantity,
                price=request.price,
                status="FAILED",
                reason="exchange_order_failed",
                limit_decision=limit_decision,
            )

        status = str(order_response.get("status", "UNKNOWN"))
        executed_quantity = _extract_executed_quantity(order_response, risk_decision.quantity)
        execution_price = _extract_execution_price(order_response, request.price)

        trade_id = self._persist_trade(
            symbol=request.symbol,
            action=risk_decision.action,
            quantity=executed_quantity,
            price=execution_price,
            status=status,
            order_response=order_response,
            metadata={
                **request.metadata,
                "risk_reason": risk_decision.reason,
                "limit_reason": limit_decision.reason,
            },
        )

        LOG.info(
            "Trade execution complete. symbol=%s side=%s quantity=%s status=%s trade_id=%s",
            request.symbol,
            risk_decision.action.value,
            executed_quantity,
            status,
            trade_id,
        )

        return TradeExecutionResult(
            success=status.upper() in {"FILLED", "PARTIALLY_FILLED", "SUCCESS"},
            action=risk_decision.action,
            symbol=request.symbol,
            quantity=executed_quantity,
            price=execution_price,
            status=status,
            reason="trade_executed",
            order_response=order_response,
            trade_id=trade_id,
            limit_decision=limit_decision,
            metadata=request.metadata,
        )

    def _persist_trade(
        self,
        *,
        symbol: str,
        action: TradeAction,
        quantity: float,
        price: float,
        status: str,
        order_response: dict[str, Any],
        metadata: dict[str, Any],
    ) -> int | None:
        """Persist a trade when a database session is available."""
        if self._db is None:
            return None

        trade = Trade(
            symbol=symbol,
            side=action.value,
            quantity=Decimal(str(quantity)),
            price=Decimal(str(price)),
            confidence=_optional_decimal(metadata.get("confidence")),
            status=status,
            order_id=_optional_str(order_response.get("orderId") or order_response.get("clientOrderId")),
            fill_price=_optional_decimal(order_response.get("price")) or Decimal(str(price)),
            commission=_optional_decimal(order_response.get("commission")),
            commission_asset=_optional_str(order_response.get("commissionAsset")),
        )

        try:
            self._db.add(trade)
            self._db.commit()
            self._db.refresh(trade)
            return int(trade.id)
        except Exception:
            LOG.exception("Failed to persist executed trade.")
            self._db.rollback()
            raise


def execute_market_trade(
    request: TradeExecutionRequest,
    *,
    exchange: BinanceManager | None = None,
    db: Session | None = None,
) -> TradeExecutionResult:
    """Convenience function for one-off market trade execution."""
    return TradeExecutionService(exchange=exchange, db=db).execute_market_trade(request)


def _extract_executed_quantity(order_response: dict[str, Any], fallback: float) -> float:
    for key in ("executedQty", "origQty", "qty", "quantity"):
        value = order_response.get(key)
        if value not in (None, ""):
            return float(value)
    return float(fallback)


def _extract_execution_price(order_response: dict[str, Any], fallback: float) -> float:
    raw_price = order_response.get("price")
    if raw_price not in (None, ""):
        price = float(raw_price)
        if price > 0:
            return price
    return float(fallback)


def _optional_decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    return Decimal(str(value))


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
