from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from app.core.config import TRADE_SYMBOL
from app.domain.risk_management import RiskDecision
from app.domain.signals import TradeAction
from app.utils.validators import normalize_symbol


class LimitDecisionStatus(StrEnum):
    """Outcome of guardrail validation."""

    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class TradeLimitConfig:
    """Operational limits applied after strategy and risk approval."""

    allowed_symbols: tuple[str, ...] = (TRADE_SYMBOL,)
    allowed_actions: tuple[TradeAction, ...] = (TradeAction.BUY, TradeAction.SELL)
    max_order_notional: float = 2_500.0
    max_daily_notional: float = 10_000.0
    max_daily_trades: int = 10
    min_seconds_between_trades: int = 30
    require_risk_approval: bool = True

    def __post_init__(self) -> None:
        if not self.allowed_symbols:
            raise ValueError("allowed_symbols must contain at least one symbol.")
        if not self.allowed_actions:
            raise ValueError("allowed_actions must contain at least one action.")
        if self.max_order_notional <= 0:
            raise ValueError("max_order_notional must be greater than 0.")
        if self.max_daily_notional <= 0:
            raise ValueError("max_daily_notional must be greater than 0.")
        if self.max_daily_trades < 1:
            raise ValueError("max_daily_trades must be at least 1.")
        if self.min_seconds_between_trades < 0:
            raise ValueError("min_seconds_between_trades must be greater than or equal to 0.")


@dataclass(frozen=True, slots=True)
class TradeLimitState:
    """Current trading activity used by guardrails."""

    trades_today: int = 0
    notional_traded_today: float = 0.0
    last_trade_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class TradeLimitDecision:
    """Final guardrail decision for an order candidate."""

    status: LimitDecisionStatus
    allowed: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


def is_symbol_allowed(symbol: str, config: TradeLimitConfig | None = None) -> bool:
    """Return True when the symbol is in the configured allow-list."""
    limit_config = config or TradeLimitConfig()
    allowed_symbols = {normalize_symbol(allowed) for allowed in limit_config.allowed_symbols}
    return normalize_symbol(symbol) in allowed_symbols


def has_trade_cooldown_elapsed(
    last_trade_at: datetime | None,
    *,
    now: datetime | None = None,
    config: TradeLimitConfig | None = None,
) -> bool:
    """Return True when enough time has passed since the last trade."""
    if last_trade_at is None:
        return True

    limit_config = config or TradeLimitConfig()
    current_time = now or datetime.now(UTC)

    if last_trade_at.tzinfo is None:
        last_trade_at = last_trade_at.replace(tzinfo=UTC)

    elapsed = current_time - last_trade_at
    return elapsed >= timedelta(seconds=limit_config.min_seconds_between_trades)


def check_trade_limits(
    *,
    symbol: str,
    action: TradeAction,
    quantity: float,
    price: float,
    state: TradeLimitState | None = None,
    config: TradeLimitConfig | None = None,
    risk_decision: RiskDecision | None = None,
    now: datetime | None = None,
) -> TradeLimitDecision:
    """Validate an order candidate against operational trade limits."""
    limit_config = config or TradeLimitConfig()
    limit_state = state or TradeLimitState()
    normalized_symbol = normalize_symbol(symbol)
    normalized_quantity = float(quantity)
    normalized_price = float(price)

    if normalized_quantity <= 0:
        return TradeLimitDecision(
            status=LimitDecisionStatus.BLOCKED,
            allowed=False,
            reason="quantity_must_be_positive",
            metadata={"quantity": normalized_quantity},
        )

    if normalized_price <= 0:
        return TradeLimitDecision(
            status=LimitDecisionStatus.BLOCKED,
            allowed=False,
            reason="price_must_be_positive",
            metadata={"price": normalized_price},
        )

    if action == TradeAction.HOLD:
        return TradeLimitDecision(
            status=LimitDecisionStatus.SKIPPED,
            allowed=False,
            reason="hold_action_has_no_order",
        )

    if action not in limit_config.allowed_actions:
        return TradeLimitDecision(
            status=LimitDecisionStatus.BLOCKED,
            allowed=False,
            reason="action_not_allowed",
            metadata={"action": action.value},
        )

    if not is_symbol_allowed(normalized_symbol, config=limit_config):
        return TradeLimitDecision(
            status=LimitDecisionStatus.BLOCKED,
            allowed=False,
            reason="symbol_not_allowed",
            metadata={
                "symbol": normalized_symbol,
                "allowed_symbols": list(limit_config.allowed_symbols),
            },
        )

    if limit_config.require_risk_approval and (risk_decision is None or not risk_decision.approved):
        return TradeLimitDecision(
            status=LimitDecisionStatus.BLOCKED,
            allowed=False,
            reason="risk_approval_required",
            metadata={
                "risk_reason": risk_decision.reason if risk_decision is not None else None,
            },
        )

    order_notional = normalized_quantity * normalized_price

    if order_notional > limit_config.max_order_notional:
        return TradeLimitDecision(
            status=LimitDecisionStatus.BLOCKED,
            allowed=False,
            reason="max_order_notional_exceeded",
            metadata={
                "order_notional": order_notional,
                "max_order_notional": limit_config.max_order_notional,
            },
        )

    projected_daily_notional = limit_state.notional_traded_today + order_notional
    if projected_daily_notional > limit_config.max_daily_notional:
        return TradeLimitDecision(
            status=LimitDecisionStatus.BLOCKED,
            allowed=False,
            reason="max_daily_notional_exceeded",
            metadata={
                "projected_daily_notional": projected_daily_notional,
                "max_daily_notional": limit_config.max_daily_notional,
            },
        )

    if limit_state.trades_today >= limit_config.max_daily_trades:
        return TradeLimitDecision(
            status=LimitDecisionStatus.BLOCKED,
            allowed=False,
            reason="max_daily_trades_exceeded",
            metadata={
                "trades_today": limit_state.trades_today,
                "max_daily_trades": limit_config.max_daily_trades,
            },
        )

    if not has_trade_cooldown_elapsed(
        limit_state.last_trade_at,
        now=now,
        config=limit_config,
    ):
        return TradeLimitDecision(
            status=LimitDecisionStatus.BLOCKED,
            allowed=False,
            reason="trade_cooldown_active",
            metadata={
                "last_trade_at": limit_state.last_trade_at.isoformat()
                if limit_state.last_trade_at is not None
                else None,
                "min_seconds_between_trades": limit_config.min_seconds_between_trades,
            },
        )

    return TradeLimitDecision(
        status=LimitDecisionStatus.ALLOWED,
        allowed=True,
        reason="trade_limits_passed",
        metadata={
            "symbol": normalized_symbol,
            "action": action.value,
            "quantity": normalized_quantity,
            "price": normalized_price,
            "order_notional": order_notional,
            "projected_daily_notional": projected_daily_notional,
            "projected_trades_today": limit_state.trades_today + 1,
        },
    )


def check_risk_decision_limits(
    *,
    symbol: str,
    price: float,
    risk_decision: RiskDecision,
    state: TradeLimitState | None = None,
    config: TradeLimitConfig | None = None,
    now: datetime | None = None,
) -> TradeLimitDecision:
    """Validate a RiskDecision using its approved action and quantity."""
    return check_trade_limits(
        symbol=symbol,
        action=risk_decision.action,
        quantity=risk_decision.quantity,
        price=price,
        state=state,
        config=config,
        risk_decision=risk_decision,
        now=now,
    )
