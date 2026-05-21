"""Tests for the trade limits guardrails that enforce operational risk management."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain.risk_management import RiskDecision, RiskDecisionStatus
from app.domain.signals import TradeAction
from app.guardrails.trade_limits import (
    LimitDecisionStatus,
    TradeLimitConfig,
    TradeLimitState,
    check_risk_decision_limits,
    check_trade_limits,
    has_trade_cooldown_elapsed,
    is_symbol_allowed,
)


def approved_risk_decision(
    action: TradeAction = TradeAction.BUY, quantity: float = 1.0
) -> RiskDecision:
    """Create an approved risk decision for guardrail tests."""
    return RiskDecision(
        status=RiskDecisionStatus.APPROVED,
        action=action,
        quantity=quantity,
        notional_value=quantity * 100,
        reason="risk_approved",
    )


def rejected_risk_decision(action: TradeAction = TradeAction.BUY) -> RiskDecision:
    """Create a rejected risk decision for guardrail tests."""
    return RiskDecision(
        status=RiskDecisionStatus.REJECTED,
        action=action,
        quantity=0.0,
        notional_value=0.0,
        reason="risk_rejected",
    )


def test_trade_limit_config_rejects_invalid_values() -> None:
    """Operational trade limits should reject unusable configurations."""
    with pytest.raises(ValueError, match="allowed_symbols"):
        TradeLimitConfig(allowed_symbols=())

    with pytest.raises(ValueError, match="allowed_actions"):
        TradeLimitConfig(allowed_actions=())

    with pytest.raises(ValueError, match="max_order_notional"):
        TradeLimitConfig(max_order_notional=0)

    with pytest.raises(ValueError, match="max_daily_notional"):
        TradeLimitConfig(max_daily_notional=0)

    with pytest.raises(ValueError, match="max_daily_trades"):
        TradeLimitConfig(max_daily_trades=0)

    with pytest.raises(ValueError, match="min_seconds_between_trades"):
        TradeLimitConfig(min_seconds_between_trades=-1)


def test_is_symbol_allowed_normalizes_symbols() -> None:
    """Symbol allow-list checks should be case/spacing tolerant."""
    config = TradeLimitConfig(allowed_symbols=("BTCUSDT", "ETHUSDT"))

    assert is_symbol_allowed(" btcusdt ", config=config) is True
    assert is_symbol_allowed("ethusdt", config=config) is True
    assert is_symbol_allowed("adausdt", config=config) is False


def test_has_trade_cooldown_elapsed_handles_none_and_naive_timestamps() -> None:
    """Cooldown checks should allow first trades and normalize naive timestamps."""
    config = TradeLimitConfig(min_seconds_between_trades=30)
    now = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)

    assert has_trade_cooldown_elapsed(None, now=now, config=config) is True
    assert (
        has_trade_cooldown_elapsed(
            datetime(2026, 5, 16, 11, 59, 0),
            now=now,
            config=config,
        )
        is True
    )
    assert (
        has_trade_cooldown_elapsed(
            now - timedelta(seconds=10),
            now=now,
            config=config,
        )
        is False
    )


@pytest.mark.parametrize(
    ("quantity", "price", "reason"),
    [
        (0, 100, "quantity_must_be_positive"),
        (1, 0, "price_must_be_positive"),
    ],
)
def test_check_trade_limits_rejects_invalid_quantity_or_price(
    quantity: float,
    price: float,
    reason: str,
) -> None:
    """Raw order values must be positive before deeper guardrails run."""
    decision = check_trade_limits(
        symbol="BTCUSDT",
        action=TradeAction.BUY,
        quantity=quantity,
        price=price,
        config=TradeLimitConfig(require_risk_approval=False),
    )

    assert decision.status == LimitDecisionStatus.BLOCKED
    assert decision.allowed is False
    assert decision.reason == reason


def test_check_trade_limits_skips_hold_action() -> None:
    """HOLD is intentionally skipped because it does not create an order."""
    decision = check_trade_limits(
        symbol="BTCUSDT",
        action=TradeAction.HOLD,
        quantity=1,
        price=100,
        config=TradeLimitConfig(require_risk_approval=False),
    )

    assert decision.status == LimitDecisionStatus.SKIPPED
    assert decision.allowed is False
    assert decision.reason == "hold_action_has_no_order"


def test_check_trade_limits_blocks_disallowed_action_and_symbol() -> None:
    """Action and symbol allow-lists should block unsupported orders."""
    action_decision = check_trade_limits(
        symbol="BTCUSDT",
        action=TradeAction.SELL,
        quantity=1,
        price=100,
        config=TradeLimitConfig(
            allowed_symbols=("BTCUSDT",),
            allowed_actions=(TradeAction.BUY,),
            require_risk_approval=False,
        ),
    )
    symbol_decision = check_trade_limits(
        symbol="ETHUSDT",
        action=TradeAction.BUY,
        quantity=1,
        price=100,
        config=TradeLimitConfig(allowed_symbols=("BTCUSDT",), require_risk_approval=False),
    )

    assert action_decision.status == LimitDecisionStatus.BLOCKED
    assert action_decision.reason == "action_not_allowed"
    assert symbol_decision.status == LimitDecisionStatus.BLOCKED
    assert symbol_decision.reason == "symbol_not_allowed"
    assert symbol_decision.metadata["symbol"] == "ETHUSDT"


def test_check_trade_limits_requires_approved_risk_decision_by_default() -> None:
    """Risk approval should be required unless explicitly disabled."""
    missing_risk = check_trade_limits(
        symbol="BTCUSDT",
        action=TradeAction.BUY,
        quantity=1,
        price=100,
    )
    rejected_risk = check_trade_limits(
        symbol="BTCUSDT",
        action=TradeAction.BUY,
        quantity=1,
        price=100,
        risk_decision=rejected_risk_decision(),
    )

    assert missing_risk.reason == "risk_approval_required"
    assert missing_risk.metadata["risk_reason"] is None
    assert rejected_risk.reason == "risk_approval_required"
    assert rejected_risk.metadata["risk_reason"] == "risk_rejected"


def test_check_trade_limits_blocks_order_notional_and_daily_notional() -> None:
    """Per-order and daily notional caps should both be enforced."""
    risk_decision = approved_risk_decision()
    order_cap = check_trade_limits(
        symbol="BTCUSDT",
        action=TradeAction.BUY,
        quantity=3,
        price=100,
        config=TradeLimitConfig(max_order_notional=250),
        risk_decision=risk_decision,
    )
    daily_cap = check_trade_limits(
        symbol="BTCUSDT",
        action=TradeAction.BUY,
        quantity=1,
        price=100,
        state=TradeLimitState(notional_traded_today=950),
        config=TradeLimitConfig(max_order_notional=500, max_daily_notional=1_000),
        risk_decision=risk_decision,
    )

    assert order_cap.reason == "max_order_notional_exceeded"
    assert order_cap.metadata["order_notional"] == 300
    assert daily_cap.reason == "max_daily_notional_exceeded"
    assert daily_cap.metadata["projected_daily_notional"] == 1050


def test_check_trade_limits_blocks_daily_trade_count_and_cooldown() -> None:
    """Daily count and cooldown limits should prevent overtrading."""
    risk_decision = approved_risk_decision()
    now = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)
    count_limit = check_trade_limits(
        symbol="BTCUSDT",
        action=TradeAction.BUY,
        quantity=1,
        price=100,
        state=TradeLimitState(trades_today=3),
        config=TradeLimitConfig(max_daily_trades=3),
        risk_decision=risk_decision,
        now=now,
    )
    cooldown = check_trade_limits(
        symbol="BTCUSDT",
        action=TradeAction.BUY,
        quantity=1,
        price=100,
        state=TradeLimitState(last_trade_at=now - timedelta(seconds=10)),
        config=TradeLimitConfig(min_seconds_between_trades=30),
        risk_decision=risk_decision,
        now=now,
    )

    assert count_limit.reason == "max_daily_trades_exceeded"
    assert cooldown.reason == "trade_cooldown_active"
    assert cooldown.metadata["min_seconds_between_trades"] == 30


def test_check_trade_limits_allows_valid_order() -> None:
    """A valid order should pass guardrail checks and include projected metadata."""
    now = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)
    state = TradeLimitState(trades_today=1, notional_traded_today=200)
    config = TradeLimitConfig(
        allowed_symbols=("BTCUSDT",),
        max_order_notional=500,
        max_daily_notional=1_000,
        max_daily_trades=5,
        min_seconds_between_trades=0,
    )

    decision = check_trade_limits(
        symbol="btcusdt",
        action=TradeAction.BUY,
        quantity=2,
        price=100,
        state=state,
        config=config,
        risk_decision=approved_risk_decision(),
        now=now,
    )

    assert decision.status == LimitDecisionStatus.ALLOWED
    assert decision.allowed is True
    assert decision.reason == "trade_limits_passed"
    assert decision.metadata["symbol"] == "BTCUSDT"
    assert decision.metadata["action"] == "BUY"
    assert decision.metadata["order_notional"] == 200
    assert decision.metadata["projected_daily_notional"] == 400
    assert decision.metadata["projected_trades_today"] == 2


def test_check_risk_decision_limits_uses_risk_decision_action_and_quantity() -> None:
    """RiskDecision validation should reuse the approved action and quantity."""
    risk_decision = approved_risk_decision(action=TradeAction.SELL, quantity=0.5)

    decision = check_risk_decision_limits(
        symbol="BTCUSDT",
        price=200,
        risk_decision=risk_decision,
        config=TradeLimitConfig(min_seconds_between_trades=0),
    )

    assert decision.allowed is True
    assert decision.metadata["action"] == "SELL"
    assert decision.metadata["quantity"] == 0.5
    assert decision.metadata["order_notional"] == 100
