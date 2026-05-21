"""Unit tests for risk management logic, including position sizing, stop-loss/take-profit calculations, and trade review decisions."""

from __future__ import annotations

import pytest

from app.domain.risk_management import (
    AccountState,
    PositionSize,
    RiskConfig,
    RiskDecisionStatus,
    calculate_drawdown_pct,
    calculate_position_size,
    calculate_stop_loss_price,
    calculate_take_profit_price,
    is_daily_loss_limit_reached,
    is_drawdown_limit_reached,
    review_trade,
)
from app.domain.signals import TradeAction
from app.domain.trading_strategy import StrategyDecision


def make_decision(action: TradeAction, reason: str = "unit_test") -> StrategyDecision:
    """Create a lightweight strategy decision for risk tests."""
    signal = 1 if action == TradeAction.BUY else -1 if action == TradeAction.SELL else 0
    return StrategyDecision(action=action, signal=signal, confidence=0.95, reason=reason)


def test_risk_config_rejects_invalid_values() -> None:
    """Risk config should reject impossible balances and ratios."""
    with pytest.raises(ValueError, match="initial_balance"):
        RiskConfig(initial_balance=0)

    with pytest.raises(ValueError, match="transaction_fee_pct"):
        RiskConfig(transaction_fee_pct=-0.01)

    with pytest.raises(ValueError, match="max_position_pct"):
        RiskConfig(max_position_pct=1.1)


def test_account_state_total_equity_uses_explicit_equity_or_fallback() -> None:
    """Account equity should prefer explicit equity and otherwise include position value."""
    explicit = AccountState(cash_balance=1000, equity=1200, current_position_value=300)
    fallback = AccountState(cash_balance=1000, current_position_value=300)

    assert explicit.total_equity == 1200
    assert fallback.total_equity == 1300


def test_calculate_position_size_caps_by_max_position() -> None:
    """Position sizing should respect maximum account exposure."""
    config = RiskConfig(
        transaction_fee_pct=0.01,
        default_trade_quantity=1,
        max_position_pct=0.25,
    )
    account = AccountState(cash_balance=1000, equity=2000)

    size = calculate_position_size(
        100,
        account,
        config=config,
        requested_quantity=10,
    )

    assert isinstance(size, PositionSize)
    assert size.notional_value == pytest.approx(500)
    assert size.quantity == pytest.approx(5)
    assert size.estimated_fee == pytest.approx(5)
    assert size.total_cost == pytest.approx(505)


def test_calculate_position_size_caps_by_available_cash() -> None:
    """Cash balance should cap orders after estimated transaction fees."""
    config = RiskConfig(
        transaction_fee_pct=0.01,
        default_trade_quantity=10,
        max_position_pct=1.0,
    )
    account = AccountState(cash_balance=50, equity=1000)

    size = calculate_position_size(100, account, config=config)

    assert size.notional_value == pytest.approx(50 / 1.01)
    assert size.total_cost == pytest.approx(50)


def test_calculate_position_size_rejects_invalid_inputs() -> None:
    """Entry price, cash, and requested quantity must be valid."""
    account = AccountState(cash_balance=1000)

    with pytest.raises(ValueError, match="entry_price"):
        calculate_position_size(0, account)

    with pytest.raises(ValueError, match="cash_balance"):
        calculate_position_size(100, AccountState(cash_balance=-1))

    with pytest.raises(ValueError, match="requested_quantity"):
        calculate_position_size(100, account, requested_quantity=0)


def test_stop_loss_and_take_profit_prices_for_buy_and_sell() -> None:
    """Stop-loss and take-profit prices should mirror long and short exposure."""
    config = RiskConfig(stop_loss_pct=0.03, take_profit_pct=0.06)

    assert calculate_stop_loss_price(100, TradeAction.BUY, config=config) == pytest.approx(97)
    assert calculate_take_profit_price(100, TradeAction.BUY, config=config) == pytest.approx(106)
    assert calculate_stop_loss_price(100, TradeAction.SELL, config=config) == pytest.approx(103)
    assert calculate_take_profit_price(100, TradeAction.SELL, config=config) == pytest.approx(94)
    assert calculate_stop_loss_price(100, TradeAction.HOLD, config=config) == pytest.approx(100)
    assert calculate_take_profit_price(100, TradeAction.HOLD, config=config) == pytest.approx(100)


def test_drawdown_calculation_and_limits() -> None:
    """Drawdown helpers should calculate and compare against configured limits."""
    config = RiskConfig(max_drawdown_pct=0.2)
    account = AccountState(cash_balance=800, equity=800, peak_equity=1000)

    assert calculate_drawdown_pct(800, 1000) == pytest.approx(0.2)
    assert is_drawdown_limit_reached(account, config=config) is True

    with pytest.raises(ValueError, match="peak_equity"):
        calculate_drawdown_pct(100, 0)


def test_daily_loss_limit_detection() -> None:
    """Realized losses should stop trading once daily loss threshold is reached."""
    config = RiskConfig(max_daily_loss_pct=0.05)
    account = AccountState(cash_balance=1000, equity=1000, realized_daily_pnl=-50)

    assert is_daily_loss_limit_reached(account, config=config) is True
    assert (
        is_daily_loss_limit_reached(
            AccountState(cash_balance=1000, equity=1000, realized_daily_pnl=-49),
            config=config,
        )
        is False
    )


def test_review_trade_skips_hold_decision() -> None:
    """HOLD decisions should not reach execution."""
    decision = make_decision(TradeAction.HOLD, reason="neutral")

    risk_decision = review_trade(decision, 100, AccountState(cash_balance=1000))

    assert risk_decision.status == RiskDecisionStatus.SKIPPED
    assert risk_decision.approved is False
    assert risk_decision.reason == "strategy_decision_is_hold"
    assert risk_decision.metadata["strategy_reason"] == "neutral"


def test_review_trade_rejects_daily_loss_and_drawdown_limits() -> None:
    """Risk limits should reject otherwise actionable decisions."""
    decision = make_decision(TradeAction.BUY)
    config = RiskConfig(max_daily_loss_pct=0.05, max_drawdown_pct=0.2)

    daily_loss = review_trade(
        decision,
        100,
        AccountState(cash_balance=1000, equity=1000, realized_daily_pnl=-50),
        config=config,
    )
    drawdown = review_trade(
        decision,
        100,
        AccountState(cash_balance=1000, equity=800, peak_equity=1000),
        config=config,
    )

    assert daily_loss.status == RiskDecisionStatus.REJECTED
    assert daily_loss.reason == "daily_loss_limit_reached"
    assert drawdown.status == RiskDecisionStatus.REJECTED
    assert drawdown.reason == "max_drawdown_limit_reached"


def test_review_trade_rejects_sell_without_position() -> None:
    """SELL decisions need an existing position to exit."""
    risk_decision = review_trade(
        make_decision(TradeAction.SELL),
        100,
        AccountState(cash_balance=1000, current_position_qty=0),
    )

    assert risk_decision.status == RiskDecisionStatus.REJECTED
    assert risk_decision.reason == "no_position_to_sell"


def test_review_trade_approves_sell_for_existing_position() -> None:
    """Existing position quantity should define approved SELL size."""
    config = RiskConfig(transaction_fee_pct=0.01)
    risk_decision = review_trade(
        make_decision(TradeAction.SELL),
        100,
        AccountState(cash_balance=1000, current_position_qty=2),
        config=config,
    )

    assert risk_decision.approved is True
    assert risk_decision.status == RiskDecisionStatus.APPROVED
    assert risk_decision.quantity == 2
    assert risk_decision.notional_value == 200
    assert risk_decision.reason == "sell_approved"
    assert risk_decision.metadata["estimated_fee"] == pytest.approx(2)


def test_review_trade_rejects_buy_below_minimum_trade_value() -> None:
    """Very small allowed orders should be rejected before execution."""
    config = RiskConfig(default_trade_quantity=0.01, min_trade_value=10)
    risk_decision = review_trade(
        make_decision(TradeAction.BUY),
        100,
        AccountState(cash_balance=1000),
        config=config,
    )

    assert risk_decision.status == RiskDecisionStatus.REJECTED
    assert risk_decision.reason == "trade_value_below_minimum"
    assert risk_decision.notional_value == pytest.approx(1)


def test_review_trade_approves_buy_with_protective_prices() -> None:
    """Approved BUY decisions should include execution metadata."""
    config = RiskConfig(
        transaction_fee_pct=0.01,
        default_trade_quantity=1,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        min_trade_value=10,
    )

    risk_decision = review_trade(
        make_decision(TradeAction.BUY, reason="model_signal_confirmed"),
        100,
        AccountState(cash_balance=1000, equity=1000),
        config=config,
    )

    assert risk_decision.approved is True
    assert risk_decision.status == RiskDecisionStatus.APPROVED
    assert risk_decision.quantity == pytest.approx(1)
    assert risk_decision.notional_value == pytest.approx(100)
    assert risk_decision.reason == "buy_approved"
    assert risk_decision.metadata["estimated_fee"] == pytest.approx(1)
    assert risk_decision.metadata["total_cost"] == pytest.approx(101)
    assert risk_decision.metadata["stop_loss_price"] == pytest.approx(97)
    assert risk_decision.metadata["take_profit_price"] == pytest.approx(106)
    assert risk_decision.metadata["strategy_reason"] == "model_signal_confirmed"
