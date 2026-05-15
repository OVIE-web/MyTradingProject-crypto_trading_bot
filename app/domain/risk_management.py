from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.core.config import INITIAL_BALANCE, TRADE_QUANTITY, TRANSACTION_FEE_PCT
from app.domain.trading_strategy import StrategyDecision, TradeAction


class RiskDecisionStatus(StrEnum):
    """Risk review outcome."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Risk limits used before a trade is sent to execution."""

    initial_balance: float = float(INITIAL_BALANCE)
    transaction_fee_pct: float = float(TRANSACTION_FEE_PCT)
    default_trade_quantity: float = float(TRADE_QUANTITY)
    max_position_pct: float = 0.25
    max_risk_per_trade_pct: float = 0.02
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.20
    min_trade_value: float = 10.0

    def __post_init__(self) -> None:
        _require_positive("initial_balance", self.initial_balance)
        _require_non_negative("transaction_fee_pct", self.transaction_fee_pct)
        _require_positive("default_trade_quantity", self.default_trade_quantity)
        _require_ratio("max_position_pct", self.max_position_pct)
        _require_ratio("max_risk_per_trade_pct", self.max_risk_per_trade_pct)
        _require_ratio("stop_loss_pct", self.stop_loss_pct)
        _require_ratio("take_profit_pct", self.take_profit_pct)
        _require_ratio("max_daily_loss_pct", self.max_daily_loss_pct)
        _require_ratio("max_drawdown_pct", self.max_drawdown_pct)
        _require_non_negative("min_trade_value", self.min_trade_value)


@dataclass(frozen=True, slots=True)
class AccountState:
    """Current account state used by the risk layer."""

    cash_balance: float
    equity: float | None = None
    current_position_qty: float = 0.0
    current_position_value: float = 0.0
    realized_daily_pnl: float = 0.0
    peak_equity: float | None = None

    @property
    def total_equity(self) -> float:
        """Return current account equity, falling back to cash plus position value."""
        if self.equity is not None:
            return float(self.equity)
        return float(self.cash_balance + self.current_position_value)


@dataclass(frozen=True, slots=True)
class PositionSize:
    """Calculated order size and estimated costs."""

    quantity: float
    notional_value: float
    estimated_fee: float
    total_cost: float


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Final risk approval result."""

    status: RiskDecisionStatus
    action: TradeAction
    quantity: float
    notional_value: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        """Whether the trade is allowed to proceed."""
        return self.status == RiskDecisionStatus.APPROVED


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")


def _require_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to 0.")


def _require_ratio(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def calculate_position_size(
    entry_price: float,
    account: AccountState,
    *,
    config: RiskConfig | None = None,
    requested_quantity: float | None = None,
) -> PositionSize:
    """Calculate a capped position size for a potential entry."""
    risk_config = config or RiskConfig()
    price = float(entry_price)
    _require_positive("entry_price", price)
    _require_non_negative("cash_balance", account.cash_balance)

    max_notional = account.total_equity * risk_config.max_position_pct
    cash_limited_notional = account.cash_balance / (1 + risk_config.transaction_fee_pct)

    if requested_quantity is not None:
        _require_positive("requested_quantity", requested_quantity)
        requested_notional = requested_quantity * price
    else:
        requested_notional = risk_config.default_trade_quantity * price

    notional_value = min(requested_notional, max_notional, cash_limited_notional)
    quantity = notional_value / price
    estimated_fee = notional_value * risk_config.transaction_fee_pct

    return PositionSize(
        quantity=quantity,
        notional_value=notional_value,
        estimated_fee=estimated_fee,
        total_cost=notional_value + estimated_fee,
    )


def calculate_stop_loss_price(
    entry_price: float,
    action: TradeAction,
    *,
    config: RiskConfig | None = None,
) -> float:
    """Calculate stop-loss price for long or short exposure."""
    risk_config = config or RiskConfig()
    price = float(entry_price)
    _require_positive("entry_price", price)

    if action == TradeAction.BUY:
        return price * (1 - risk_config.stop_loss_pct)
    if action == TradeAction.SELL:
        return price * (1 + risk_config.stop_loss_pct)
    return price


def calculate_take_profit_price(
    entry_price: float,
    action: TradeAction,
    *,
    config: RiskConfig | None = None,
) -> float:
    """Calculate take-profit price for long or short exposure."""
    risk_config = config or RiskConfig()
    price = float(entry_price)
    _require_positive("entry_price", price)

    if action == TradeAction.BUY:
        return price * (1 + risk_config.take_profit_pct)
    if action == TradeAction.SELL:
        return price * (1 - risk_config.take_profit_pct)
    return price


def calculate_drawdown_pct(current_equity: float, peak_equity: float) -> float:
    """Calculate percentage drawdown from peak equity."""
    _require_positive("peak_equity", peak_equity)
    _require_non_negative("current_equity", current_equity)
    return max(0.0, (peak_equity - current_equity) / peak_equity)


def is_daily_loss_limit_reached(
    account: AccountState,
    *,
    config: RiskConfig | None = None,
) -> bool:
    """Return True when realized daily loss exceeds the configured limit."""
    risk_config = config or RiskConfig()
    max_daily_loss = account.total_equity * risk_config.max_daily_loss_pct
    return account.realized_daily_pnl <= -max_daily_loss


def is_drawdown_limit_reached(
    account: AccountState,
    *,
    config: RiskConfig | None = None,
) -> bool:
    """Return True when account drawdown exceeds the configured maximum."""
    risk_config = config or RiskConfig()
    peak_equity = account.peak_equity or account.total_equity
    drawdown = calculate_drawdown_pct(account.total_equity, peak_equity)
    return drawdown >= risk_config.max_drawdown_pct


def review_trade(
    decision: StrategyDecision,
    entry_price: float,
    account: AccountState,
    *,
    config: RiskConfig | None = None,
    requested_quantity: float | None = None,
) -> RiskDecision:
    """Approve, reject, or skip a strategy decision based on risk controls."""
    risk_config = config or RiskConfig()
    price = float(entry_price)
    _require_positive("entry_price", price)

    if decision.action == TradeAction.HOLD:
        return RiskDecision(
            status=RiskDecisionStatus.SKIPPED,
            action=decision.action,
            quantity=0.0,
            notional_value=0.0,
            reason="strategy_decision_is_hold",
            metadata={"strategy_reason": decision.reason},
        )

    if is_daily_loss_limit_reached(account, config=risk_config):
        return RiskDecision(
            status=RiskDecisionStatus.REJECTED,
            action=decision.action,
            quantity=0.0,
            notional_value=0.0,
            reason="daily_loss_limit_reached",
            metadata={"realized_daily_pnl": account.realized_daily_pnl},
        )

    if is_drawdown_limit_reached(account, config=risk_config):
        return RiskDecision(
            status=RiskDecisionStatus.REJECTED,
            action=decision.action,
            quantity=0.0,
            notional_value=0.0,
            reason="max_drawdown_limit_reached",
            metadata={
                "equity": account.total_equity,
                "peak_equity": account.peak_equity or account.total_equity,
            },
        )

    if decision.action == TradeAction.SELL:
        if account.current_position_qty <= 0:
            return RiskDecision(
                status=RiskDecisionStatus.REJECTED,
                action=decision.action,
                quantity=0.0,
                notional_value=0.0,
                reason="no_position_to_sell",
                metadata={"strategy_reason": decision.reason},
            )

        quantity = account.current_position_qty
        notional_value = quantity * price
        return RiskDecision(
            status=RiskDecisionStatus.APPROVED,
            action=decision.action,
            quantity=quantity,
            notional_value=notional_value,
            reason="sell_approved",
            metadata={
                "estimated_fee": notional_value * risk_config.transaction_fee_pct,
                "strategy_reason": decision.reason,
            },
        )

    position_size = calculate_position_size(
        price,
        account,
        config=risk_config,
        requested_quantity=requested_quantity,
    )

    if position_size.notional_value < risk_config.min_trade_value:
        return RiskDecision(
            status=RiskDecisionStatus.REJECTED,
            action=decision.action,
            quantity=0.0,
            notional_value=position_size.notional_value,
            reason="trade_value_below_minimum",
            metadata={
                "min_trade_value": risk_config.min_trade_value,
                "strategy_reason": decision.reason,
            },
        )

    return RiskDecision(
        status=RiskDecisionStatus.APPROVED,
        action=decision.action,
        quantity=position_size.quantity,
        notional_value=position_size.notional_value,
        reason="buy_approved",
        metadata={
            "estimated_fee": position_size.estimated_fee,
            "total_cost": position_size.total_cost,
            "stop_loss_price": calculate_stop_loss_price(
                price, decision.action, config=risk_config
            ),
            "take_profit_price": calculate_take_profit_price(
                price,
                decision.action,
                config=risk_config,
            ),
            "strategy_reason": decision.reason,
        },
    )
