from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NotRequired, Required, TypedDict

from app.domain.risk_management import AccountState, RiskConfig, RiskDecision
from app.domain.signals import TradeAction
from app.domain.trading_strategy import StrategyConfig, StrategyDecision
from app.guardrails.trade_limits import (
    TradeLimitConfig,
    TradeLimitDecision,
    TradeLimitState,
)
from app.services.trade_execution_service import TradeExecutionResult


class TradingAgentRunStatus(StrEnum):
    """High-level outcome of a trading-agent run."""

    HOLD = "HOLD"
    NEEDS_APPROVAL = "NEEDS_APPROVAL"
    BLOCKED = "BLOCKED"
    DRY_RUN = "DRY_RUN"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class HumanApproval:
    """Human approval decision for a proposed trade."""

    approved: bool | None = None
    reviewer: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    """Controls when the agent must pause for human approval."""

    require_for_live_trade: bool = True
    require_for_notional_over: float = 1_000.0
    require_for_confidence_below: float | None = None

    def __post_init__(self) -> None:
        if self.require_for_notional_over < 0:
            raise ValueError("require_for_notional_over must be greater than or equal to 0.")
        if self.require_for_confidence_below is not None and not (
            0 <= self.require_for_confidence_below <= 1
        ):
            raise ValueError("require_for_confidence_below must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class TradingAgentRequest:
    """Input required to run the trading-agent decision workflow."""

    symbol: str
    price: float
    model_signal: int | float | str | TradeAction
    model_confidence: float
    account: AccountState
    current_position: int = 0
    requested_quantity: float | None = None
    market_context: dict[str, Any] = field(default_factory=dict)
    research_context: str | None = None
    strategy_config: StrategyConfig | None = None
    risk_config: RiskConfig | None = None
    limit_state: TradeLimitState | None = None
    limit_config: TradeLimitConfig | None = None
    approval_policy: ApprovalPolicy = field(default_factory=ApprovalPolicy)
    human_approval: HumanApproval | None = None
    execute_live: bool = False
    execution_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol must not be blank.")
        if self.price <= 0:
            raise ValueError("price must be greater than 0.")
        if not 0 <= float(self.model_confidence) <= 1:
            raise ValueError("model_confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class TradingAgentResult:
    """Final structured output from the trading agent."""

    status: TradingAgentRunStatus
    action: TradeAction
    symbol: str
    market_summary: str
    signal_explanation: str
    strategy_decision: StrategyDecision
    risk_decision: RiskDecision
    limit_decision: TradeLimitDecision | None
    execution_result: TradeExecutionResult | None = None
    approval_required: bool = False
    approval: HumanApproval | None = None
    operator_rationale: str | None = None
    messages: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class TradingAgentState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    request: Required[TradingAgentRequest]
    messages: Required[list[str]]
    metadata: Required[dict[str, Any]]
    market_summary: NotRequired[str]
    signal_explanation: NotRequired[str]
    strategy_decision: NotRequired[StrategyDecision]
    risk_decision: NotRequired[RiskDecision]
    limit_decision: NotRequired[TradeLimitDecision | None]
    execution_result: NotRequired[TradeExecutionResult | None]
    approval_required: NotRequired[bool]
    approval: NotRequired[HumanApproval | None]
    status: NotRequired[TradingAgentRunStatus]


class TradingAgentStateUpdate(TypedDict, total=False):
    """Partial state returned by individual LangGraph nodes."""

    market_summary: str
    signal_explanation: str
    strategy_decision: StrategyDecision
    risk_decision: RiskDecision
    limit_decision: TradeLimitDecision | None
    execution_result: TradeExecutionResult | None
    approval_required: bool
    approval: HumanApproval | None
    messages: list[str]
    status: TradingAgentRunStatus
    metadata: dict[str, Any]
