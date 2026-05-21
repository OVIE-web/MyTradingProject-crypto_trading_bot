from __future__ import annotations

from typing import Any

from app.agents.agents_schemas import (
    ApprovalPolicy,
    HumanApproval,
    TradingAgentRequest,
)
from app.domain.risk_management import (
    RiskDecision,
    RiskDecisionStatus,
    review_trade,
)
from app.domain.signals import TradeAction, action_from_signal
from app.domain.trading_strategy import StrategyDecision, decide_from_model_signal
from app.guardrails.trade_limits import (
    LimitDecisionStatus,
    TradeLimitDecision,
    check_risk_decision_limits,
)
from app.services.trade_execution_service import (
    TradeExecutionRequest,
    TradeExecutionResult,
    TradeExecutionService,
)


def summarize_market_context(request: TradingAgentRequest) -> str:
    """Create a concise, deterministic market summary for the agent run."""
    context = request.market_context
    parts = [f"{request.symbol.upper()} is trading at {request.price:.8g}."]

    rsi = _optional_float(context.get("rsi"))
    if rsi is not None:
        if rsi <= 30:
            parts.append(f"RSI is oversold at {rsi:.2f}.")
        elif rsi >= 70:
            parts.append(f"RSI is overbought at {rsi:.2f}.")
        else:
            parts.append(f"RSI is neutral at {rsi:.2f}.")

    trend = context.get("trend")
    if trend:
        parts.append(f"Trend context: {trend}.")

    volatility = _optional_float(context.get("volatility"))
    if volatility is not None:
        parts.append(f"Volatility estimate is {volatility:.4g}.")

    if request.research_context:
        parts.append(f"Research context available: {request.research_context[:240]}")

    return " ".join(parts)


def explain_model_signal(request: TradingAgentRequest) -> str:
    """Explain the raw model signal in plain language without using an LLM."""
    action = action_from_signal(request.model_signal)
    confidence_pct = request.model_confidence * 100

    if action == TradeAction.HOLD:
        return f"Model recommends HOLD with {confidence_pct:.2f}% confidence."

    return (
        f"Model recommends {action.value} for {request.symbol.upper()} "
        f"with {confidence_pct:.2f}% confidence."
    )


def propose_strategy_decision(request: TradingAgentRequest) -> StrategyDecision:
    """Convert the model signal into the canonical strategy decision."""
    return decide_from_model_signal(
        request.model_signal,
        request.model_confidence,
        config=request.strategy_config,
        current_position=request.current_position,
        metadata={
            "symbol": request.symbol.upper(),
            "market_context": request.market_context,
        },
    )


def review_strategy_risk(
    request: TradingAgentRequest,
    strategy_decision: StrategyDecision,
) -> RiskDecision:
    """Run the risk layer for the strategy decision."""
    return review_trade(
        strategy_decision,
        request.price,
        request.account,
        config=request.risk_config,
        requested_quantity=request.requested_quantity,
    )


def requires_human_approval(
    *,
    request: TradingAgentRequest,
    strategy_decision: StrategyDecision,
    risk_decision: RiskDecision,
    policy: ApprovalPolicy | None = None,
) -> bool:
    """Return True when this proposed action should pause for a human."""
    approval_policy = policy or request.approval_policy

    if not strategy_decision.is_actionable or not risk_decision.approved:
        return False

    if request.execute_live and approval_policy.require_for_live_trade:
        return True

    if risk_decision.notional_value >= approval_policy.require_for_notional_over:
        return True

    if (
        approval_policy.require_for_confidence_below is not None
        and strategy_decision.confidence < approval_policy.require_for_confidence_below
    ):
        return True

    return False


def resolve_human_approval(
    *,
    request: TradingAgentRequest,
    approval_required: bool,
) -> HumanApproval | None:
    """Return the supplied approval decision when one is needed."""
    if not approval_required:
        return request.human_approval

    return request.human_approval or HumanApproval(
        approved=None,
        reason="human_approval_required",
    )


def apply_guardrails(
    *,
    request: TradingAgentRequest,
    strategy_decision: StrategyDecision,
    risk_decision: RiskDecision,
    approval_required: bool,
    approval: HumanApproval | None,
) -> TradeLimitDecision:
    """Apply operational guardrails after strategy, risk, and approval checks."""
    if strategy_decision.action == TradeAction.HOLD:
        return TradeLimitDecision(
            status=LimitDecisionStatus.SKIPPED,
            allowed=False,
            reason="strategy_decision_is_hold",
            metadata={"strategy_reason": strategy_decision.reason},
        )

    if not risk_decision.approved:
        return TradeLimitDecision(
            status=LimitDecisionStatus.BLOCKED,
            allowed=False,
            reason="risk_decision_not_approved",
            metadata={"risk_reason": risk_decision.reason},
        )

    if approval_required and approval is not None and approval.approved is not True:
        return TradeLimitDecision(
            status=LimitDecisionStatus.BLOCKED,
            allowed=False,
            reason="human_approval_required",
            metadata={
                "approval_status": approval.approved,
                "approval_reason": approval.reason,
            },
        )

    return check_risk_decision_limits(
        symbol=request.symbol,
        price=request.price,
        risk_decision=risk_decision,
        state=request.limit_state,
        config=request.limit_config,
    )


def execute_if_approved(
    *,
    request: TradingAgentRequest,
    risk_decision: RiskDecision,
    limit_decision: TradeLimitDecision,
    execution_service: TradeExecutionService | None = None,
) -> TradeExecutionResult | None:
    """Execute the trade only when live mode and all approvals are satisfied."""
    if not request.execute_live or not limit_decision.allowed:
        return None

    service = execution_service or TradeExecutionService()
    return service.execute_market_trade(
        TradeExecutionRequest(
            symbol=request.symbol,
            price=request.price,
            risk_decision=risk_decision,
            limit_state=request.limit_state,
            limit_config=request.limit_config,
            metadata={
                **request.execution_metadata,
                "confidence": risk_decision.metadata.get("confidence"),
            },
        )
    )


def status_message(reason: str, metadata: dict[str, Any] | None = None) -> str:
    """Create a compact agent message for audit trails and tests."""
    if not metadata:
        return reason

    details = ", ".join(f"{key}={value}" for key, value in sorted(metadata.items()))
    return f"{reason} ({details})"


def skipped_risk_decision(action: TradeAction, reason: str) -> RiskDecision:
    """Create a skipped risk decision for non-actionable paths."""
    return RiskDecision(
        status=RiskDecisionStatus.SKIPPED,
        action=action,
        quantity=0.0,
        notional_value=0.0,
        reason=reason,
    )


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
