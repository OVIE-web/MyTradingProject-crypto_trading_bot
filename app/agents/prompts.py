"""Module defining provider-neutral prompt construction for trading agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from app.agents.agents_schemas import TradingAgentRequest, TradingAgentResult
from app.domain.risk_management import RiskDecision
from app.domain.trading_strategy import StrategyDecision

PromptRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class PromptMessage:
    """Provider-neutral chat message used by optional LLM integrations."""

    role: PromptRole
    content: str


TRADING_AGENT_SYSTEM_PROMPT = """
You are the reasoning assistant for OvieX Quant Engine, an AI-powered crypto
trading system.

Your job is to summarize market context, explain model signals, and produce
operator-facing trade rationales. You must never bypass deterministic strategy,
risk management, human approval, guardrails, or trade execution checks.

Rules:
- Do not invent market data, balances, approvals, or exchange execution results.
- Do not provide financial guarantees or personalized financial advice.
- Treat live execution as high risk.
- If guardrails block a trade, explain the block clearly and do not argue against it.
- Final permission to trade belongs only to deterministic guardrails and execution services.
""".strip()


def build_market_context_prompt(request: TradingAgentRequest) -> tuple[PromptMessage, ...]:
    """Build a prompt that asks an LLM to summarize market context safely."""
    return (
        PromptMessage("system", TRADING_AGENT_SYSTEM_PROMPT),
        PromptMessage(
            "user",
            "\n".join(
                [
                    "Summarize the following market context for a trading operator.",
                    "Keep it concise, factual, and risk-aware.",
                    "",
                    _request_snapshot(request),
                    "",
                    "Market context:",
                    _json_block(request.market_context),
                    "",
                    "Research context:",
                    request.research_context or "No external research context provided.",
                ]
            ),
        ),
    )


def build_signal_explanation_prompt(
    request: TradingAgentRequest,
    *,
    market_summary: str | None = None,
) -> tuple[PromptMessage, ...]:
    """Build a prompt that explains the model signal without changing it."""
    return (
        PromptMessage("system", TRADING_AGENT_SYSTEM_PROMPT),
        PromptMessage(
            "user",
            "\n".join(
                [
                    "Explain the model signal in plain language.",
                    "Do not change the signal, confidence, or final trade decision.",
                    "",
                    _request_snapshot(request),
                    f"Existing market summary: {market_summary or 'Not provided.'}",
                ]
            ),
        ),
    )


def build_human_approval_prompt(
    *,
    request: TradingAgentRequest,
    strategy_decision: StrategyDecision,
    risk_decision: RiskDecision,
) -> tuple[PromptMessage, ...]:
    """Build a prompt for a human-review approval brief."""
    return (
        PromptMessage("system", TRADING_AGENT_SYSTEM_PROMPT),
        PromptMessage(
            "user",
            "\n".join(
                [
                    "Prepare a human approval brief for this proposed trade.",
                    "Include the proposed action, model confidence, risk status, notional value,",
                    "and the exact reason a human should approve, reject, or request more context.",
                    "",
                    _request_snapshot(request),
                    _strategy_snapshot(strategy_decision),
                    _risk_snapshot(risk_decision),
                ]
            ),
        ),
    )


def build_trade_rationale_prompt(result: TradingAgentResult) -> tuple[PromptMessage, ...]:
    """Build a prompt for explaining the final trading-agent outcome."""
    return (
        PromptMessage("system", TRADING_AGENT_SYSTEM_PROMPT),
        PromptMessage(
            "user",
            "\n".join(
                [
                    "Write a final operator-facing rationale for this trading-agent run.",
                    "Explain why the workflow ended with this status.",
                    "Keep the explanation short, auditable, and aligned with guardrails.",
                    "",
                    f"Status: {result.status.value}",
                    f"Action: {result.action.value}",
                    f"Symbol: {result.symbol}",
                    f"Approval required: {result.approval_required}",
                    f"Market summary: {result.market_summary}",
                    f"Signal explanation: {result.signal_explanation}",
                    _strategy_snapshot(result.strategy_decision),
                    _risk_snapshot(result.risk_decision),
                    f"Guardrail decision: {result.limit_decision}",
                    f"Execution result: {result.execution_result}",
                    f"Messages: {', '.join(result.messages) or 'None'}",
                ]
            ),
        ),
    )


def format_prompt_messages(messages: tuple[PromptMessage, ...]) -> str:
    """Render prompt messages for logs, smoke tests, or non-chat clients."""
    return "\n\n".join(f"{message.role.upper()}:\n{message.content}" for message in messages)


def _request_snapshot(request: TradingAgentRequest) -> str:
    return "\n".join(
        [
            f"Symbol: {request.symbol.upper()}",
            f"Price: {request.price}",
            f"Model signal: {request.model_signal}",
            f"Model confidence: {request.model_confidence:.4f}",
            f"Current position: {request.current_position}",
            f"Requested quantity: {request.requested_quantity}",
            f"Live execution requested: {request.execute_live}",
        ]
    )


def _strategy_snapshot(strategy_decision: StrategyDecision) -> str:
    return "\n".join(
        [
            "Strategy decision:",
            f"- action: {strategy_decision.action.value}",
            f"- confidence: {strategy_decision.confidence:.4f}",
            f"- reason: {strategy_decision.reason}",
        ]
    )


def _risk_snapshot(risk_decision: RiskDecision) -> str:
    return "\n".join(
        [
            "Risk decision:",
            f"- status: {risk_decision.status.value}",
            f"- action: {risk_decision.action.value}",
            f"- quantity: {risk_decision.quantity}",
            f"- notional value: {risk_decision.notional_value}",
            f"- reason: {risk_decision.reason}",
        ]
    )


def _json_block(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


__all__ = [
    "PromptMessage",
    "TRADING_AGENT_SYSTEM_PROMPT",
    "build_human_approval_prompt",
    "build_market_context_prompt",
    "build_signal_explanation_prompt",
    "build_trade_rationale_prompt",
    "format_prompt_messages",
]
