from __future__ import annotations

from app.agents import (
    TradingAgentRunStatus,
    build_human_approval_prompt,
    build_market_context_prompt,
    build_signal_explanation_prompt,
    build_trade_rationale_prompt,
    format_prompt_messages,
    run_trading_agent,
)
from app.agents.prompts import TRADING_AGENT_SYSTEM_PROMPT
from app.domain.risk_management import AccountState
from app.domain.signals import TradeAction
from tests.agents.test_trading_agent import make_request


def test_system_prompt_preserves_guardrail_authority() -> None:
    assert "must never bypass deterministic strategy" in TRADING_AGENT_SYSTEM_PROMPT
    assert "Final permission to trade" in TRADING_AGENT_SYSTEM_PROMPT


def test_market_context_prompt_includes_request_and_context() -> None:
    request = make_request(market_context={"rsi": 28, "trend": "uptrend"})

    prompt = build_market_context_prompt(request)
    rendered = format_prompt_messages(prompt)

    assert prompt[0].role == "system"
    assert "BTCUSDT" in rendered
    assert "Model confidence: 0.9500" in rendered
    assert '"rsi": 28' in rendered
    assert '"trend": "uptrend"' in rendered


def test_signal_explanation_prompt_does_not_change_model_signal() -> None:
    request = make_request(model_signal=TradeAction.SELL, model_confidence=0.88)

    rendered = format_prompt_messages(
        build_signal_explanation_prompt(request, market_summary="Momentum is weakening.")
    )

    assert "Explain the model signal" in rendered
    assert "Do not change the signal" in rendered
    assert "Model signal: SELL" in rendered
    assert "Momentum is weakening." in rendered


def test_human_approval_prompt_includes_strategy_and_risk_decisions() -> None:
    request = make_request(requested_quantity=2.0)
    result = run_trading_agent(request, use_langgraph=False)

    rendered = format_prompt_messages(
        build_human_approval_prompt(
            request=request,
            strategy_decision=result.strategy_decision,
            risk_decision=result.risk_decision,
        )
    )

    assert "Prepare a human approval brief" in rendered
    assert "- action: BUY" in rendered
    assert "- status: APPROVED" in rendered
    assert "- notional value:" in rendered


def test_trade_rationale_prompt_explains_final_status() -> None:
    request = make_request(account=AccountState(cash_balance=10_000.0))
    result = run_trading_agent(request, use_langgraph=False)

    rendered = format_prompt_messages(build_trade_rationale_prompt(result))

    assert result.status == TradingAgentRunStatus.DRY_RUN
    assert "Write a final operator-facing rationale" in rendered
    assert "Status: DRY_RUN" in rendered
    assert "Guardrail decision:" in rendered
    assert "workflow_completed" in rendered
