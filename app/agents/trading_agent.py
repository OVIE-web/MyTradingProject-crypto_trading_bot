from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

from app.agents.agents_schemas import (
    HumanApproval,
    TradingAgentRequest,
    TradingAgentResult,
    TradingAgentRunStatus,
    TradingAgentState,
    TradingAgentStateUpdate,
)
from app.agents.agents_tools import (
    apply_guardrails,
    execute_if_approved,
    explain_model_signal,
    propose_strategy_decision,
    requires_human_approval,
    resolve_human_approval,
    review_strategy_risk,
    status_message,
    summarize_market_context,
)
from app.agents.llm_client import GeminiLLMClient, LLMClientError
from app.agents.prompts import build_trade_rationale_prompt
from app.domain.risk_management import AccountState, RiskDecision
from app.domain.signals import TradeAction
from app.domain.trading_strategy import StrategyDecision
from app.guardrails.trade_limits import TradeLimitDecision
from app.services.trade_execution_service import TradeExecutionService

AgentNode = Callable[[TradingAgentState], TradingAgentStateUpdate]


class TradingAgent:
    """LangGraph-compatible trading workflow with a deterministic fallback.

    The agent orchestrates strategy, risk, approval, guardrails, and optional
    execution. It does not let an LLM or agent node bypass deterministic safety
    checks.
    """

    def __init__(
        self,
        *,
        execution_service: TradeExecutionService | None = None,
        llm_client: GeminiLLMClient | None = None,
        use_langgraph: bool = True,
    ) -> None:
        self._execution_service = execution_service
        self._llm_client = llm_client
        self._graph = self._build_langgraph() if use_langgraph else None

    @property
    def uses_langgraph(self) -> bool:
        """Return True when a LangGraph graph was compiled successfully."""
        return self._graph is not None

    def run(self, request: TradingAgentRequest) -> TradingAgentResult:
        """Run the trading-agent workflow and return a structured result."""
        initial_state: TradingAgentState = {
            "request": request,
            "messages": [],
            "metadata": {"runtime": "langgraph" if self.uses_langgraph else "fallback"},
        }

        if self._graph is not None:
            final_state = self._graph.invoke(
                initial_state,
                config=_langsmith_run_config(request),
            )
        else:
            final_state = self._run_fallback(initial_state)

        result = self._build_result(final_state)
        return self._with_operator_rationale(result)

    def _build_langgraph(self) -> Any | None:
        """Compile the LangGraph workflow when LangGraph is installed."""
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            return None

        graph = StateGraph(TradingAgentState)
        graph.add_node("summarize_market", self._summarize_market_node)
        graph.add_node("explain_signal", self._explain_signal_node)
        graph.add_node("propose_strategy", self._propose_strategy_node)
        graph.add_node("review_risk", self._review_risk_node)
        graph.add_node("request_approval", self._request_approval_node)
        graph.add_node("apply_guardrails", self._apply_guardrails_node)
        graph.add_node("execute_trade", self._execute_trade_node)

        graph.add_edge(START, "summarize_market")
        graph.add_edge("summarize_market", "explain_signal")
        graph.add_edge("explain_signal", "propose_strategy")
        graph.add_edge("propose_strategy", "review_risk")
        graph.add_edge("review_risk", "request_approval")
        graph.add_edge("request_approval", "apply_guardrails")
        graph.add_edge("apply_guardrails", "execute_trade")
        graph.add_edge("execute_trade", END)

        return graph.compile()

    def _run_fallback(self, state: TradingAgentState) -> TradingAgentState:
        """Run the same node sequence without LangGraph installed."""
        for node in (
            self._summarize_market_node,
            self._explain_signal_node,
            self._propose_strategy_node,
            self._review_risk_node,
            self._request_approval_node,
            self._apply_guardrails_node,
            self._execute_trade_node,
        ):
            state = _merge_state(state, node(state))
        return state

    def _summarize_market_node(self, state: TradingAgentState) -> TradingAgentStateUpdate:
        request = state["request"]
        summary = summarize_market_context(request)
        return {
            "market_summary": summary,
            "messages": _append_message(state, "market_context_summarized"),
        }

    def _explain_signal_node(self, state: TradingAgentState) -> TradingAgentStateUpdate:
        request = state["request"]
        explanation = explain_model_signal(request)
        return {
            "signal_explanation": explanation,
            "messages": _append_message(state, "model_signal_explained"),
        }

    def _propose_strategy_node(self, state: TradingAgentState) -> TradingAgentStateUpdate:
        request = state["request"]
        decision = propose_strategy_decision(request)
        return {
            "strategy_decision": decision,
            "messages": _append_message(
                state,
                status_message(
                    "strategy_decision_created",
                    {"action": decision.action.value, "reason": decision.reason},
                ),
            ),
        }

    def _review_risk_node(self, state: TradingAgentState) -> TradingAgentStateUpdate:
        request = state["request"]
        strategy_decision = _require_strategy_decision(state)
        risk_decision = review_strategy_risk(request, strategy_decision)
        return {
            "risk_decision": risk_decision,
            "messages": _append_message(
                state,
                status_message(
                    "risk_review_completed",
                    {
                        "status": risk_decision.status.value,
                        "reason": risk_decision.reason,
                    },
                ),
            ),
        }

    def _request_approval_node(self, state: TradingAgentState) -> TradingAgentStateUpdate:
        request = state["request"]
        strategy_decision = _require_strategy_decision(state)
        risk_decision = _require_risk_decision(state)
        approval_required = requires_human_approval(
            request=request,
            strategy_decision=strategy_decision,
            risk_decision=risk_decision,
        )
        approval = resolve_human_approval(
            request=request,
            approval_required=approval_required,
        )

        if approval_required and approval is not None and approval.approved is True:
            message = "human_approval_received"
        elif approval_required:
            message = "human_approval_required"
        else:
            message = "human_approval_not_required"

        return {
            "approval_required": approval_required,
            "approval": approval,
            "messages": _append_message(state, message),
        }

    def _apply_guardrails_node(self, state: TradingAgentState) -> TradingAgentStateUpdate:
        request = state["request"]
        limit_decision = apply_guardrails(
            request=request,
            strategy_decision=_require_strategy_decision(state),
            risk_decision=_require_risk_decision(state),
            approval_required=state.get("approval_required", False),
            approval=state.get("approval"),
        )
        return {
            "limit_decision": limit_decision,
            "messages": _append_message(
                state,
                status_message(
                    "guardrails_checked",
                    {
                        "status": limit_decision.status.value,
                        "reason": limit_decision.reason,
                    },
                ),
            ),
        }

    def _execute_trade_node(self, state: TradingAgentState) -> TradingAgentStateUpdate:
        request = state["request"]
        strategy_decision = _require_strategy_decision(state)
        risk_decision = _require_risk_decision(state)
        limit_decision = _require_limit_decision(state)

        execution_result = execute_if_approved(
            request=request,
            risk_decision=risk_decision,
            limit_decision=limit_decision,
            execution_service=self._execution_service,
        )
        status = _resolve_status(
            request=request,
            action=strategy_decision.action,
            approval_required=state.get("approval_required", False),
            approval=state.get("approval"),
            limit_allowed=limit_decision.allowed,
            execution_success=execution_result.success if execution_result else None,
        )
        return {
            "execution_result": execution_result,
            "status": status,
            "messages": _append_message(
                state, status_message("workflow_completed", {"status": status.value})
            ),
        }

    def _build_result(self, state: TradingAgentState) -> TradingAgentResult:
        request = state["request"]
        strategy_decision = _require_strategy_decision(state)
        risk_decision = _require_risk_decision(state)

        return TradingAgentResult(
            status=state.get("status", TradingAgentRunStatus.FAILED),
            action=strategy_decision.action,
            symbol=request.symbol.upper(),
            market_summary=state.get("market_summary", ""),
            signal_explanation=state.get("signal_explanation", ""),
            strategy_decision=strategy_decision,
            risk_decision=risk_decision,
            limit_decision=state.get("limit_decision"),
            execution_result=state.get("execution_result"),
            approval_required=state.get("approval_required", False),
            approval=state.get("approval"),
            messages=tuple(state.get("messages", [])),
            metadata=state.get("metadata", {}),
        )

    def _with_operator_rationale(self, result: TradingAgentResult) -> TradingAgentResult:
        """Attach an optional LLM-generated rationale after safety checks finish."""
        if self._llm_client is None:
            return result

        try:
            rationale = self._llm_client.try_generate_text(build_trade_rationale_prompt(result))
        except LLMClientError:
            rationale = None

        if not rationale:
            return result

        return TradingAgentResult(
            status=result.status,
            action=result.action,
            symbol=result.symbol,
            market_summary=result.market_summary,
            signal_explanation=result.signal_explanation,
            strategy_decision=result.strategy_decision,
            risk_decision=result.risk_decision,
            limit_decision=result.limit_decision,
            execution_result=result.execution_result,
            approval_required=result.approval_required,
            approval=result.approval,
            operator_rationale=rationale,
            messages=(*result.messages, "operator_rationale_generated"),
            metadata={**result.metadata, "operator_rationale_source": "gemini"},
        )


def run_trading_agent(
    request: TradingAgentRequest,
    *,
    execution_service: TradeExecutionService | None = None,
    llm_client: GeminiLLMClient | None = None,
    use_langgraph: bool = True,
) -> TradingAgentResult:
    """Convenience function for one-off trading-agent runs."""
    return TradingAgent(
        execution_service=execution_service,
        llm_client=llm_client,
        use_langgraph=use_langgraph,
    ).run(request)


def _langsmith_run_config(request: TradingAgentRequest) -> dict[str, Any]:
    """Return LangGraph run metadata used by LangSmith tracing."""
    symbol = request.symbol.upper()
    return {
        "run_name": "trading_agent_decision",
        "tags": ["trading-agent", symbol],
        "metadata": {
            "symbol": symbol,
            "execute_live": request.execute_live,
            "model_confidence": request.model_confidence,
        },
    }


def _append_message(state: TradingAgentState, message: str) -> list[str]:
    messages = list(state.get("messages", []))
    messages.append(message)
    return messages


def _merge_state(
    state: TradingAgentState,
    update: TradingAgentStateUpdate,
) -> TradingAgentState:
    """Return a new state with a node update applied."""
    return cast(TradingAgentState, {**state, **update})


def _require_strategy_decision(state: TradingAgentState) -> StrategyDecision:
    """Return the strategy decision or fail fast when graph order is broken."""
    strategy_decision = state.get("strategy_decision")
    if strategy_decision is None:
        raise RuntimeError("strategy_decision must be set before this node.")
    return strategy_decision


def _require_risk_decision(state: TradingAgentState) -> RiskDecision:
    """Return the risk decision or fail fast when graph order is broken."""
    risk_decision = state.get("risk_decision")
    if risk_decision is None:
        raise RuntimeError("risk_decision must be set before this node.")
    return risk_decision


def _require_limit_decision(state: TradingAgentState) -> TradeLimitDecision:
    """Return the guardrail decision or fail fast when graph order is broken."""
    limit_decision = state.get("limit_decision")
    if limit_decision is None:
        raise RuntimeError("limit_decision must be set before execution.")
    return limit_decision


def _resolve_status(
    *,
    request: TradingAgentRequest,
    action: TradeAction,
    approval_required: bool,
    approval: HumanApproval | None,
    limit_allowed: bool,
    execution_success: bool | None,
) -> TradingAgentRunStatus:
    if action == TradeAction.HOLD:
        return TradingAgentRunStatus.HOLD

    if approval_required and (approval is None or approval.approved is None):
        return TradingAgentRunStatus.NEEDS_APPROVAL

    if approval_required and approval is not None and approval.approved is False:
        return TradingAgentRunStatus.BLOCKED

    if not limit_allowed:
        return TradingAgentRunStatus.BLOCKED

    if not request.execute_live:
        return TradingAgentRunStatus.DRY_RUN

    if execution_success is True:
        return TradingAgentRunStatus.EXECUTED

    return TradingAgentRunStatus.FAILED


def _main() -> None:
    """Run a safe dry-run smoke test for local LangGraph/LangSmith checks."""
    result = run_trading_agent(
        TradingAgentRequest(
            symbol="BTCUSDT",
            price=65_000.0,
            model_signal=TradeAction.BUY,
            model_confidence=0.82,
            account=AccountState(cash_balance=10_000.0),
            execute_live=False,
            market_context={
                "timeframe": "4h",
                "trend": "sample dry-run context",
                "source": "local_cli_smoke_test",
            },
        )
    )
    print(
        json.dumps(
            {
                "status": result.status.value,
                "action": result.action.value,
                "symbol": result.symbol,
                "approval_required": result.approval_required,
                "operator_rationale": result.operator_rationale,
                "messages": result.messages,
                "metadata": result.metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
