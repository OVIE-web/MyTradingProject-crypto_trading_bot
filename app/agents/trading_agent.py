from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
from app.domain.signals import TradeAction
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
        use_langgraph: bool = True,
    ) -> None:
        self._execution_service = execution_service
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
            final_state = self._graph.invoke(initial_state)
        else:
            final_state = self._run_fallback(initial_state)

        return self._build_result(final_state)

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
            state.update(node(state))
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
        strategy_decision = state["strategy_decision"]
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
        strategy_decision = state["strategy_decision"]
        risk_decision = state["risk_decision"]
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
            strategy_decision=state["strategy_decision"],
            risk_decision=state["risk_decision"],
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
        risk_decision = state["risk_decision"]
        limit_decision = state["limit_decision"]
        if limit_decision is None:
            raise RuntimeError("limit_decision must be set before execution.")

        execution_result = execute_if_approved(
            request=request,
            risk_decision=risk_decision,
            limit_decision=limit_decision,
            execution_service=self._execution_service,
        )
        status = _resolve_status(
            request=request,
            action=state["strategy_decision"].action,
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
        strategy_decision = state["strategy_decision"]
        risk_decision = state["risk_decision"]

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


def run_trading_agent(
    request: TradingAgentRequest,
    *,
    execution_service: TradeExecutionService | None = None,
    use_langgraph: bool = True,
) -> TradingAgentResult:
    """Convenience function for one-off trading-agent runs."""
    return TradingAgent(
        execution_service=execution_service,
        use_langgraph=use_langgraph,
    ).run(request)


def _append_message(state: TradingAgentState, message: str) -> list[str]:
    messages = list(state.get("messages", []))
    messages.append(message)
    return messages


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
