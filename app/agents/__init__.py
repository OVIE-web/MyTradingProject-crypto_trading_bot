from typing import TYPE_CHECKING

from app.agents.agents_schemas import (
    ApprovalPolicy,
    HumanApproval,
    TradingAgentRequest,
    TradingAgentResult,
    TradingAgentRunStatus,
)
from app.agents.llm_client import (
    GeminiLLMClient,
    GeminiLLMConfig,
    LLMClientError,
    LLMConfigurationError,
)
from app.agents.prompts import (
    PromptMessage,
    build_human_approval_prompt,
    build_market_context_prompt,
    build_signal_explanation_prompt,
    build_trade_rationale_prompt,
    format_prompt_messages,
)

if TYPE_CHECKING:
    from app.agents.trading_agent import TradingAgent, run_trading_agent

__all__ = [
    "ApprovalPolicy",
    "GeminiLLMClient",
    "GeminiLLMConfig",
    "HumanApproval",
    "LLMClientError",
    "LLMConfigurationError",
    "PromptMessage",
    "TradingAgent",
    "TradingAgentRequest",
    "TradingAgentResult",
    "TradingAgentRunStatus",
    "build_human_approval_prompt",
    "build_market_context_prompt",
    "build_signal_explanation_prompt",
    "build_trade_rationale_prompt",
    "format_prompt_messages",
    "run_trading_agent",
]


def __getattr__(name: str) -> object:
    """Lazily expose trading-agent helpers without pre-importing the module."""
    if name in {"TradingAgent", "run_trading_agent"}:
        from app.agents.trading_agent import TradingAgent, run_trading_agent

        return {
            "TradingAgent": TradingAgent,
            "run_trading_agent": run_trading_agent,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
