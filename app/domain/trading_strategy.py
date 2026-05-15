from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.config import CONFIDENCE_THRESHOLD, RSI_LOWER_QUANTILE, RSI_UPPER_QUANTILE
from app.domain.signals import (
    MODEL_SIGNAL_TO_ACTION,
    SignalValue,
    TradeAction,
    normalize_signal,
)
from app.domain.signals import (
    action_from_signal as action_from_signal,
)


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Runtime controls for converting signals into trade decisions."""

    confidence_threshold: float = CONFIDENCE_THRESHOLD
    buy_rsi_threshold: float = RSI_LOWER_QUANTILE * 100
    sell_rsi_threshold: float = RSI_UPPER_QUANTILE * 100
    allow_short_selling: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1.")

        if not 0 <= self.buy_rsi_threshold <= 100:
            raise ValueError("buy_rsi_threshold must be between 0 and 100.")

        if not 0 <= self.sell_rsi_threshold <= 100:
            raise ValueError("sell_rsi_threshold must be between 0 and 100.")

        if self.buy_rsi_threshold >= self.sell_rsi_threshold:
            raise ValueError("buy_rsi_threshold must be lower than sell_rsi_threshold.")


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """Final trading decision produced by the strategy layer."""

    action: TradeAction
    signal: int
    confidence: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Whether this decision should create or close a market position."""
        return self.action in {TradeAction.BUY, TradeAction.SELL}


def decide_from_model_signal(
    signal: int | float | str | SignalValue | TradeAction,
    confidence: float,
    *,
    config: StrategyConfig | None = None,
    current_position: int = 0,
    metadata: dict[str, Any] | None = None,
) -> StrategyDecision:
    """Create a trade decision from an ML model signal and confidence score.

    Signal convention:
        1  -> buy / enter long
        0  -> hold
       -1  -> sell / exit long
    """
    strategy_config = config or StrategyConfig()
    normalized_signal = normalize_signal(signal)
    normalized_confidence = float(confidence)
    decision_metadata = metadata or {}

    if not 0 <= normalized_confidence <= 1:
        raise ValueError("confidence must be between 0 and 1.")

    if normalized_confidence < strategy_config.confidence_threshold:
        return StrategyDecision(
            action=TradeAction.HOLD,
            signal=0,
            confidence=normalized_confidence,
            reason="confidence_below_threshold",
            metadata={
                **decision_metadata,
                "raw_signal": normalized_signal,
                "confidence_threshold": strategy_config.confidence_threshold,
            },
        )

    action = MODEL_SIGNAL_TO_ACTION[normalized_signal]

    if action == TradeAction.BUY and current_position > 0:
        return StrategyDecision(
            action=TradeAction.HOLD,
            signal=0,
            confidence=normalized_confidence,
            reason="already_in_long_position",
            metadata={**decision_metadata, "raw_signal": normalized_signal},
        )

    if (
        action == TradeAction.SELL
        and current_position <= 0
        and not strategy_config.allow_short_selling
    ):
        return StrategyDecision(
            action=TradeAction.HOLD,
            signal=0,
            confidence=normalized_confidence,
            reason="no_long_position_to_exit",
            metadata={**decision_metadata, "raw_signal": normalized_signal},
        )

    return StrategyDecision(
        action=action,
        signal=normalized_signal,
        confidence=normalized_confidence,
        reason="model_signal_confirmed",
        metadata=decision_metadata,
    )


def decide_from_rsi(
    rsi: float,
    *,
    config: StrategyConfig | None = None,
    current_position: int = 0,
    metadata: dict[str, Any] | None = None,
) -> StrategyDecision:
    """Create a rule-based decision from RSI values."""
    strategy_config = config or StrategyConfig()
    rsi_value = float(rsi)

    if not 0 <= rsi_value <= 100:
        raise ValueError("rsi must be between 0 and 100.")

    if rsi_value <= strategy_config.buy_rsi_threshold:
        signal = 1
        reason = "rsi_oversold"
    elif rsi_value >= strategy_config.sell_rsi_threshold:
        signal = -1
        reason = "rsi_overbought"
    else:
        signal = 0
        reason = "rsi_neutral"

    decision = decide_from_model_signal(
        signal,
        confidence=1.0,
        config=strategy_config,
        current_position=current_position,
        metadata={
            **(metadata or {}),
            "rsi": rsi_value,
            "buy_rsi_threshold": strategy_config.buy_rsi_threshold,
            "sell_rsi_threshold": strategy_config.sell_rsi_threshold,
        },
    )

    return StrategyDecision(
        action=decision.action,
        signal=decision.signal,
        confidence=decision.confidence,
        reason=reason if decision.is_actionable else decision.reason,
        metadata=decision.metadata,
    )


def decide_from_latest_prediction(
    predictions: list[int] | tuple[int, ...],
    confidences: list[float] | tuple[float, ...],
    *,
    config: StrategyConfig | None = None,
    current_position: int = 0,
    metadata: dict[str, Any] | None = None,
) -> StrategyDecision:
    """Create a decision from the latest model prediction arrays."""
    if not predictions:
        return StrategyDecision(
            action=TradeAction.HOLD,
            signal=0,
            confidence=0.0,
            reason="no_predictions_available",
            metadata=metadata or {},
        )

    if not confidences:
        return StrategyDecision(
            action=TradeAction.HOLD,
            signal=0,
            confidence=0.0,
            reason="no_confidence_available",
            metadata=metadata or {},
        )

    return decide_from_model_signal(
        predictions[-1],
        confidences[-1],
        config=config,
        current_position=current_position,
        metadata=metadata,
    )


def should_execute_trade(decision: StrategyDecision) -> bool:
    """Return True when the decision should be sent to an execution service."""
    return decision.is_actionable
