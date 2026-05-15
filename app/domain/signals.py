from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TradeAction(StrEnum):
    """Executable action requested by strategy logic."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalValue(StrEnum):
    """Canonical human-readable signal labels."""

    SELL = "sell"
    HOLD = "hold"
    BUY = "buy"


class SignalSource(StrEnum):
    """Where a signal originated."""

    MODEL = "model"
    RSI = "rsi"
    MOVING_AVERAGE = "moving_average"
    MANUAL = "manual"
    COMPOSITE = "composite"


MODEL_SIGNAL_TO_ACTION: dict[int, TradeAction] = {
    -1: TradeAction.SELL,
    0: TradeAction.HOLD,
    1: TradeAction.BUY,
}

ACTION_TO_MODEL_SIGNAL: dict[TradeAction, int] = {
    TradeAction.SELL: -1,
    TradeAction.HOLD: 0,
    TradeAction.BUY: 1,
}


@dataclass(frozen=True, slots=True)
class TradingSignal:
    """Normalized signal emitted by model, indicator, or manual logic."""

    value: int
    action: TradeAction
    confidence: float
    source: SignalSource
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Whether this signal represents a possible trade."""
        return self.action in {TradeAction.BUY, TradeAction.SELL}


def normalize_signal(signal: int | float | str | SignalValue | TradeAction) -> int:
    """Convert supported signal representations into -1, 0, or 1."""
    if isinstance(signal, TradeAction):
        return ACTION_TO_MODEL_SIGNAL[signal]

    if isinstance(signal, SignalValue):
        return {
            SignalValue.SELL: -1,
            SignalValue.HOLD: 0,
            SignalValue.BUY: 1,
        }[signal]

    if isinstance(signal, str):
        normalized = signal.strip().lower()
        text_signals = {
            "sell": -1,
            "short": -1,
            "exit": -1,
            "-1": -1,
            "hold": 0,
            "neutral": 0,
            "wait": 0,
            "0": 0,
            "buy": 1,
            "long": 1,
            "enter": 1,
            "1": 1,
        }
        if normalized in text_signals:
            return text_signals[normalized]
        raise ValueError(f"Unsupported signal value: {signal!r}")

    numeric_signal = int(signal)
    if numeric_signal not in MODEL_SIGNAL_TO_ACTION:
        raise ValueError(f"Unsupported signal value: {signal!r}")

    return numeric_signal


def action_from_signal(signal: int | float | str | SignalValue | TradeAction) -> TradeAction:
    """Map a signal value to a trade action."""
    return MODEL_SIGNAL_TO_ACTION[normalize_signal(signal)]


def confidence_is_valid(confidence: float) -> bool:
    """Return True when confidence is in the normalized 0-1 range."""
    return 0 <= float(confidence) <= 1


def create_signal(
    signal: int | float | str | SignalValue | TradeAction,
    confidence: float,
    *,
    source: SignalSource = SignalSource.MODEL,
    reason: str = "signal_normalized",
    metadata: dict[str, Any] | None = None,
) -> TradingSignal:
    """Normalize raw signal input into a typed TradingSignal."""
    if not confidence_is_valid(confidence):
        raise ValueError("confidence must be between 0 and 1.")

    normalized_signal = normalize_signal(signal)
    return TradingSignal(
        value=normalized_signal,
        action=MODEL_SIGNAL_TO_ACTION[normalized_signal],
        confidence=float(confidence),
        source=source,
        reason=reason,
        metadata=metadata or {},
    )


def combine_signals(
    signals: list[TradingSignal] | tuple[TradingSignal, ...],
    *,
    min_confidence: float = 0.0,
) -> TradingSignal:
    """Combine multiple signals using confidence-weighted voting."""
    if not signals:
        return create_signal(
            SignalValue.HOLD,
            0.0,
            source=SignalSource.COMPOSITE,
            reason="no_signals_available",
        )

    if not confidence_is_valid(min_confidence):
        raise ValueError("min_confidence must be between 0 and 1.")

    eligible_signals = [signal for signal in signals if signal.confidence >= min_confidence]
    if not eligible_signals:
        return create_signal(
            SignalValue.HOLD,
            0.0,
            source=SignalSource.COMPOSITE,
            reason="no_signals_met_min_confidence",
            metadata={"min_confidence": min_confidence},
        )

    weighted_score = sum(signal.value * signal.confidence for signal in eligible_signals)
    confidence_total = sum(signal.confidence for signal in eligible_signals)
    average_score = weighted_score / confidence_total if confidence_total else 0.0

    if average_score > 0:
        combined_value = 1
    elif average_score < 0:
        combined_value = -1
    else:
        combined_value = 0

    combined_confidence = min(1.0, abs(average_score))
    return create_signal(
        combined_value,
        combined_confidence,
        source=SignalSource.COMPOSITE,
        reason="confidence_weighted_vote",
        metadata={
            "signal_count": len(signals),
            "eligible_signal_count": len(eligible_signals),
            "average_score": average_score,
        },
    )
