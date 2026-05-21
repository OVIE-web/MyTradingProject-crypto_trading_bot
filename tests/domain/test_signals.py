"""Unit tests for signal processing logic, including normalization, confidence validation, and signal combination rules."""

from __future__ import annotations

import pytest

from app.domain.signals import (
    SignalSource,
    SignalValue,
    TradeAction,
    TradingSignal,
    action_from_signal,
    combine_signals,
    confidence_is_valid,
    create_signal,
    normalize_signal,
)


@pytest.mark.parametrize(
    ("raw_signal", "expected"),
    [
        (-1, -1),
        (0, 0),
        (1, 1),
        (-1.0, -1),
        ("sell", -1),
        ("SHORT", -1),
        ("hold", 0),
        ("wait", 0),
        ("buy", 1),
        ("ENTER", 1),
        (SignalValue.BUY, 1),
        (TradeAction.SELL, -1),
    ],
)
def test_normalize_signal_accepts_supported_representations(raw_signal, expected) -> None:
    """Supported model, text, enum, and action values should normalize to -1/0/1."""
    assert normalize_signal(raw_signal) == expected


@pytest.mark.parametrize("raw_signal", ["panic", 2, -2])
def test_normalize_signal_rejects_unsupported_values(raw_signal) -> None:
    """Unknown values should fail before they reach strategy execution."""
    with pytest.raises(ValueError, match="Unsupported signal value"):
        normalize_signal(raw_signal)


def test_action_from_signal_maps_to_trade_action() -> None:
    """Normalized signal values should map to executable trade actions."""
    assert action_from_signal("buy") == TradeAction.BUY
    assert action_from_signal("hold") == TradeAction.HOLD
    assert action_from_signal("sell") == TradeAction.SELL


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(-0.01, False), (0.0, True), (0.5, True), (1.0, True), (1.01, False)],
)
def test_confidence_is_valid(confidence: float, expected: bool) -> None:
    """Signal confidence must stay inside the normalized 0-1 range."""
    assert confidence_is_valid(confidence) is expected


def test_create_signal_returns_typed_trading_signal() -> None:
    """Raw signal inputs should become immutable TradingSignal objects."""
    signal = create_signal(
        "long",
        0.82,
        source=SignalSource.RSI,
        reason="rsi_oversold",
        metadata={"rsi": 24.5},
    )

    assert isinstance(signal, TradingSignal)
    assert signal.value == 1
    assert signal.action == TradeAction.BUY
    assert signal.confidence == 0.82
    assert signal.source == SignalSource.RSI
    assert signal.reason == "rsi_oversold"
    assert signal.metadata == {"rsi": 24.5}
    assert signal.is_actionable is True


def test_create_signal_rejects_invalid_confidence() -> None:
    """Invalid confidence should not produce a trade signal."""
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        create_signal("buy", 1.5)


def test_combine_signals_returns_hold_when_empty() -> None:
    """No inputs should produce a safe HOLD composite signal."""
    signal = combine_signals([])

    assert signal.action == TradeAction.HOLD
    assert signal.confidence == 0.0
    assert signal.source == SignalSource.COMPOSITE
    assert signal.reason == "no_signals_available"


def test_combine_signals_uses_confidence_weighted_vote() -> None:
    """Higher-confidence signals should dominate lower-confidence disagreement."""
    buy_signal = create_signal("buy", 0.9)
    sell_signal = create_signal("sell", 0.2)

    combined = combine_signals([buy_signal, sell_signal])

    assert combined.action == TradeAction.BUY
    assert combined.value == 1
    assert combined.confidence == pytest.approx((0.9 - 0.2) / 1.1)
    assert combined.metadata["signal_count"] == 2
    assert combined.metadata["eligible_signal_count"] == 2


def test_combine_signals_filters_by_min_confidence() -> None:
    """Signals below the minimum confidence should be ignored."""
    weak_buy = create_signal("buy", 0.4)
    strong_sell = create_signal("sell", 0.8)

    combined = combine_signals([weak_buy, strong_sell], min_confidence=0.5)

    assert combined.action == TradeAction.SELL
    assert combined.confidence == 1.0
    assert combined.metadata["eligible_signal_count"] == 1


def test_combine_signals_returns_hold_when_no_signal_meets_threshold() -> None:
    """If all signals are too weak, the composite result should be HOLD."""
    combined = combine_signals([create_signal("buy", 0.2)], min_confidence=0.9)

    assert combined.action == TradeAction.HOLD
    assert combined.reason == "no_signals_met_min_confidence"
    assert combined.metadata == {"min_confidence": 0.9}
