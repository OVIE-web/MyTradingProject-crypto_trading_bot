"""Unit tests for trading strategy logic, including decision-making based on model signals, RSI values, and overall strategy configuration."""

from __future__ import annotations

import pytest

from app.domain.signals import TradeAction
from app.domain.trading_strategy import (
    StrategyConfig,
    StrategyDecision,
    decide_from_latest_prediction,
    decide_from_model_signal,
    decide_from_rsi,
    should_execute_trade,
)


def test_strategy_config_rejects_invalid_thresholds() -> None:
    """Strategy config should reject impossible confidence or RSI ranges."""
    with pytest.raises(ValueError, match="confidence_threshold"):
        StrategyConfig(confidence_threshold=1.1)

    with pytest.raises(ValueError, match="buy_rsi_threshold"):
        StrategyConfig(buy_rsi_threshold=-1)

    with pytest.raises(ValueError, match="sell_rsi_threshold"):
        StrategyConfig(sell_rsi_threshold=101)

    with pytest.raises(ValueError, match="buy_rsi_threshold must be lower"):
        StrategyConfig(buy_rsi_threshold=70, sell_rsi_threshold=30)


def test_decide_from_model_signal_holds_below_confidence_threshold() -> None:
    """Low confidence should block otherwise actionable model signals."""
    config = StrategyConfig(confidence_threshold=0.8)

    decision = decide_from_model_signal(
        "buy",
        0.7,
        config=config,
        metadata={"source": "unit-test"},
    )

    assert decision.action == TradeAction.HOLD
    assert decision.signal == 0
    assert decision.reason == "confidence_below_threshold"
    assert decision.metadata["raw_signal"] == 1
    assert decision.metadata["confidence_threshold"] == 0.8
    assert decision.metadata["source"] == "unit-test"


def test_decide_from_model_signal_confirms_buy() -> None:
    """A high-confidence BUY signal should become an actionable decision."""
    decision = decide_from_model_signal("buy", 0.92, config=StrategyConfig(0.5))

    assert decision.action == TradeAction.BUY
    assert decision.signal == 1
    assert decision.reason == "model_signal_confirmed"
    assert decision.is_actionable is True


def test_decide_from_model_signal_blocks_duplicate_long_entry() -> None:
    """BUY should become HOLD when we already have a long position."""
    decision = decide_from_model_signal("buy", 0.95, current_position=1)

    assert decision.action == TradeAction.HOLD
    assert decision.reason == "already_in_long_position"
    assert decision.metadata["raw_signal"] == 1


def test_decide_from_model_signal_blocks_sell_without_position_by_default() -> None:
    """SELL should not execute without a long position unless shorting is enabled."""
    decision = decide_from_model_signal("sell", 0.95, current_position=0)

    assert decision.action == TradeAction.HOLD
    assert decision.reason == "no_long_position_to_exit"


def test_decide_from_model_signal_allows_short_selling_when_configured() -> None:
    """Short selling can be explicitly enabled in strategy config."""
    config = StrategyConfig(allow_short_selling=True)

    decision = decide_from_model_signal("sell", 0.95, config=config, current_position=0)

    assert decision.action == TradeAction.SELL
    assert decision.signal == -1


def test_decide_from_model_signal_rejects_invalid_confidence() -> None:
    """Confidence outside 0-1 should fail fast."""
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        decide_from_model_signal("buy", 1.2)


def test_decide_from_rsi_creates_oversold_buy_decision() -> None:
    """RSI below the buy threshold should produce a BUY decision."""
    config = StrategyConfig(confidence_threshold=0.5, buy_rsi_threshold=30, sell_rsi_threshold=70)

    decision = decide_from_rsi(25, config=config)

    assert decision.action == TradeAction.BUY
    assert decision.signal == 1
    assert decision.reason == "rsi_oversold"
    assert decision.metadata["rsi"] == 25


def test_decide_from_rsi_creates_overbought_sell_decision_when_position_exists() -> None:
    """RSI above the sell threshold should exit an existing long position."""
    config = StrategyConfig(confidence_threshold=0.5, buy_rsi_threshold=30, sell_rsi_threshold=70)

    decision = decide_from_rsi(75, config=config, current_position=1)

    assert decision.action == TradeAction.SELL
    assert decision.signal == -1
    assert decision.reason == "rsi_overbought"


def test_decide_from_rsi_returns_hold_for_neutral_value() -> None:
    """Neutral RSI values should not create a trade."""
    config = StrategyConfig(confidence_threshold=0.5, buy_rsi_threshold=30, sell_rsi_threshold=70)

    decision = decide_from_rsi(50, config=config)

    assert decision.action == TradeAction.HOLD
    assert decision.reason == "model_signal_confirmed"
    assert decision.metadata["rsi"] == 50


def test_decide_from_rsi_rejects_invalid_rsi() -> None:
    """RSI must remain within 0-100."""
    with pytest.raises(ValueError, match="rsi must be between 0 and 100"):
        decide_from_rsi(120)


def test_decide_from_latest_prediction_uses_last_values() -> None:
    """Latest prediction and confidence should drive the decision."""
    decision = decide_from_latest_prediction(
        [0, 1],
        [0.4, 0.95],
        config=StrategyConfig(confidence_threshold=0.5),
    )

    assert decision.action == TradeAction.BUY
    assert decision.confidence == 0.95


def test_decide_from_latest_prediction_handles_empty_inputs() -> None:
    """Missing model outputs should safely hold."""
    no_predictions = decide_from_latest_prediction([], [0.9])
    no_confidences = decide_from_latest_prediction([1], [])

    assert no_predictions.action == TradeAction.HOLD
    assert no_predictions.reason == "no_predictions_available"
    assert no_confidences.action == TradeAction.HOLD
    assert no_confidences.reason == "no_confidence_available"


def test_should_execute_trade_matches_decision_actionability() -> None:
    """Only BUY and SELL decisions should be execution candidates."""
    buy_decision = StrategyDecision(TradeAction.BUY, 1, 0.9, "confirmed")
    hold_decision = StrategyDecision(TradeAction.HOLD, 0, 0.9, "hold")

    assert should_execute_trade(buy_decision) is True
    assert should_execute_trade(hold_decision) is False
