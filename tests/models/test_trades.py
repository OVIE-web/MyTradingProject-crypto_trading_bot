"""Unit tests for app.models.trades."""

from __future__ import annotations

from app.models import trades as trades_alias
from app.models.trade import Trade


def test_trades_module_alias_exports_trade_model() -> None:
    """app.models.trades should remain a compatibility alias for Trade."""
    assert trades_alias.__all__ == ["Trade"]
    assert trades_alias.Trade is Trade
