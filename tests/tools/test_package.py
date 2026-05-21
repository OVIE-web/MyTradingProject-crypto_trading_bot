from __future__ import annotations

import app.tools as tools
from app.tools.feature_engineer import (
    apply_rsi_labels,
    calculate_technical_indicators,
    get_rsi_quantile_thresholds,
    normalize_features,
)
from app.tools.technical_indicators import (
    add_atr,
    add_bollinger_bands,
    add_momentum,
    add_moving_averages,
    add_rsi,
    add_technical_indicators,
    validate_ohlcv_columns,
)


def test_tools_package_exports_public_api() -> None:
    assert tools.__all__ == [
        "add_atr",
        "add_bollinger_bands",
        "add_momentum",
        "add_moving_averages",
        "add_rsi",
        "add_technical_indicators",
        "apply_rsi_labels",
        "calculate_technical_indicators",
        "get_rsi_quantile_thresholds",
        "normalize_features",
        "validate_ohlcv_columns",
    ]


def test_tools_package_exports_match_source_modules() -> None:
    assert tools.add_atr is add_atr
    assert tools.add_bollinger_bands is add_bollinger_bands
    assert tools.add_momentum is add_momentum
    assert tools.add_moving_averages is add_moving_averages
    assert tools.add_rsi is add_rsi
    assert tools.add_technical_indicators is add_technical_indicators
    assert tools.apply_rsi_labels is apply_rsi_labels
    assert tools.calculate_technical_indicators is calculate_technical_indicators
    assert tools.get_rsi_quantile_thresholds is get_rsi_quantile_thresholds
    assert tools.normalize_features is normalize_features
    assert tools.validate_ohlcv_columns is validate_ohlcv_columns
