"""Debug feature alignment between the saved model and live market data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from app.core.config import (
    INITIAL_CANDLES_HISTORY,
    MODEL_SAVE_PATH,
    TRADE_INTERVAL,
    TRADE_SYMBOL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DEBUG_LIMIT = max(120, min(INITIAL_CANDLES_HISTORY, 500))


def debug_feature_alignment(
    symbol: str = TRADE_SYMBOL,
    interval: str = TRADE_INTERVAL,
    limit: int = DEFAULT_DEBUG_LIMIT,
    model_path: str = MODEL_SAVE_PATH,
) -> dict[str, Any]:
    """Compare model feature names against the current feature pipeline."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    import xgboost as xgb

    from app.services.binance_service import BinanceManager
    from app.tools.feature_engineer import calculate_technical_indicators

    model = xgb.XGBClassifier()
    model.load_model(str(path))

    expected_features = model.get_booster().feature_names or []
    num_features = model.n_features_in_

    logger.info("Loaded model from %s", path)
    logger.info("Model expects %s features", num_features)
    logger.info("Expected feature names: %s", expected_features)

    binance = BinanceManager()
    candles = binance.get_latest_ohlcv(symbol, interval=interval, limit=limit)

    if candles.empty:
        raise ValueError(f"No candles returned for {symbol} on interval {interval}")

    features = calculate_technical_indicators(candles)
    if features.empty:
        raise ValueError(
            "Feature engineering returned no rows. Increase --limit so rolling indicators have enough data."
        )

    logger.info("Current features shape: %s", features.shape)
    logger.info("Current feature names: %s", list(features.columns))

    numeric_features = features.select_dtypes(include=["number", "bool"])
    logger.info("After filtering to numeric: %s", numeric_features.shape)
    logger.info("Numeric feature names: %s", list(numeric_features.columns))

    missing: list[str] = []
    extra: list[str] = []

    if expected_features:
        missing = [
            feature for feature in expected_features if feature not in numeric_features.columns
        ]
        extra = [
            feature for feature in numeric_features.columns if feature not in expected_features
        ]

        if missing:
            logger.warning("Missing from data: %s", missing)
        if extra:
            logger.warning("Extra in data: %s", extra)

    summary = {
        "model_path": str(path),
        "symbol": symbol,
        "interval": interval,
        "candles_shape": tuple(candles.shape),
        "features_shape": tuple(features.shape),
        "numeric_features_shape": tuple(numeric_features.shape),
        "expected_features": expected_features,
        "numeric_features": list(numeric_features.columns),
        "missing_features": missing,
        "extra_features": extra,
        "aligned": not missing,
    }

    if summary["aligned"]:
        logger.info("Feature pipeline is aligned with the saved model.")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the debug utility."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=TRADE_SYMBOL, help="Trading pair to inspect.")
    parser.add_argument("--interval", default=TRADE_INTERVAL, help="Candlestick interval.")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_DEBUG_LIMIT,
        help="Number of candles to fetch. Use at least 120 for rolling indicators.",
    )
    parser.add_argument("--model-path", default=MODEL_SAVE_PATH, help="Path to the XGBoost model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    debug_feature_alignment(
        symbol=args.symbol,
        interval=args.interval,
        limit=args.limit,
        model_path=args.model_path,
    )
