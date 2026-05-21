from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest

from app.tools.debug_features import debug_feature_alignment


@pytest.fixture
def local_tmp_dir() -> Iterator[Path]:
    path = Path("tests/tools/.tmp") / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)

    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        parent = path.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


def test_debug_feature_alignment_returns_model_and_pipeline_summary(
    local_tmp_dir: Path,
) -> None:
    model_path = local_tmp_dir / "xgboost_model.json"
    model_path.write_text("fake model", encoding="utf-8")

    model = MagicMock()
    booster = MagicMock()
    booster.feature_names = ["rsi", "sma_20"]
    model.get_booster.return_value = booster
    model.n_features_in_ = 2

    candles = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [95.0, 96.0],
            "close": [102.0, 103.0],
            "volume": [1_000.0, 1_100.0],
        }
    )
    features = pd.DataFrame(
        {
            "rsi": [42.0, 47.0],
            "sma_20": [101.0, 102.0],
            "extra_feature": [1.0, 1.0],
            "label": ["hold", "buy"],
        }
    )

    with (
        patch("xgboost.XGBClassifier", return_value=model) as classifier,
        patch("app.services.binance_service.BinanceManager") as manager_cls,
        patch(
            "app.tools.feature_engineer.calculate_technical_indicators",
            return_value=features,
        ) as calculate_indicators,
    ):
        manager = manager_cls.return_value
        manager.get_latest_ohlcv.return_value = candles

        summary = debug_feature_alignment(
            symbol="BTCUSDT",
            interval="1h",
            limit=120,
            model_path=str(model_path),
        )

    classifier.assert_called_once_with()
    model.load_model.assert_called_once_with(str(model_path))
    manager.get_latest_ohlcv.assert_called_once_with("BTCUSDT", interval="1h", limit=120)
    calculate_indicators.assert_called_once_with(candles)

    assert summary["model_path"] == str(model_path)
    assert summary["expected_features"] == ["rsi", "sma_20"]
    assert summary["numeric_features"] == ["rsi", "sma_20", "extra_feature"]
    assert summary["missing_features"] == []
    assert summary["extra_features"] == ["extra_feature"]
    assert summary["aligned"] is True


def test_debug_feature_alignment_raises_for_missing_model(local_tmp_dir: Path) -> None:
    missing_model = local_tmp_dir / "missing_model.json"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        debug_feature_alignment(model_path=str(missing_model))


def test_debug_feature_alignment_raises_when_exchange_returns_no_candles(
    local_tmp_dir: Path,
) -> None:
    model_path = local_tmp_dir / "xgboost_model.json"
    model_path.write_text("fake model", encoding="utf-8")

    model = MagicMock()
    model.get_booster.return_value.feature_names = ["rsi"]
    model.n_features_in_ = 1

    with (
        patch("xgboost.XGBClassifier", return_value=model),
        patch("app.services.binance_service.BinanceManager") as manager_cls,
    ):
        manager_cls.return_value.get_latest_ohlcv.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="No candles returned"):
            debug_feature_alignment(model_path=str(model_path))
