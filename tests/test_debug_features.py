"""
Tests for src/debug_features.py module.
Tests feature debugging functionality.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestDebugFeatures:
    """Test cases for debug_features script."""

    @patch("xgboost.XGBClassifier")
    @patch("src.debug_features.BinanceManager")
    @patch("src.debug_features.calculate_technical_indicators")
    @patch("src.debug_features.logging")
    def test_debug_features_execution(
        self,
        mock_logging: MagicMock,
        mock_calculate_indicators: MagicMock,
        mock_binance_manager: MagicMock,
        mock_xgb_classifier: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that debug script executes without errors and logs expected information."""
        # Setup environment
        monkeypatch.setenv("MODEL_SAVE_PATH", "/tmp/test_model.json")

        # Mock model
        mock_model_instance = MagicMock()
        mock_booster = MagicMock()
        mock_booster.feature_names = ["rsi", "bb_upper", "bb_lower", "sma_20"]
        mock_model_instance.get_booster.return_value = mock_booster
        mock_model_instance.n_features_in_ = 4
        mock_xgb_classifier.return_value = mock_model_instance

        # Mock Binance data
        mock_binance_instance = MagicMock()
        mock_klines = [
            [
                1640995200000,
                "50000.0",
                "51000.0",
                "49000.0",
                "50500.0",
                "100.0",
                1640998800000,
                "5000.0",
                "50",
                "25.0",
                "25.0",
                "0",
            ]
        ] * 50  # 50 rows
        mock_binance_instance.get_latest_ohlcv.return_value = mock_klines
        mock_binance_manager.return_value = mock_binance_instance

        # Mock feature engineering
        mock_features_df = pd.DataFrame(
            {
                "rsi": [30.0] * 50,
                "bb_upper": [51000.0] * 50,
                "bb_lower": [49000.0] * 50,
                "sma_20": [50000.0] * 50,
                "extra_feature": [1.0] * 50,  # Extra feature not in model
            }
        )
        mock_calculate_indicators.return_value = mock_features_df

        # Execute the debug script by importing it (this runs the module-level code)
        # We need to reload the module to execute it again
        import importlib

        import src.debug_features

        importlib.reload(src.debug_features)

        # Verify logging calls
        assert mock_logging.info.call_count >= 4  # Multiple info logs expected

        # Check that expected logs were made
        log_calls = [str(call) for call in mock_logging.info.call_args_list]
        assert any("Model expects 4 features" in call for call in log_calls)
        assert any("Expected feature names:" in call for call in log_calls)
        assert any("Current features shape:" in call for call in log_calls)
        assert any("Current feature names:" in call for call in log_calls)

    @patch("xgboost.XGBClassifier")
    @patch("src.debug_features.logging")
    def test_debug_features_missing_model_file(
        self,
        mock_logging: MagicMock,
        mock_xgb_classifier: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test behavior when model file doesn't exist."""
        monkeypatch.setenv("MODEL_SAVE_PATH", "/nonexistent/model.json")

        # Mock model to raise exception on load
        mock_model_instance = MagicMock()
        mock_model_instance.load_model.side_effect = Exception("Model file not found")
        mock_xgb_classifier.return_value = mock_model_instance

        # This should raise an exception when trying to load the model
        with pytest.raises(Exception, match="Model file not found"):
            import importlib

            import src.debug_features

            importlib.reload(src.debug_features)
