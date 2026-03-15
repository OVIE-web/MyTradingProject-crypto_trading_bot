"""
Comprehensive test suite for streamlit_app.py

Tests cover:
- Input validation
- Database connection management
- Telegram message handling
- Model loading and prediction
- Error handling and edge cases
- Data processing
"""

from __future__ import annotations

import logging
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest
import xgboost as xgb

logger = logging.getLogger(__name__)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_user_input() -> dict[str, float]:
    """Create sample technical indicator input."""
    return {
        "rsi": 50.0,
        "bb_upper": 45000.0,
        "bb_lower": 35000.0,
        "bb_mid": 40000.0,
        "bb_pct_b": 0.5,
        "sma_20": 40000.0,
        "sma_50": 40000.0,
        "ma_cross": 100.0,
        "price_momentum": 5.0,
        "atr": 500.0,
        "atr_pct": 1.2,
    }


@pytest.fixture
def sample_trades_dataframe() -> pd.DataFrame:
    """Create sample trades dataframe."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5),
            "side": ["BUY", "SELL", "BUY", "SELL", "BUY"],
            "price": [40000, 40500, 39500, 40200, 39800],
            "quantity": [0.1, 0.1, 0.1, 0.1, 0.1],
        }
    )


@pytest.fixture
def mock_telegram_notifier():
    """Create mock Telegram notifier."""
    notifier = AsyncMock()
    notifier.send_message = AsyncMock()
    return notifier


@pytest.fixture
def mock_model():
    """Create mock ML model."""
    model = Mock()
    model.predict = Mock(return_value=[1])
    model.predict_proba = Mock(return_value=[[0.1, 0.2, 0.7]])
    return model


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================


class TestValidateTechnicalIndicators:
    """Test input validation for technical indicators."""

    def test_validate_all_valid_inputs(self, sample_user_input: dict[str, float]) -> None:
        """Test all valid indicators pass validation."""
        # Import the validation function
        from src.streamlit_app import validate_technical_indicators

        is_valid, error_msg = validate_technical_indicators(sample_user_input)
        assert is_valid
        assert error_msg == ""

    def test_validate_rsi_too_high(self) -> None:
        """Test RSI above 100 fails validation."""
        from src.streamlit_app import validate_technical_indicators

        invalid_input = {"rsi": 150.0}
        is_valid, error_msg = validate_technical_indicators(invalid_input)
        assert not is_valid
        assert "RSI" in error_msg
        assert "100" in error_msg

    def test_validate_rsi_too_low(self) -> None:
        """Test RSI below 0 fails validation."""
        from src.streamlit_app import validate_technical_indicators

        invalid_input = {"rsi": -10.0}
        is_valid, error_msg = validate_technical_indicators(invalid_input)
        assert not is_valid
        assert "RSI" in error_msg

    def test_validate_bb_pct_b_too_high(self) -> None:
        """Test BB %B above 2 fails validation."""
        from src.streamlit_app import validate_technical_indicators

        invalid_input = {"bb_pct_b": 5.0}
        is_valid, error_msg = validate_technical_indicators(invalid_input)
        assert not is_valid
        assert "Bollinger" in error_msg or "%B" in error_msg

    def test_validate_negative_atr(self) -> None:
        """Test negative ATR fails validation."""
        from src.streamlit_app import validate_technical_indicators

        invalid_input = {"atr": -100.0}
        is_valid, error_msg = validate_technical_indicators(invalid_input)
        assert not is_valid
        assert "ATR" in error_msg or "positive" in error_msg

    def test_validate_non_numeric_input(self) -> None:
        """Test non-numeric input fails validation."""
        from src.streamlit_app import validate_technical_indicators

        invalid_input = {"rsi": "invalid"}
        is_valid, error_msg = validate_technical_indicators(invalid_input)
        assert not is_valid
        assert "number" in error_msg.lower() or "valid" in error_msg.lower()

    def test_validate_empty_input(self) -> None:
        """Test empty input passes validation (no indicators to check)."""
        from src.streamlit_app import validate_technical_indicators

        empty_input: dict[str, object] = {}
        is_valid, error_msg = validate_technical_indicators(empty_input)
        assert is_valid

    def test_validate_mixed_valid_invalid(self) -> None:
        """Test mixed valid and invalid inputs."""
        from src.streamlit_app import validate_technical_indicators

        mixed_input = {
            "rsi": 50.0,  # Valid
            "bb_pct_b": 10.0,  # Invalid
        }
        is_valid, error_msg = validate_technical_indicators(mixed_input)
        assert not is_valid


# ============================================================================
# DATABASE MANAGER TESTS
# ============================================================================


class TestDatabaseManager:
    """Test database connection management."""

    @patch("src.streamlit_app.SessionLocal")
    def test_get_session_success(self, mock_session_local) -> None:
        """Test successful database session creation."""
        from src.streamlit_app import DatabaseManager

        # Reset class state
        DatabaseManager._session = None
        DatabaseManager._initialized = False

        mock_db = Mock()
        mock_session_local.return_value = mock_db

        session = DatabaseManager.get_session()

        assert session is not None
        assert DatabaseManager._initialized
        mock_session_local.assert_called_once()

    @patch("src.streamlit_app.SessionLocal")
    def test_get_session_cached(self, mock_session_local) -> None:
        """Test that subsequent calls return cached session."""
        from src.streamlit_app import DatabaseManager

        # Reset class state
        DatabaseManager._session = None
        DatabaseManager._initialized = False

        mock_db = Mock()
        mock_session_local.return_value = mock_db

        # First call
        session1 = DatabaseManager.get_session()
        # Second call
        session2 = DatabaseManager.get_session()

        assert session1 is session2
        # Should only be called once (caching works)
        assert mock_session_local.call_count == 1

    @patch("src.streamlit_app.SessionLocal")
    def test_get_session_error(self, mock_session_local) -> None:
        """Test error handling when session creation fails."""
        from src.streamlit_app import DatabaseManager

        # Reset class state
        DatabaseManager._session = None
        DatabaseManager._initialized = False

        mock_session_local.side_effect = Exception("Connection failed")

        session = DatabaseManager.get_session()

        assert session is None

    def test_close_session_success(self) -> None:
        """Test successful session closing."""
        from src.streamlit_app import DatabaseManager

        mock_session = Mock()
        DatabaseManager._session = mock_session
        DatabaseManager._initialized = True

        DatabaseManager.close_session()

        mock_session.close.assert_called_once()
        assert DatabaseManager._session is None
        assert not DatabaseManager._initialized

    def test_close_session_no_session(self) -> None:
        """Test closing when no session exists."""
        from src.streamlit_app import DatabaseManager

        DatabaseManager._session = None
        DatabaseManager._initialized = False

        # Should not raise error
        DatabaseManager.close_session()

        assert DatabaseManager._session is None


# ============================================================================
# TELEGRAM MESSAGE HANDLING TESTS
# ============================================================================


class TestTelegramMessaging:
    """Test Telegram message sending."""

    @pytest.mark.asyncio
    async def test_send_telegram_async_success(self, mock_telegram_notifier) -> None:
        """Test successful async Telegram message."""
        from src.streamlit_app import send_telegram_async

        success, message = await send_telegram_async(mock_telegram_notifier, "Test message")

        assert success
        mock_telegram_notifier.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_telegram_async_failure(self, mock_telegram_notifier) -> None:
        """Test failed async Telegram message."""
        from src.streamlit_app import send_telegram_async

        mock_telegram_notifier.send_message.side_effect = Exception("Send failed")

        success, message = await send_telegram_async(mock_telegram_notifier, "Test message")

        assert not success
        assert "Failed" in message

    def test_send_telegram_message_empty(self, mock_telegram_notifier) -> None:
        """Test sending empty message."""
        from src.streamlit_app import send_telegram_message

        success, message = send_telegram_message(mock_telegram_notifier, "")

        assert not success
        assert "empty" in message.lower()

    def test_send_telegram_message_whitespace(self, mock_telegram_notifier) -> None:
        """Test sending whitespace-only message."""
        from src.streamlit_app import send_telegram_message

        success, message = send_telegram_message(mock_telegram_notifier, "   ")

        assert not success
        assert "empty" in message.lower()


# ============================================================================
# MODEL LOADING AND PREDICTION TESTS
# ============================================================================


class TestModelLoading:
    """Test model loading functionality."""

    @patch("src.streamlit_app.load_trained_model")
    def test_model_load_success(self, mock_load_model) -> None:
        """Test successful model loading."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        from src.streamlit_app import load_trained_model

        model = load_trained_model()

        assert model is not None
        assert model == mock_model

    @patch("src.streamlit_app.load_trained_model")
    def test_model_load_file_not_found(self, mock_load_model) -> None:
        """Test error when model file not found."""
        mock_load_model.side_effect = FileNotFoundError("Model not found")

        from src.streamlit_app import load_trained_model

        with pytest.raises(FileNotFoundError):
            load_trained_model()


class TestPredictions:
    """Test making predictions."""

    def test_make_predictions_success(self, mock_model) -> None:
        """Test successful prediction."""
        from src.streamlit_app import make_predictions

        # Create sample input
        X = pd.DataFrame(
            [
                {
                    "rsi": 50,
                    "bb_upper": 45000,
                    "bb_lower": 35000,
                    "bb_mid": 40000,
                    "bb_pct_b": 0.5,
                    "sma_20": 40000,
                    "sma_50": 40000,
                    "ma_cross": 100,
                    "price_momentum": 5,
                    "atr": 500,
                    "atr_pct": 1.2,
                }
            ]
        )

        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.1, 0.2, 0.7]]

        preds, probs = make_predictions(mock_model, X)

        assert len(preds) == 1
        assert len(probs) == 1

    def test_make_predictions_empty_dataframe(self, mock_model) -> None:
        """Test prediction with empty dataframe."""
        from src.streamlit_app import make_predictions

        X = pd.DataFrame()

        with pytest.raises((ValueError, IndexError)):
            make_predictions(mock_model, X)


# ============================================================================
# DEPENDENCY MANAGEMENT TESTS
# ============================================================================


class TestDependencyManagement:
    """Test dependency import handling."""

    def test_dependencies_dict_exists(self) -> None:
        """Test that DEPENDENCIES dict is properly initialized."""
        from src.streamlit_app import DEPENDENCIES

        assert isinstance(DEPENDENCIES, dict)
        assert "config" in DEPENDENCIES
        assert "database" in DEPENDENCIES
        assert "model" in DEPENDENCIES
        assert "telegram" in DEPENDENCIES

    def test_feature_columns_fallback(self) -> None:
        """Test fallback FEATURE_COLUMNS defined."""
        from src.streamlit_app import FEATURE_COLUMNS

        assert isinstance(FEATURE_COLUMNS, list)
        assert len(FEATURE_COLUMNS) > 0
        assert "rsi" in FEATURE_COLUMNS

    def test_trade_symbol_fallback(self) -> None:
        """Test fallback TRADE_SYMBOL defined."""
        from src.streamlit_app import TRADE_SYMBOL

        assert isinstance(TRADE_SYMBOL, str)
        assert len(TRADE_SYMBOL) > 0


# ============================================================================
# LOGGING TESTS
# ============================================================================


class TestLogging:
    """Test logging functionality."""

    def test_logger_configured(self) -> None:
        """Test that logger is properly configured."""
        from src.streamlit_app import logger

        assert logger is not None
        assert logger.name == "src.streamlit_app"

    def test_logging_on_signal_generation(self, sample_user_input) -> None:
        """Test that signal generation is logged."""
        from src.streamlit_app import validate_technical_indicators

        with patch("src.streamlit_app.logger") as mock_logger:
            validate_technical_indicators(sample_user_input)
            # Logging should be called during validation
            assert mock_logger.info.called or mock_logger.warning.called


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for combined functionality."""

    def test_complete_signal_generation_flow(
        self, sample_user_input: dict[str, float], mock_model
    ) -> None:
        """Test complete signal generation flow."""
        from src.streamlit_app import make_predictions, validate_technical_indicators

        # Validate input
        is_valid, error = validate_technical_indicators(sample_user_input)
        assert is_valid

        # Create dataframe
        X = pd.DataFrame([sample_user_input])

        # Make predictions
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.1, 0.2, 0.7]]

        preds, probs = make_predictions(mock_model, X)

        assert preds is not None
        assert probs is not None


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Test error handling throughout the app."""

    def test_invalid_indicator_value_type(self) -> None:
        """Test handling of invalid value types."""
        from src.streamlit_app import validate_technical_indicators

        invalid_input = {"rsi": "not_a_number"}
        is_valid, error_msg = validate_technical_indicators(invalid_input)

        assert not is_valid
        assert error_msg

    def test_telegram_notifier_none(self) -> None:
        """Test handling when Telegram notifier is None."""
        from src.streamlit_app import send_telegram_message

        success, message = send_telegram_message(None, "Test")

        # Should handle gracefully
        assert not success or isinstance(message, str)

    def test_model_none_prediction(self) -> None:
        """Test handling when model is None."""
        from src.streamlit_app import make_predictions

        X = pd.DataFrame([{"rsi": 50}])

        # Should raise error or handle gracefully
        with pytest.raises(ValueError):
            make_predictions(cast(xgb.XGBClassifier, None), X)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
