"""
Streamlit Trading Signal Assistant Application
FIXED: use_container_width must be bool, not string "stretch"
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import sys
import threading
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("streamlit_app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Add the repository root to Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

# ============================================================================
# DEPENDENCY MANAGEMENT
# ============================================================================

# Track which dependencies are available
DEPENDENCIES = {
    "config": False,
    "database": False,
    "model": False,
    "telegram": False,
}

# Fallback constants
FEATURE_COLUMNS = [
    "rsi",
    "bb_upper",
    "bb_lower",
    "bb_mid",
    "bb_pct_b",
    "sma_20",
    "sma_50",
    "ma_cross",
    "price_momentum",
    "atr",
    "atr_pct",
]
TRADE_SYMBOL = "BTCUSDT"

# Try importing configuration
try:
    from src.config import FEATURE_COLUMNS, TRADE_SYMBOL

    DEPENDENCIES["config"] = True
    logger.info("[OK] Configuration loaded successfully")
except ImportError as e:
    logger.warning(f"[ERROR] Cannot import config: {e}")
    st.warning("⚠️ Using default configuration (custom settings unavailable)")

# Try importing database
try:
    from src.db import SessionLocal, Trade

    DEPENDENCIES["database"] = True
    logger.info("[OK] Database module loaded successfully")
except ImportError as e:
    logger.warning(f"[ERROR] Cannot import database: {e}")

# Try importing model
try:
    from xgboost import XGBClassifier

    from src.model_manager import load_trained_model, make_predictions

    DEPENDENCIES["model"] = True
    logger.info("[OK] Model manager loaded successfully")
except ImportError as e:
    logger.warning(f"[ERROR] Cannot import model manager: {e}")

# Try importing Telegram notifier
try:
    from src.notifications.notifier import TelegramNotifier

    DEPENDENCIES["telegram"] = True
    logger.info("[OK] Telegram notifier loaded successfully")
except ImportError as e:
    logger.warning(f"[ERROR] Cannot import Telegram notifier: {e}")

# ============================================================================
# DATABASE CONNECTION MANAGER
# ============================================================================


class DatabaseManager:
    """Manage database connections safely with pooling."""

    _session: Session | None = None
    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_session(cls) -> Session | None:
        """Get database session with error handling and connection pooling."""
        if not DEPENDENCIES["database"]:
            logger.error("Database module not available")
            return None

        with cls._lock:
            if not cls._initialized:
                try:
                    cls._session = SessionLocal()
                    cls._initialized = True
                    logger.info("Database session initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize database: {e}", exc_info=True)
                    return None

        return cls._session

    @classmethod
    def close_session(cls) -> None:
        """Close database session safely."""
        with cls._lock:
            if cls._session:
                try:
                    cls._session.close()
                    cls._session = None
                    cls._initialized = False
                    logger.info("Database session closed")
                except Exception as e:
                    logger.error(f"Error closing session: {e}", exc_info=True)


# ============================================================================
# INPUT VALIDATION
# ============================================================================


def validate_technical_indicators(indicators: Mapping[str, object]) -> tuple[bool, str]:
    """
    Validate technical indicator values.

    Args:
        indicators: Dictionary of indicator values

    Returns:
        Tuple of (is_valid, error_message)
    """
    logger.info("Validating technical indicators")
    validation_rules = {
        "rsi": (0, 100, "RSI must be between 0 and 100"),
        "bb_pct_b": (-1, 2, "Bollinger Band %B must be between -1 and 2"),
        "bb_upper": (0, float("inf"), "BB Upper must be positive"),
        "bb_lower": (0, float("inf"), "BB Lower must be positive"),
        "bb_mid": (0, float("inf"), "BB Mid must be positive"),
        "atr": (0, float("inf"), "ATR must be positive"),
        "atr_pct": (0, 100, "ATR % must be between 0 and 100"),
        "sma_20": (0, float("inf"), "SMA 20 must be positive"),
        "sma_50": (0, float("inf"), "SMA 50 must be positive"),
        "ma_cross": (-1000, 1000, "MA Cross must be reasonable value"),
        "price_momentum": (-100, 100, "Price momentum must be between -100 and 100"),
    }

    for indicator, value in indicators.items():
        if indicator in validation_rules:
            min_val, max_val, error_msg = validation_rules[indicator]
            try:
                float_value = float(cast(Any, value))
                if not (min_val <= float_value <= max_val):
                    return False, error_msg
            except (ValueError, TypeError):
                return False, f"{indicator} must be a valid number"

    return True, ""


# ============================================================================
# TELEGRAM NOTIFICATION HANDLER
# ============================================================================


async def send_telegram_async(notifier: Any, message: str) -> tuple[bool, str]:
    """
    Send Telegram message asynchronously.

    Args:
        notifier: TelegramNotifier instance
        message: Message to send

    Returns:
        Tuple of (success, message)
    """
    try:
        await notifier.send_message(message)
        logger.info("Telegram message sent successfully")
        return True, "Message sent to Telegram!"
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}", exc_info=True)
        return False, f"Failed to send message: {str(e)}"


def send_telegram_message(notifier: Any, message: str) -> tuple[bool, str]:
    """
    Send Telegram message with proper async handling.

    Args:
        notifier: TelegramNotifier instance
        message: Message to send

    Returns:
        Tuple of (success, message)
    """
    if not message.strip():
        return False, "Message cannot be empty"

    try:
        # Try to get current event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Event loop is running (Streamlit context), use threading
                logger.info("Using threaded async execution for Telegram")
                result_holder: dict[str, Any] = {"success": False, "message": ""}

                def run_async() -> None:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        success, msg = new_loop.run_until_complete(
                            send_telegram_async(notifier, message)
                        )
                        result_holder["success"] = success
                        result_holder["message"] = msg
                    finally:
                        new_loop.close()

                thread = threading.Thread(target=run_async, daemon=True)
                thread.start()
                thread.join(timeout=10)  # Wait max 10 seconds

                success_val = bool(result_holder["success"])
                message_val = str(result_holder["message"])
                return success_val, message_val
            else:
                # Event loop exists but not running
                success, msg = loop.run_until_complete(send_telegram_async(notifier, message))
                return success, msg
        except RuntimeError:
            # No event loop, create new one
            logger.info("Creating new event loop for Telegram")
            # Type hint for mypy to help with asyncio.run return type inference
            res: tuple[bool, str] = asyncio.run(send_telegram_async(notifier, message))
            success, msg = res
            return success, msg
    except Exception as e:
        logger.error(f"Telegram async error: {e}", exc_info=True)
        return False, f"Telegram error: {str(e)}"


# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Signal Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #c62828;
        padding: 1rem;
        border-radius: 4px;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #2e7d32;
        padding: 1rem;
        border-radius: 4px;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #e65100;
        padding: 1rem;
        border-radius: 4px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Title
st.markdown("<h1 class='main-title'>📈 Signal Trading Assistant</h1>", unsafe_allow_html=True)

# Display system status
with st.sidebar:
    st.markdown("### System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Configuration:**")
        st.write("**Database:**")
        st.write("**Model:**")
        st.write("**Telegram:**")
    with col2:
        st.write("✓" if DEPENDENCIES["config"] else "✗")
        st.write("✓" if DEPENDENCIES["database"] else "✗")
        st.write("✓" if DEPENDENCIES["model"] else "✗")
        st.write("✓" if DEPENDENCIES["telegram"] else "✗")

# ============================================================================
# MODEL LOADING
# ============================================================================

model: XGBClassifier | None = None
if DEPENDENCIES["model"]:

    @st.cache_resource
    def get_model() -> XGBClassifier | None:
        """Load trained model with caching."""
        try:
            logger.info("Loading trained model...")
            loaded_model = load_trained_model()
            logger.info("[OK] Model loaded successfully")
            return loaded_model
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            st.error("❌ Model file not found. Please train a model first.")
            return None
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            st.error(f"❌ Error loading model: {str(e)}")
            return None

    model = get_model()
else:
    st.error(
        "❌ **Model module not available**\n\n"
        "Cannot load trained models. Please install required dependencies: "
        "`pip install xgboost scikit-learn`"
    )

# ============================================================================
# TELEGRAM NOTIFIER INITIALIZATION
# ============================================================================

telegram_notifier: TelegramNotifier | None = None
if DEPENDENCIES["telegram"]:
    try:
        telegram_notifier = TelegramNotifier()
        logger.info("[OK] Telegram notifier initialized")
    except Exception as e:
        logger.warning(f"Could not initialize Telegram Notifier: {e}")
        st.warning(
            f"⚠️ **Telegram not available:** {str(e)}\n\n"
            "Check your TELEGRAM_BOT_TOKEN environment variable."
        )

# ============================================================================
# MAIN LAYOUT
# ============================================================================

col1, col2 = st.columns([2, 1])

# ============================================================================
# RIGHT COLUMN: ANALYSIS & CONTROLS
# ============================================================================

with col2:
    st.markdown("### 🎯 Market Analysis")

    # Technical Indicators Input
    with st.expander("📊 Technical Indicators", expanded=True):
        st.markdown("**Enter technical indicator values:**")

        user_input: dict[str, float] = {}

        # Organize inputs in columns
        col_a, col_b = st.columns(2)

        with col_a:
            user_input["rsi"] = st.number_input(
                "RSI",
                value=50.0,
                min_value=0.0,
                max_value=100.0,
                help="Relative Strength Index (0-100)",
            )
            user_input["bb_pct_b"] = st.number_input(
                "BB %B",
                value=0.5,
                min_value=-1.0,
                max_value=2.0,
                help="Bollinger Band %B (-1 to 2)",
            )
            user_input["atr_pct"] = st.number_input(
                "ATR %", value=2.0, min_value=0.0, max_value=100.0, help="ATR Percentage (0-100)"
            )
            user_input["price_momentum"] = st.number_input(
                "Momentum",
                value=0.0,
                min_value=-100.0,
                max_value=100.0,
                help="Price Momentum (-100 to 100)",
            )

        with col_b:
            user_input["bb_upper"] = st.number_input(
                "BB Upper", value=45000.0, min_value=0.0, help="Upper Bollinger Band"
            )
            user_input["bb_lower"] = st.number_input(
                "BB Lower", value=35000.0, min_value=0.0, help="Lower Bollinger Band"
            )
            user_input["bb_mid"] = st.number_input(
                "BB Mid", value=40000.0, min_value=0.0, help="Middle Bollinger Band"
            )
            user_input["atr"] = st.number_input(
                "ATR", value=500.0, min_value=0.0, help="Average True Range"
            )

        user_input["sma_20"] = st.number_input(
            "SMA 20", value=40000.0, min_value=0.0, help="Simple Moving Average (20 period)"
        )
        user_input["sma_50"] = st.number_input(
            "SMA 50", value=40000.0, min_value=0.0, help="Simple Moving Average (50 period)"
        )
        user_input["ma_cross"] = st.number_input(
            "MA Cross",
            value=0.0,
            min_value=-1000.0,
            max_value=1000.0,
            help="Moving Average Crossover",
        )

        # Signal Generation - FIX: use_container_width=True (bool, not "stretch")
        if st.button("🚀 Generate Signal", use_container_width=True, key="generate_signal"):
            logger.info(f"Signal generation requested with input: {user_input}")

            # Validate input
            is_valid, error_msg = validate_technical_indicators(user_input)
            if not is_valid:
                st.error(f"❌ **Input Validation Error:** {error_msg}")
                logger.warning(f"Input validation failed: {error_msg}")
            elif model is None:
                st.error("❌ **Model not loaded.** Cannot generate signal.")
                logger.error("Signal generation attempted but model is None")
            else:
                try:
                    # Prepare data
                    X = pd.DataFrame([user_input])[FEATURE_COLUMNS]

                    # Validate dataframe
                    if X.isnull().any().any():
                        st.error("❌ Some indicator values are missing. Please fill all fields.")
                        logger.warning("Dataframe contains NaN values")
                    else:
                        # Make prediction
                        preds, probs = make_predictions(model, X)

                        signal_map = {-1: "SELL (RED)", 0: "HOLD", 1: "BUY (GREEN)"}
                        signal = signal_map.get(int(preds[0]), "HOLD")
                        confidence = float(probs[0])

                        logger.info(f"Signal generated: {signal} (confidence: {confidence:.4f})")

                        # Display result
                        st.markdown(
                            f"""
                            <div style='background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                                <h2 style='margin: 0; color: #1E88E5;'>{signal}</h2>
                                <p style='margin: 0.5rem 0; font-size: 1.1rem;'>
                                    <strong>Confidence: {confidence:.2%}</strong>
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Log signal with timestamp
                        signal_timestamp = datetime.now().isoformat()
                        logger.info(
                            f"[{signal_timestamp}] Signal: {signal}, Confidence: {confidence:.4f}"
                        )

                except Exception as e:
                    logger.error(f"Error generating signal: {e}", exc_info=True)
                    st.error(f"❌ **Error generating signal:** {str(e)}")

    # Telegram Notification Section
    st.markdown("### 💬 Telegram Notifications")

    if telegram_notifier is None:
        st.warning("⚠️ **Telegram notifier not available.** Check your setup.")
    else:
        telegram_message = st.text_area(
            "Message to send:",
            placeholder="Enter your message here...",
            height=100,
            key="telegram_message",
        )

        # FIX: use_container_width=True (bool, not "stretch")
        if st.button("📤 Send Message", use_container_width=True, key="send_telegram_btn"):
            if not telegram_message.strip():
                st.warning("⚠️ **Message is empty.** Please enter a message to send.")
                logger.warning("Empty message attempted to be sent to Telegram")
            else:
                with st.spinner("📤 Sending message..."):
                    success, message = send_telegram_message(telegram_notifier, telegram_message)

                    if success:
                        st.success(message)
                        logger.info("Telegram message sent successfully")
                    else:
                        st.error(f"❌ {message}")
                        logger.error(f"Failed to send Telegram message: {message}")

# ============================================================================
# LEFT COLUMN: PORTFOLIO OVERVIEW
# ============================================================================

with col1:
    st.markdown("### 💼 Portfolio Overview")

    # Fetch recent trades
    trades_data: list[dict[str, str]] = []
    if DEPENDENCIES["database"]:
        try:
            db = DatabaseManager.get_session()
            if db:
                recent_trades = db.query(Trade).order_by(Trade.timestamp.desc()).limit(10).all()

                if recent_trades:
                    trades_data = [
                        {
                            "Time": trade.timestamp.strftime("%Y-%m-%d %H:%M"),
                            "Action": trade.side.upper(),
                            "Price": f"${trade.price:.2f}",
                            "Quantity": f"{trade.quantity:.4f} {TRADE_SYMBOL[:3]}",
                        }
                        for trade in recent_trades
                    ]
                    logger.info(f"Loaded {len(trades_data)} recent trades from database")
                else:
                    logger.info("No trades found in database")
                    st.info("ℹ️ No recent trades found.")
            else:
                raise ConnectionError("Failed to get database session")
        except Exception as db_error:
            logger.warning(f"Database error: {db_error}")
            st.warning(f"⚠️ **Database unavailable:** {str(db_error)}\n\nUsing demo data instead.")

            # Demo data fallback
            trades_data = [
                {
                    "Time": (datetime.now() - timedelta(minutes=i * 15)).strftime("%Y-%m-%d %H:%M"),
                    "Action": "BUY" if i % 2 == 0 else "SELL",
                    "Price": f"${40000 + (i * 100):.2f}",
                    "Quantity": f"0.1 {TRADE_SYMBOL[:3]}",
                }
                for i in range(5)
            ]
            logger.info("Loaded demo trade data")
    else:
        # Demo data when database unavailable
        trades_data = [
            {
                "Time": (datetime.now() - timedelta(minutes=i * 15)).strftime("%Y-%m-%d %H:%M"),
                "Action": "BUY" if i % 2 == 0 else "SELL",
                "Price": f"${40000 + (i * 100):.2f}",
                "Quantity": f"0.1 {TRADE_SYMBOL[:3]}",
            }
            for i in range(5)
        ]

    # Display trades table - FIX: use_container_width=True (bool, not "stretch")
    if trades_data:
        st.dataframe(pd.DataFrame(trades_data), use_container_width=True, hide_index=True)

    # Performance Chart
    st.markdown("### 📈 Performance Chart")

    try:
        # Generate sample performance data
        dates = pd.date_range(start="2025-01-01", end=datetime.now().date(), freq="D")
        base_value = 10000
        performance_values = [base_value * (1 + i / 200) for i in range(len(dates))]

        performance_df = pd.DataFrame({"date": dates, "value": performance_values})

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=performance_df["date"],
                y=performance_df["value"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#1E88E5", width=2),
                fill="tozeroy",
                fillcolor="rgba(30, 136, 229, 0.1)",
                hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Value:</b> $%{y:,.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Date",
            hovermode="x unified",
            template="plotly_white",
        )

        # FIX: use_container_width=True (bool, not "stretch")
        st.plotly_chart(fig, use_container_width=True)
        logger.info("Performance chart rendered successfully")
    except Exception as e:
        logger.error(f"Error rendering performance chart: {e}", exc_info=True)
        st.error(f"❌ Could not render chart: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>📊 Trading Signal Assistant v1.0 |
        <a href='#'>Documentation</a> |
        <a href='#'>Report Issue</a></p>
        <p>Last updated: """
    + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    + """</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Cleanup on exit
atexit.register(DatabaseManager.close_session)

logger.info("Streamlit app rendered successfully")
