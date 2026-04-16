import asyncio
import logging
import signal
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Mock configuration values if they are used directly in the module before any test setup
# These are internal to the module, so we'll patch them where needed or rely on the fixture
# to set up mocks for functions that consume them.
# However, `FEATURE_COLUMNS` is used for DataFrame creation, so it's good to define.
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
ATR_WINDOW = 14  # Dummy value for tests, actual value from src.config
INITIAL_CANDLES_HISTORY = 100  # Dummy value for tests, actual value from src.config

# Import the module under test. This will run module-level code,
# so patching `os.getenv` for `DEFAULT_INTERVAL` and `CONCURRENCY_LIMIT`
# needs to happen *before* this import if the constants are evaluated at import time.
# For simplicity, we'll patch the constants *in* the `br` module directly if needed.
import src.bot_runner as br


# Define the mock `Trade` model used for SQLAlchemy persistence
class MockTrade:
    """A mock SQLAlchemy Trade model for testing persistence."""

    def __init__(self, symbol, side, price, qty, confidence, id=None, timestamp=None):
        self.id = id if id is not None else 1
        self.symbol = symbol
        self.side = side
        self.price = price
        self.qty = qty
        self.confidence = confidence
        self.timestamp = timestamp if timestamp is not None else datetime.now(UTC)

    def __repr__(self):
        return f"<MockTrade {self.symbol} {self.side} {self.qty} @ {self.price}>"

    # For comparison in assertions
    def __eq__(self, other):
        if not isinstance(other, MockTrade):
            return NotImplemented
        return (
            self.symbol == other.symbol
            and self.side == other.side
            and self.price == other.price
            and self.qty == other.qty
            and self.confidence == other.confidence
        )

    def __hash__(self):
        return hash((self.symbol, self.side, self.price, self.qty, self.confidence))


# --- Helper functions for creating mock data ---
def create_ohlcv_df(rows=100):
    """Generates a dummy OHLCV DataFrame."""
    data = {
        "open_time": pd.to_datetime(pd.date_range(end=datetime.now(UTC), periods=rows, freq="4H")),
        "open": np.random.rand(rows) * 100 + 60000,
        "high": np.random.rand(rows) * 100 + 60100,
        "low": np.random.rand(rows) * 100 + 59900,
        "close": np.random.rand(rows) * 100 + 60050,
        "volume": np.random.rand(rows) * 1000,
    }
    df = pd.DataFrame(data)
    df.set_index("open_time", inplace=True)
    return df


def create_features_df(rows=10, latest_values=None):
    """Generates a dummy features DataFrame."""
    df = pd.DataFrame(np.random.rand(rows, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    if latest_values is not None:
        if len(latest_values) != len(FEATURE_COLUMNS):
            raise ValueError(
                f"latest_values must match FEATURE_COLUMNS length ({len(FEATURE_COLUMNS)})"
            )
        df.iloc[-1] = latest_values
    return df


# --- Pytest Fixtures ---
@pytest.fixture(autouse=True)
def mock_db_trade_model():
    """Patch src.db.Trade to use MockTrade for all tests."""
    with patch("src.db.Trade", new=MockTrade) as _mock_trade:
        yield _mock_trade


@pytest.fixture
def mock_resources(mocker):
    """
    Fixture for common mocks used in lifespan and do_iteration.
    Returns a dict with mocked components.
    """
    mock_model = MagicMock()
    mock_metadata = {"model_name": "test_model"}
    mock_binance = MagicMock()
    mock_notifier = AsyncMock()
    mock_db_session = MagicMock()
    mock_db_session_class = mocker.patch("src.db.SessionLocal", return_value=mock_db_session)

    # Mock specific methods/functions
    mock_calc_indicators = mocker.patch(
        "src.feature_engineer.calculate_technical_indicators", return_value=create_features_df()
    )
    mock_make_predictions = mocker.patch(
        "src.model_manager.make_predictions", return_value=(np.array([0]), np.array([0.5]))
    )
    mock_load_model = mocker.patch(
        "src.model_manager.load_trained_model", return_value=(mock_model, mock_metadata)
    )

    # Setup default return values for mocked methods
    mock_binance.get_latest_ohlcv.return_value = create_ohlcv_df(
        rows=max(INITIAL_CANDLES_HISTORY, ATR_WINDOW + 5)
    )
    mock_notifier.send_message.return_value = True

    resources = {
        "model": mock_model,
        "metadata": mock_metadata,
        "binance": mock_binance,
        "notifier": mock_notifier,
        "db": mock_db_session,
        "db_session_class": mock_db_session_class,
        "src.feature_engineer.calculate_technical_indicators": mock_calc_indicators,
        "src.model_manager.make_predictions": mock_make_predictions,
        "src.model_manager.load_trained_model": mock_load_model,
    }
    return resources


@pytest.fixture
def caplog_bot_runner(caplog):
    """Fixture to capture logs specifically from the 'bot_runner' logger."""
    logger = logging.getLogger("bot_runner")
    logger.propagate = True  # Ensure logs reach the root logger captured by caplog
    with caplog.at_level(logging.INFO, logger="bot_runner"):
        yield caplog
    logger.propagate = False  # Reset propagation


@pytest.fixture
def patch_time():
    """Fixture to patch datetime.now for consistent time testing."""
    with patch("src.bot_runner.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC)
        mock_dt.utcnow.return_value = datetime(2023, 1, 1, 10, 0, 0)  # Some libs might use utcnow
        mock_dt.side_effect = lambda *args, **kw: datetime(
            *args, **kw
        )  # Allow datetime() constructor
        mock_dt.UTC = UTC
        yield mock_dt


# --- Test Cases ---


# --- Test scheduling helpers ---
@pytest.mark.parametrize(
    "last_run, interval, current_time, expected",
    [
        (None, 60, datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC), True),  # No previous run
        (
            datetime(2023, 1, 1, 9, 59, 0, tzinfo=UTC),
            60,
            datetime(2023, 1, 1, 10, 0, 1, tzinfo=UTC),
            True,
        ),  # Interval passed
        (
            datetime(2023, 1, 1, 9, 59, 0, tzinfo=UTC),
            60,
            datetime(2023, 1, 1, 9, 59, 59, tzinfo=UTC),
            False,
        ),  # Interval not passed
        (
            datetime(2023, 1, 1, 9, 59, 0, tzinfo=UTC),
            60,
            datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC),
            True,
        ),  # Exactly interval
        (
            datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC),
            0,
            datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC),
            True,
        ),  # Zero interval
    ],
)
def test_is_time_to_run(last_run, interval, current_time, expected, patch_time):
    """Test is_time_to_run logic for various scenarios."""
    patch_time.now.return_value = current_time
    assert br.is_time_to_run(last_run, interval) == expected


@pytest.mark.parametrize(
    "last_task_finished, expected",
    [
        (True, False),  # Task finished, don't skip
        (False, True),  # Task not finished, skip
    ],
)
def test_should_skip_if_running(last_task_finished, expected):
    """Test should_skip_if_running logic."""
    assert br.should_skip_if_running(last_task_finished) == expected


# --- Test notify_all_channels ---
@pytest.mark.asyncio
# --- Test do_iteration (Core Logic) ---
@pytest.mark.asyncio
async def test_do_iteration_buy_signal(mock_resources, caplog_bot_runner):
    """Test do_iteration with a BUY signal."""
    mock_binance = mock_resources["binance"]
    mock_calc_indicators = mock_resources["src.feature_engineer.calculate_technical_indicators"]
    mock_make_predictions = mock_resources["src.model_manager.make_predictions"]
    mock_notifier = mock_resources["notifier"]
    mock_db_session = mock_resources["db"]

    mock_make_predictions.return_value = (np.array([1]), np.array([0.95]))  # BUY signal

    await br.do_iteration(mock_resources)

    mock_binance.get_latest_ohlcv.assert_called_once()
    mock_calc_indicators.assert_called_once()
    mock_make_predictions.assert_called_once()
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()

    expected_trade = MockTrade(
        symbol="BTCUSDT", side="BUY", price="68000", qty="0.001", confidence=0.95
    )
    added_trade = mock_db_session.add.call_args[0][0]
    # Verify the created trade object
    assert added_trade.symbol == expected_trade.symbol
    assert added_trade.side == expected_trade.side
    assert added_trade.price == expected_trade.price
    assert added_trade.qty == expected_trade.qty
    assert abs(added_trade.confidence - expected_trade.confidence) < 1e-6

    assert "BUY signal detected" in caplog_bot_runner.text
    assert "Trade Executed Successfully!" in caplog_bot_runner.text
    assert "Iteration complete" in caplog_bot_runner.text


@pytest.mark.asyncio
async def test_do_iteration_sell_signal(mock_resources, caplog_bot_runner):
    """Test do_iteration with a SELL signal."""
    mock_make_predictions = mock_resources["src.model_manager.make_predictions"]
    mock_notifier = mock_resources["notifier"]
    mock_db_session = mock_resources["db"]

    mock_make_predictions.return_value = (np.array([-1]), np.array([0.88]))  # SELL signal

    await br.do_iteration(mock_resources)

    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()

    expected_trade = MockTrade(
        symbol="BTCUSDT", side="SELL", price="68200", qty="0.001", confidence=0.88
    )
    added_trade = mock_db_session.add.call_args[0][0]
    assert added_trade.symbol == expected_trade.symbol
    assert added_trade.side == expected_trade.side
    assert added_trade.price == expected_trade.price
    assert added_trade.qty == expected_trade.qty
    assert abs(added_trade.confidence - expected_trade.confidence) < 1e-6

    assert "SELL signal detected" in caplog_bot_runner.text
    assert "Trade Executed Successfully!" in caplog_bot_runner.text


@pytest.mark.asyncio
async def test_do_iteration_hold_signal(mock_resources, caplog_bot_runner):
    """Test do_iteration with a HOLD signal (no trade)."""
    mock_make_predictions = mock_resources["src.model_manager.make_predictions"]
    mock_db_session = mock_resources["db"]

    mock_make_predictions.return_value = (np.array([0]), np.array([0.55]))  # HOLD signal

    await br.do_iteration(mock_resources)

    mock_db_session.add.assert_not_called()
    mock_db_session.commit.assert_not_called()

    assert "No actionable signal this round (HOLD position)." in caplog_bot_runner.text


@pytest.mark.asyncio
async def test_do_iteration_fetch_data_failure(mock_resources, caplog_bot_runner):
    """Test do_iteration handles Binance data fetching errors."""
    mock_resources["binance"].get_latest_ohlcv.side_effect = Exception("Binance API error")

    await br.do_iteration(mock_resources)

    assert "Failed to fetch OHLCV data from Binance" in caplog_bot_runner.text
    mock_resources["src.feature_engineer.calculate_technical_indicators"].assert_not_called()
    mock_resources["src.model_manager.make_predictions"].assert_not_called()
    mock_resources["db"].add.assert_not_called()


@pytest.mark.asyncio
async def test_do_iteration_calculate_indicators_failure(mock_resources, caplog_bot_runner):
    """Test do_iteration handles technical indicator calculation errors."""
    mock_resources["src.feature_engineer.calculate_technical_indicators"].side_effect = Exception(
        "Calc error"
    )

    await br.do_iteration(mock_resources)

    assert "Failed to calculate technical indicators" in caplog_bot_runner.text
    mock_resources["src.model_manager.make_predictions"].assert_not_called()
    mock_resources["db"].add.assert_not_called()


@pytest.mark.asyncio
async def test_do_iteration_missing_features(mock_resources, caplog_bot_runner):
    """Test do_iteration handles missing required features after calculation."""
    # Create a features_df missing one of the FEATURE_COLUMNS
    df_missing_feature = pd.DataFrame(
        np.random.rand(10, len(FEATURE_COLUMNS) - 1), columns=FEATURE_COLUMNS[:-1]
    )
    mock_resources[
        "src.feature_engineer.calculate_technical_indicators"
    ].return_value = df_missing_feature

    await br.do_iteration(mock_resources)

    assert f"Missing required features: ['{FEATURE_COLUMNS[-1]}']" in caplog_bot_runner.text
    mock_resources["src.model_manager.make_predictions"].assert_not_called()
    mock_resources["db"].add.assert_not_called()


@pytest.mark.asyncio
async def test_do_iteration_empty_features_df(mock_resources, caplog_bot_runner):
    """Test do_iteration handles an empty features DataFrame."""
    mock_resources[
        "src.feature_engineer.calculate_technical_indicators"
    ].return_value = pd.DataFrame(columns=FEATURE_COLUMNS)

    await br.do_iteration(mock_resources)

    assert "No valid features after computation (all NaN rows dropped)" in caplog_bot_runner.text
    mock_resources["src.model_manager.make_predictions"].assert_not_called()
    mock_resources["db"].add.assert_not_called()


@pytest.mark.asyncio
async def test_do_iteration_prediction_value_error(mock_resources, caplog_bot_runner):
    """Test do_iteration handles ValueError during prediction."""
    mock_resources["src.model_manager.make_predictions"].side_effect = ValueError(
        "Invalid input for model"
    )

    await br.do_iteration(mock_resources)

    assert "Prediction error: Invalid input for model" in caplog_bot_runner.text
    mock_resources["db"].add.assert_not_called()


@pytest.mark.asyncio
async def test_do_iteration_prediction_generic_exception(mock_resources, caplog_bot_runner):
    """Test do_iteration handles generic Exception during prediction."""
    mock_resources["src.model_manager.make_predictions"].side_effect = Exception("Model crashed")

    await br.do_iteration(mock_resources)

    assert "Prediction failed" in caplog_bot_runner.text
    assert "exception" in caplog_bot_runner.text  # Check for exception traceback
    mock_resources["db"].add.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "preds_format, conf_format",
    [
        (lambda: pd.DataFrame({"pred": [1]}), lambda: pd.DataFrame({"conf": [0.9]})),
        (lambda: [1], lambda: [0.9]),
        (lambda: np.array([1]), lambda: np.array([0.9])),
        (lambda: pd.DataFrame({"pred": []}), lambda: pd.DataFrame({"conf": []})),  # Empty DataFrame
        (lambda: [], lambda: []),  # Empty list
        (lambda: np.array([]), lambda: np.array([])),  # Empty numpy array
    ],
)
async def test_do_iteration_prediction_output_formats(mock_resources, preds_format, conf_format):
    """Test do_iteration correctly extracts latest signal/confidence from various formats."""
    mock_make_predictions = mock_resources["src.model_manager.make_predictions"]
    mock_db_session = mock_resources["db"]

    preds = preds_format()
    conf = conf_format()
    mock_make_predictions.return_value = (preds, conf)

    await br.do_iteration(mock_resources)

    if (isinstance(preds, (list, np.ndarray)) and len(preds) > 0) or (
        isinstance(preds, pd.DataFrame) and not preds.empty
    ):
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        assert "Trade Executed Successfully!" in caplog_bot_runner.text
    else:
        mock_db_session.add.assert_not_called()
        mock_db_session.commit.assert_not_called()
        assert "No actionable signal this round" in caplog_bot_runner.text


@pytest.mark.asyncio
async def test_do_iteration_db_persistence_failure(mock_resources, caplog_bot_runner):
    """Test do_iteration handles DB persistence errors (add/commit)."""
    mock_make_predictions = mock_resources["src.model_manager.make_predictions"]
    mock_db_session = mock_resources["db"]

    mock_make_predictions.return_value = (np.array([1]), np.array([0.9]))  # BUY signal
    mock_db_session.commit.side_effect = Exception("DB commit error")

    await br.do_iteration(mock_resources)

    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.rollback.assert_called_once()  # Rollback should be called
    assert "DB persistence failed" in caplog_bot_runner.text


@pytest.mark.asyncio
async def test_do_iteration_db_rollback_failure(mock_resources, caplog_bot_runner):
    """Test do_iteration handles DB rollback errors after commit failure."""
    mock_make_predictions = mock_resources["src.model_manager.make_predictions"]
    mock_db_session = mock_resources["db"]

    mock_make_predictions.return_value = (np.array([1]), np.array([0.9]))  # BUY signal
    mock_db_session.commit.side_effect = Exception("DB commit error")
    mock_db_session.rollback.side_effect = Exception("DB rollback error")

    await br.do_iteration(mock_resources)

    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.rollback.assert_called_once()
    assert "DB persistence failed" in caplog_bot_runner.text
    assert "DB rollback failed" in caplog_bot_runner.text


# --- Test runner_loop ---
@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
@patch("asyncio.wait_for", new_callable=AsyncMock)
@patch("asyncio.create_task")
@patch("asyncio.gather", new_callable=AsyncMock)
async def test_runner_loop_run_once(
    mock_gather, mock_create_task, mock_wait_for, mock_sleep, mock_resources, caplog_bot_runner
):
    """Test runner_loop in run_once mode."""
    # Ensure do_iteration is mocked properly as it's called inside
    mock_do_iteration = mock_resources["src.bot_runner.do_iteration"]
    mock_do_iteration.return_value = None  # No-op for do_iteration

    # Simulate immediate stop after one iteration for `wait_for`
    mock_wait_for.side_effect = [TimeoutError, asyncio.CancelledError]

    # Test runner_loop with run_once=True
    await br.runner_loop(run_once=True, interval_seconds=1)

    mock_do_iteration.assert_awaited_once()
    mock_wait_for.assert_not_awaited()  # In run_once, it breaks before wait_for
    mock_sleep.assert_not_awaited()
    assert "Run-once mode finished; exiting loop" in caplog_bot_runner.text
    # In run_once mode, the loop directly breaks after one run and does not `await asyncio.wait_for`
    # or process signals, so `stop_event.set()` isn't hit from a signal.
    # The `_run_task` is awaited inside `asyncio.gather` for cleanup if there are pending tasks.
    # mock_gather.assert_awaited_once_with(mock_create_task.return_value, return_exceptions=True)


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
@patch("asyncio.wait_for", new_callable=AsyncMock)
@patch("asyncio.create_task")
@patch("asyncio.get_running_loop")
@patch("asyncio.Semaphore", new_callable=MagicMock)
async def test_runner_loop_continuous_with_signal(
    mock_semaphore_class,
    mock_get_running_loop,
    mock_create_task,
    mock_wait_for,
    mock_sleep,
    mock_resources,
    caplog_bot_runner,
    patch_time,
):
    """Test runner_loop in continuous mode, stopping via signal."""
    # Mock do_iteration to ensure it runs
    mock_do_iteration = mock_resources["src.bot_runner.do_iteration"]
    mock_do_iteration.return_value = None

    # Mock the signal handling setup
    mock_loop = AsyncMock()
    mock_get_running_loop.return_value = mock_loop
    mock_loop.add_signal_handler.return_value = None

    # Simulate multiple iterations and then a stop event
    iteration_count = 0

    def do_iteration_side_effect(*args, **kwargs):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count >= 2:  # Stop after 2 iterations
            mock_resources["stop_event"].set()  # Set the stop event to exit the loop
        return None

    mock_do_iteration.side_effect = do_iteration_side_effect

    # Make `wait_for` timeout once, then detect the stop event
    mock_wait_for.side_effect = [
        TimeoutError,
        asyncio.CancelledError,
    ]  # Timeout for first loop, then exit

    # To control time for is_time_to_run
    current_time = datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC)
    patch_time.now.side_effect = [
        current_time,  # First is_time_to_run check
        current_time + timedelta(seconds=1),  # After 1st iteration to trigger next
        current_time + timedelta(seconds=2),  # After 2nd iteration to trigger next
    ]

    # Patch the semaphore acquire/release for easier testing without actual concurrency
    mock_semaphore = AsyncMock()
    mock_semaphore_class.return_value = mock_semaphore
    mock_semaphore.__aenter__.return_value = None
    mock_semaphore.__aexit__.return_value = False

    # Patch asyncio.create_task to return a simple mock task
    mock_task = AsyncMock()
    mock_create_task.return_value = mock_task
    mock_task.add_done_callback.return_value = None

    # Mock the stop_event
    stop_event = AsyncMock(spec=asyncio.Event)
    stop_event.is_set.side_effect = [False, False, False, True]  # Loop twice, then exit
    stop_event.wait.return_value = True  # Makes await stop_event.wait() finish
    mock_resources["stop_event"] = stop_event  # Store it for access by side_effect

    with patch("asyncio.Event", return_value=stop_event):
        await br.runner_loop(run_once=False, interval_seconds=1)

    assert mock_do_iteration.await_count == 2  # Expect two successful iterations
    mock_create_task.assert_called_with(pytest.anything())  # Check task was created
    assert (
        "Runner loop exiting; waiting for background tasks to finish..." in caplog_bot_runner.text
    )
    mock_loop.add_signal_handler.assert_any_call(signal.SIGINT, pytest.anything())
    mock_loop.add_signal_handler.assert_any_call(signal.SIGTERM, pytest.anything())
    assert stop_event.is_set.called
    assert stop_event.wait.called


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
@patch("asyncio.wait_for", new_callable=AsyncMock)
@patch("asyncio.create_task")
@patch("asyncio.get_running_loop")
@patch("asyncio.gather", new_callable=AsyncMock)
async def test_runner_loop_iteration_failure_continues(
    mock_gather,
    mock_get_running_loop,
    mock_create_task,
    mock_wait_for,
    mock_sleep,
    mock_resources,
    caplog_bot_runner,
    patch_time,
):
    """Test runner_loop continues even if an iteration fails."""
    mock_do_iteration = mock_resources["src.bot_runner.do_iteration"]
    # First iteration fails, second succeeds
    mock_do_iteration.side_effect = [Exception("Iteration failed!"), None]

    # Mock the signal handling setup
    mock_loop = AsyncMock()
    mock_get_running_loop.return_value = mock_loop
    mock_loop.add_signal_handler.return_value = None

    # Simulate two iterations and then a stop
    stop_event = AsyncMock(spec=asyncio.Event)
    stop_event.is_set.side_effect = [False, False, False, True]
    stop_event.wait.return_value = True
    mock_resources["stop_event"] = stop_event  # Store for access by side_effect

    with patch("asyncio.Event", return_value=stop_event):
        await br.runner_loop(run_once=False, interval_seconds=1)

    assert mock_do_iteration.await_count == 2
    assert "Iteration failed: Iteration failed!" in caplog_bot_runner.text
    assert "Runner loop exiting" in caplog_bot_runner.text


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
@patch("asyncio.wait_for", new_callable=AsyncMock)
@patch("asyncio.create_task")
@patch("asyncio.get_running_loop")
async def test_runner_loop_skips_if_previous_running(
    mock_get_running_loop,
    mock_create_task,
    mock_wait_for,
    mock_sleep,
    mock_resources,
    caplog_bot_runner,
    patch_time,
):
    """Test runner_loop skips new runs if the previous task is still in progress."""
    mock_do_iteration = mock_resources["src.bot_runner.do_iteration"]
    # do_iteration will await, but we control the last_run_finished flag directly
    # In this test, we want to simulate the previous task *not* finishing quickly.
    # The `_run_task`'s `finally` block sets `last_run_finished = True`.
    # We can make `do_iteration` a long-running async mock to keep the flag False.
    mock_do_iteration.return_value = asyncio.sleep(0.01)  # Simulate a short delay

    mock_loop = AsyncMock()
    mock_get_running_loop.return_value = mock_loop

    stop_event = AsyncMock(spec=asyncio.Event)
    # Simulate first iteration starts (is_set=F), second check (is_set=F, but skip=T), then stop (is_set=T)
    stop_event.is_set.side_effect = [False, False, False, True]
    mock_resources["stop_event"] = stop_event

    current_time = datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC)
    patch_time.now.side_effect = [
        current_time,  # First is_time_to_run check (True)
        current_time
        + timedelta(
            seconds=1
        ),  # Second is_time_to_run check (True), but should_skip_if_running is True
        current_time + timedelta(seconds=2),  # Third check, stop_event.is_set() returns True
    ]

    mock_wait_for.side_effect = [TimeoutError, asyncio.CancelledError]  # Timeout and then exit

    with patch("asyncio.Event", return_value=stop_event):
        await br.runner_loop(run_once=False, interval_seconds=1)

    mock_do_iteration.await_count == 1  # Only one iteration should start
    assert "Previous run still in progress; skipping scheduled run." in caplog_bot_runner.text
    assert "Runner loop exiting" in caplog_bot_runner.text


# --- Test run_once_test ---
@pytest.mark.asyncio
async def test_run_once_test_success(mock_resources, caplog_bot_runner):
    """Test run_once_test happy path."""
    mock_do_iteration = mock_resources["src.bot_runner.do_iteration"]
    mock_do_iteration.return_value = None

    result = await br.run_once_test()

    mock_do_iteration.assert_awaited_once()
    assert result is True
    assert "Test run_once_test() completed successfully" in caplog_bot_runner.text


@pytest.mark.asyncio
async def test_run_once_test_failure(mock_resources, caplog_bot_runner):
    """Test run_once_test handles do_iteration failure."""
    mock_do_iteration = mock_resources["src.bot_runner.do_iteration"]
    mock_do_iteration.side_effect = Exception("Single run error")

    result = await br.run_once_test()

    mock_do_iteration.assert_awaited_once()
    assert result is False
    assert "Error during single test iteration: Single run error" in caplog_bot_runner.text


# --- Test main entry point ---
@patch("src.bot_runner.asyncio.run", new_callable=MagicMock)
@patch("src.bot_runner.run_once_test", new_callable=AsyncMock)
@patch("src.bot_runner.runner_loop", new_callable=AsyncMock)
@patch("os.getenv")
def test_main_run_once_enabled(mock_getenv, mock_runner_loop, mock_run_once_test, mock_asyncio_run):
    """Test main function when BOT_RUN_ONCE is enabled."""
    mock_getenv.side_effect = lambda key, default: {
        "BOT_RUN_ONCE": "1",
        "BOT_INTERVAL_SECONDS": "10",
        "BOT_CONCURRENCY": "1",
    }.get(key, default)

    br.main()

    mock_run_once_test.assert_called_once()
    mock_runner_loop.assert_not_called()
    mock_asyncio_run.assert_called_once_with(mock_run_once_test.return_value)


@patch("src.bot_runner.asyncio.run", new_callable=MagicMock)
@patch("src.bot_runner.run_once_test", new_callable=AsyncMock)
@patch("src.bot_runner.runner_loop", new_callable=AsyncMock)
@patch("os.getenv")
def test_main_run_once_disabled(
    mock_getenv, mock_runner_loop, mock_run_once_test, mock_asyncio_run
):
    """Test main function when BOT_RUN_ONCE is disabled."""
    mock_getenv.side_effect = lambda key, default: {
        "BOT_RUN_ONCE": "0",
        "BOT_INTERVAL_SECONDS": "10",
        "BOT_CONCURRENCY": "1",
    }.get(key, default)

    br.main()

    mock_run_once_test.assert_not_called()
    mock_runner_loop.assert_called_once_with(run_once=False, interval_seconds=10)
    mock_asyncio_run.assert_called_once_with(mock_runner_loop.return_value)


@patch("src.bot_runner.asyncio.run", new_callable=MagicMock)
@patch("src.bot_runner.runner_loop", new_callable=AsyncMock)
@patch("os.getenv")
def test_main_uses_default_interval_if_not_set(mock_getenv, mock_runner_loop, mock_asyncio_run):
    """Test main uses default interval if BOT_INTERVAL_SECONDS is not set."""
    mock_getenv.side_effect = lambda key, default: {
        "BOT_RUN_ONCE": "0",
        # "BOT_INTERVAL_SECONDS" is missing
    }.get(key, default)

    br.main()

    mock_runner_loop.assert_called_once_with(run_once=False, interval_seconds=br.DEFAULT_INTERVAL)
