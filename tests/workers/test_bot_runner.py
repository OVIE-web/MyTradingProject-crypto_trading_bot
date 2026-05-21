from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import app.workers.bot_runner as bot_runner


class FakeTrade:
    """Small trade stand-in so worker tests do not require SQLAlchemy instrumentation."""

    def __init__(
        self,
        *,
        symbol: str,
        side: str,
        price: str,
        qty: str,
        confidence: float,
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.price = Decimal(price)
        self.qty = Decimal(qty)
        self.confidence = confidence


def make_candles(rows: int = 80) -> pd.DataFrame:
    index = pd.date_range("2026-05-16", periods=rows, freq="4h", tz=UTC)
    close = np.linspace(68_000.0, 69_000.0, rows)
    return pd.DataFrame(
        {
            "open": close - 50.0,
            "high": close + 100.0,
            "low": close - 100.0,
            "close": close,
            "volume": np.linspace(10.0, 100.0, rows),
        },
        index=index,
    )


def make_features(rows: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        np.ones((rows, len(bot_runner.FEATURE_COLUMNS))),
        columns=bot_runner.FEATURE_COLUMNS,
    )


@pytest.fixture
def resources() -> dict[str, object]:
    binance = MagicMock()
    binance.get_latest_ohlcv.return_value = make_candles()

    notifier = AsyncMock()
    notifier.send_message.return_value = True

    return {
        "model": MagicMock(name="model"),
        "metadata": {"model_name": "test_model"},
        "binance": binance,
        "notifier": notifier,
        "db": MagicMock(name="db_session"),
    }


@asynccontextmanager
async def fake_lifespan(resources: dict[str, object]) -> Iterator[dict[str, object]]:
    yield resources


@pytest.mark.parametrize(
    ("last_run", "interval_seconds", "now", "expected"),
    [
        (None, 60, datetime(2026, 5, 16, 12, 0, tzinfo=UTC), True),
        (
            datetime(2026, 5, 16, 11, 59, tzinfo=UTC),
            60,
            datetime(2026, 5, 16, 12, 0, 1, tzinfo=UTC),
            True,
        ),
        (
            datetime(2026, 5, 16, 11, 59, 30, tzinfo=UTC),
            60,
            datetime(2026, 5, 16, 12, 0, tzinfo=UTC),
            False,
        ),
    ],
)
def test_is_time_to_run(
    last_run: datetime | None,
    interval_seconds: int,
    now: datetime,
    expected: bool,
) -> None:
    with patch("app.workers.bot_runner.datetime") as datetime_mock:
        datetime_mock.now.return_value = now

        assert bot_runner.is_time_to_run(last_run, interval_seconds) is expected


@pytest.mark.parametrize(
    ("last_task_finished", "expected"),
    [(True, False), (False, True)],
)
def test_should_skip_if_running(last_task_finished: bool, expected: bool) -> None:
    assert bot_runner.should_skip_if_running(last_task_finished) is expected


@pytest.mark.asyncio
async def test_lifespan_initializes_and_closes_resources() -> None:
    db = MagicMock()
    model = MagicMock()
    metadata = {"model_name": "xgboost"}

    with (
        patch("app.workers.bot_runner.load_trained_model", return_value=(model, metadata)),
        patch("app.workers.bot_runner.BinanceManager") as binance_cls,
        patch("app.workers.bot_runner.TelegramNotifier") as notifier_cls,
        patch("app.workers.bot_runner.SessionLocal", return_value=db),
    ):
        async with bot_runner.lifespan() as loaded:
            assert loaded["model"] is model
            assert loaded["metadata"] == metadata
            assert loaded["binance"] is binance_cls.return_value
            assert loaded["notifier"] is notifier_cls.return_value
            assert loaded["db"] is db

    db.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_notify_all_channels_uses_primary_channels_when_successful() -> None:
    notifier = AsyncMock()
    notifier.send_message.return_value = True

    with (
        patch("app.workers.bot_runner.send_telegram_notification") as telegram,
        patch("app.workers.bot_runner.send_email_async_safe", return_value=True) as email_primary,
        patch("app.workers.bot_runner.send_email_sync") as email_fallback,
    ):
        await bot_runner.notify_all_channels(notifier, "Subject", "Message")

    notifier.send_message.assert_awaited_once_with("Message")
    email_primary.assert_called_once_with("Subject", "Message")
    telegram.assert_not_called()
    email_fallback.assert_not_called()


@pytest.mark.asyncio
async def test_notify_all_channels_uses_sync_fallbacks_when_primary_channels_fail() -> None:
    notifier = AsyncMock()
    notifier.send_message.side_effect = RuntimeError("telegram unavailable")

    with (
        patch("app.workers.bot_runner.send_telegram_notification") as telegram,
        patch(
            "app.workers.bot_runner.send_email_async_safe",
            side_effect=RuntimeError("email unavailable"),
        ),
        patch("app.workers.bot_runner.send_email_sync") as email_fallback,
    ):
        await bot_runner.notify_all_channels(notifier, "Subject", "Message")

    telegram.assert_called_once_with("Message")
    email_fallback.assert_called_once_with("Subject", "Message")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("signal", "confidence", "expected_side", "expected_price"),
    [(1, 0.95, "BUY", Decimal("68000")), (-1, 0.88, "SELL", Decimal("68200"))],
)
async def test_do_iteration_persists_actionable_trade(
    resources: dict[str, object],
    signal: int,
    confidence: float,
    expected_side: str,
    expected_price: Decimal,
) -> None:
    db = resources["db"]

    with (
        patch(
            "app.workers.bot_runner.calculate_technical_indicators", return_value=make_features()
        ),
        patch(
            "app.workers.bot_runner.make_predictions",
            return_value=(np.array([signal]), np.array([confidence])),
        ),
        patch("app.workers.bot_runner.notify_all_channels", new_callable=AsyncMock) as notify,
        patch("app.workers.bot_runner.Trade", FakeTrade),
    ):
        await bot_runner.do_iteration(resources)

    db.add.assert_called_once()
    db.commit.assert_called_once()
    notify.assert_awaited_once()

    trade = db.add.call_args.args[0]
    assert isinstance(trade, FakeTrade)
    assert trade.symbol == "BTCUSDT"
    assert trade.side == expected_side
    assert trade.price == expected_price
    assert trade.qty == Decimal("0.001")
    assert trade.confidence == confidence


@pytest.mark.asyncio
async def test_do_iteration_hold_signal_does_not_persist_trade(
    resources: dict[str, object],
) -> None:
    db = resources["db"]

    with (
        patch(
            "app.workers.bot_runner.calculate_technical_indicators", return_value=make_features()
        ),
        patch(
            "app.workers.bot_runner.make_predictions",
            return_value=(np.array([0]), np.array([0.55])),
        ),
        patch("app.workers.bot_runner.notify_all_channels", new_callable=AsyncMock) as notify,
    ):
        await bot_runner.do_iteration(resources)

    db.add.assert_not_called()
    db.commit.assert_not_called()
    notify.assert_not_awaited()


@pytest.mark.asyncio
async def test_do_iteration_stops_when_market_data_fetch_fails(
    resources: dict[str, object],
) -> None:
    binance = resources["binance"]
    binance.get_latest_ohlcv.side_effect = RuntimeError("binance down")

    with (
        patch("app.workers.bot_runner.calculate_technical_indicators") as indicators,
        patch("app.workers.bot_runner.make_predictions") as predictions,
    ):
        await bot_runner.do_iteration(resources)

    indicators.assert_not_called()
    predictions.assert_not_called()
    resources["db"].add.assert_not_called()


@pytest.mark.asyncio
async def test_do_iteration_stops_when_required_features_are_missing(
    resources: dict[str, object],
) -> None:
    missing_features = make_features().drop(columns=[bot_runner.FEATURE_COLUMNS[-1]])

    with (
        patch(
            "app.workers.bot_runner.calculate_technical_indicators",
            return_value=missing_features,
        ),
        patch("app.workers.bot_runner.make_predictions") as predictions,
    ):
        await bot_runner.do_iteration(resources)

    predictions.assert_not_called()
    resources["db"].add.assert_not_called()


@pytest.mark.asyncio
async def test_do_iteration_stops_when_feature_frame_is_empty(
    resources: dict[str, object],
) -> None:
    empty_features = pd.DataFrame(columns=bot_runner.FEATURE_COLUMNS)

    with (
        patch(
            "app.workers.bot_runner.calculate_technical_indicators",
            return_value=empty_features,
        ),
        patch("app.workers.bot_runner.make_predictions") as predictions,
    ):
        await bot_runner.do_iteration(resources)

    predictions.assert_not_called()
    resources["db"].add.assert_not_called()


@pytest.mark.asyncio
async def test_do_iteration_stops_when_prediction_fails(resources: dict[str, object]) -> None:
    with (
        patch(
            "app.workers.bot_runner.calculate_technical_indicators", return_value=make_features()
        ),
        patch("app.workers.bot_runner.make_predictions", side_effect=ValueError("bad features")),
    ):
        await bot_runner.do_iteration(resources)

    resources["db"].add.assert_not_called()


@pytest.mark.asyncio
async def test_do_iteration_rolls_back_when_persistence_fails(
    resources: dict[str, object],
) -> None:
    db = resources["db"]
    db.commit.side_effect = RuntimeError("db down")

    with (
        patch(
            "app.workers.bot_runner.calculate_technical_indicators", return_value=make_features()
        ),
        patch(
            "app.workers.bot_runner.make_predictions",
            return_value=(np.array([1]), np.array([0.91])),
        ),
        patch("app.workers.bot_runner.notify_all_channels", new_callable=AsyncMock),
        patch("app.workers.bot_runner.Trade", FakeTrade),
    ):
        await bot_runner.do_iteration(resources)

    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.rollback.assert_called_once()


@pytest.mark.asyncio
async def test_run_once_test_returns_true_when_iteration_succeeds(
    resources: dict[str, object],
) -> None:
    with (
        patch("app.workers.bot_runner.lifespan", return_value=fake_lifespan(resources)),
        patch("app.workers.bot_runner.do_iteration", new_callable=AsyncMock) as iteration,
    ):
        result = await bot_runner.run_once_test()

    assert result is True
    iteration.assert_awaited_once_with(resources)


@pytest.mark.asyncio
async def test_run_once_test_returns_false_when_iteration_raises(
    resources: dict[str, object],
) -> None:
    with (
        patch("app.workers.bot_runner.lifespan", return_value=fake_lifespan(resources)),
        patch(
            "app.workers.bot_runner.do_iteration",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ),
    ):
        result = await bot_runner.run_once_test()

    assert result is False


@pytest.mark.asyncio
async def test_runner_loop_run_once_executes_one_iteration(resources: dict[str, object]) -> None:
    with (
        patch("app.workers.bot_runner.lifespan", return_value=fake_lifespan(resources)),
        patch("app.workers.bot_runner.do_iteration", new_callable=AsyncMock) as iteration,
    ):
        await bot_runner.runner_loop(run_once=True, interval_seconds=1)

    iteration.assert_awaited_once_with(resources)


def test_main_runs_single_iteration_when_run_once_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_RUN_ONCE", "1")
    monkeypatch.setenv("BOT_INTERVAL_SECONDS", "10")

    with (
        patch(
            "app.workers.bot_runner.run_once_test",
            new=MagicMock(return_value="run-once-coro"),
        ) as run_once,
        patch("app.workers.bot_runner.asyncio.run") as asyncio_run,
    ):
        bot_runner.main()

    run_once.assert_called_once_with()
    asyncio_run.assert_called_once_with("run-once-coro")


def test_main_runs_loop_when_run_once_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_RUN_ONCE", "0")
    monkeypatch.setenv("BOT_INTERVAL_SECONDS", "15")

    with (
        patch(
            "app.workers.bot_runner.runner_loop",
            new=MagicMock(return_value="loop-coro"),
        ) as loop,
        patch("app.workers.bot_runner.asyncio.run") as asyncio_run,
    ):
        bot_runner.main()

    loop.assert_called_once_with(run_once=False, interval_seconds=15)
    asyncio_run.assert_called_once_with("loop-coro")
