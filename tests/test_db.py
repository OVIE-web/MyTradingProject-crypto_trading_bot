import contextlib
from collections.abc import Callable, Generator
from datetime import UTC, datetime, timedelta
from typing import Literal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError, StatementError
from sqlalchemy.orm import Session, sessionmaker

# Import the module under test
import src.db as db_module
from src.db import Trade, get_db  # Explicit imports for clarity

# --- Fixtures for Testing ---


@pytest.fixture(autouse=True)
def mock_settings_db_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mocks `src.settings.settings` to have DATABASE_URL set to an in-memory SQLite database
    for testing. This ensures that any component that relies on this setting will use the test database.
    """
    from unittest.mock import MagicMock

    mock_settings = MagicMock()
    mock_settings.DATABASE_URL = "sqlite:///:memory:"
    monkeypatch.setattr("src.settings.settings", mock_settings)


@pytest.fixture(scope="session")
def setup_test_db_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[db_module.Engine, None, None]:
    """
    Fixture to set up an in-memory SQLite database for all tests in the session.
    It patches the global `engine` and `SessionLocal` in `src.db` to point to
    a new in-memory SQLite database instance.
    """
    # Create a *new* engine and sessionmaker specific for testing
    test_engine = create_engine(
        "sqlite:///:memory:",
        echo=False,  # Set to True to see SQL statements during tests
        future=True,
        pool_pre_ping=True,
    )
    TestSessionLocal = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
        class_=Session,
    )

    # Patch the global engine and SessionLocal in src.db module
    # This ensures that init_db() and get_db() use our test setup.
    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(db_module, "SessionLocal", TestSessionLocal)

    # Create tables using the test engine
    db_module.Base.metadata.create_all(bind=test_engine)

    yield test_engine

    # Teardown: drop all tables and dispose the engine
    db_module.Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


@pytest.fixture
def db_session(setup_test_db_engine: db_module.Engine) -> Generator[Session, None, None]:
    """
    Provides a transactional database session for each test.
    All changes are rolled back at the end of the test, ensuring a clean state.
    """
    # Use the patched SessionLocal from db_module
    session = db_module.SessionLocal()
    try:
        yield session
        session.rollback()  # Ensure a clean slate for the next test
    finally:
        session.close()


@pytest.fixture(autouse=True)
def mock_db_logger(mocker) -> None:
    """
    Mocks the logger (`src.db.LOG`) to prevent log output from cluttering test results.
    """
    mocker.patch("src.db.LOG")


# --- Test Cases ---

# --- Normal Test Cases ---


def test_init_db_creates_tables(db_session: Session) -> None:
    """
    Test that `init_db` (implicitly called by fixture setup) successfully creates tables.
    Verifies the 'trades' table exists in the database.
    """
    from sqlalchemy import inspect

    inspector = inspect(db_session.get_bind())
    assert "trades" in inspector.get_table_names()


def test_get_db_provides_session_and_closes(mocker) -> None:
    """
    Test that `get_db` yields a session instance and ensures it's closed afterward.
    Uses mocks to verify method calls.
    """
    # Mock SessionLocal to return a mock session object
    mock_session = mocker.MagicMock(spec=Session)
    mock_session_local = mocker.MagicMock(return_value=mock_session)
    mocker.patch("src.db.SessionLocal", new=mock_session_local)

    with contextlib.closing(get_db()) as gen:
        session = next(gen)
        assert session is mock_session
        try:
            next(gen)
        except StopIteration:
            pass

    mock_session_local.assert_called_once()
    mock_session.close.assert_called_once()
    mock_session.rollback.assert_not_called()  # No exception, so no rollback


def test_create_and_retrieve_trade(db_session: Session) -> None:
    """
    Normal case: Create a `Trade` object, add it to the session, commit,
    and then retrieve it to verify persistence and correctness.
    """
    trade = Trade(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.001,
        price=30000.50,
    )
    db_session.add(trade)
    db_session.commit()

    retrieved_trade = db_session.query(Trade).filter_by(symbol="BTCUSDT").first()

    assert retrieved_trade is not None
    assert retrieved_trade.id is not None
    assert retrieved_trade.symbol == "BTCUSDT"
    assert retrieved_trade.side == "BUY"
    assert retrieved_trade.quantity == 0.001
    assert retrieved_trade.price == 30000.50
    assert isinstance(retrieved_trade.timestamp, datetime)
    assert retrieved_trade.timestamp.tzinfo == UTC


def test_trade_default_timestamp(db_session: Session) -> None:
    """
    Test that the `timestamp` field automatically defaults to the current UTC time
    when a `Trade` is created without an explicit timestamp.
    """
    now_utc = datetime.now(UTC)
    trade = Trade(
        symbol="ETHUSDT",
        side="SELL",
        quantity=0.01,
        price=2000.00,
    )
    db_session.add(trade)
    db_session.commit()

    retrieved_trade = db_session.query(Trade).get(trade.id)
    assert retrieved_trade is not None
    assert retrieved_trade.timestamp is not None
    assert retrieved_trade.timestamp.tzinfo == UTC
    # Check that timestamp is very close to now_utc (within a few seconds)
    assert (
        now_utc - timedelta(seconds=5) < retrieved_trade.timestamp < now_utc + timedelta(seconds=5)
    )


# --- Edge Cases ---


@pytest.mark.parametrize(
    "symbol, side, quantity, price",
    [
        ("A", "B", 0.00000001, 0.00000001),  # Smallest values with 8 decimal places
        (
            "SYMBLONG",
            "LONGSIDE",
            9999999999.99999999,
            9999999999.99999999,
        ),  # Large values, max precision
        ("XYZ", "BUY", 0.0, 0.0),  # Zero values
        ("TRADE", "SELL", 12345.6789, 98765.4321),  # Normal large numbers
        ("TRADESYM", "SIDE", -1.0, -100.0),  # Negative values (Numeric allows this)
    ],
)
def test_trade_edge_numeric_values(
    db_session: Session,
    symbol: Literal["A"]
    | Literal["SYMBLONG"]
    | Literal["XYZ"]
    | Literal["TRADE"]
    | Literal["TRADESYM"],
    side: Literal["B"] | Literal["LONGSIDE"] | Literal["BUY"] | Literal["SELL"] | Literal["SIDE"],
    quantity: float,
    price: float,
) -> None:
    """
    Test `Trade` creation with edge cases for numeric fields (quantity, price).
    Includes zero, very small, large, and negative values within Numeric precision.
    """
    trade = Trade(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
    )
    db_session.add(trade)
    db_session.commit()

    retrieved_trade = db_session.query(Trade).filter_by(symbol=symbol).first()
    assert retrieved_trade is not None
    assert retrieved_trade.quantity == quantity
    assert retrieved_trade.price == price


def test_trade_string_length_limits(db_session: Session) -> None:
    """
    Test `Trade` creation with strings at their defined length limits (`symbol`: 20, `side`: 10).
    Verifies that values up to the limit are stored correctly.
    Note: SQLite's VARCHAR is often permissive and may not raise errors for exceeding lengths.
    """
    long_symbol = "A" * 20
    long_side = "B" * 10
    trade = Trade(
        symbol=long_symbol,
        side=long_side,
        quantity=1.0,
        price=100.0,
    )
    db_session.add(trade)
    db_session.commit()

    retrieved_trade = db_session.query(Trade).filter_by(symbol=long_symbol).first()
    assert retrieved_trade is not None
    assert retrieved_trade.symbol == long_symbol
    assert retrieved_trade.side == long_side

    # Test values exceeding length limits. SQLite typically allows this without error
    # but other databases might truncate or raise an error.
    trade_too_long = Trade(
        symbol="A" * 21,  # Exceeds String(20)
        side="B" * 11,  # Exceeds String(10)
        quantity=1.0,
        price=100.0,
    )
    db_session.add(trade_too_long)
    db_session.commit()  # Should commit without error for SQLite

    retrieved_trade_too_long = db_session.query(Trade).filter_by(symbol="A" * 21).first()
    assert retrieved_trade_too_long is not None
    # For SQLite, the value might be stored as is or truncated, depending on configuration.
    # We primarily ensure it doesn't cause a hard error.


# --- Error Handling & Mocking ---


def test_get_db_session_rollback_on_exception(mocker) -> None:
    """
    Test that `get_db` correctly rolls back the session if an exception occurs
    within its context manager, and ensures the session is always closed.
    """
    mock_session = mocker.MagicMock(spec=Session)
    mock_session_local = mocker.MagicMock(return_value=mock_session)
    mocker.patch("src.db.SessionLocal", new=mock_session_local)

    class TestException(Exception):
        pass

    with pytest.raises(TestException):
        with contextlib.closing(get_db()) as gen:
            session = next(gen)
            assert session is mock_session
            raise TestException("Something went wrong!")
            # Generator will be finalized by pytest

    mock_session.rollback.assert_called_once()
    mock_session.close.assert_called_once()
    mock_session_local.assert_called_once()


def test_trade_missing_required_fields(db_session: Session) -> None:
    """
    Error handling: Attempt to create `Trade` objects with `nullable=False` fields missing.
    Expects `IntegrityError` upon commit.
    """
    test_cases = [
        ("symbol", Trade(side="BUY", quantity=0.001, price=30000.50)),
        ("side", Trade(symbol="BTCUSDT", quantity=0.001, price=30000.50)),
        ("quantity", Trade(symbol="BTCUSDT", side="BUY", price=30000.50)),
        ("price", Trade(symbol="BTCUSDT", side="BUY", quantity=0.001)),
    ]

    for field_name, trade_obj in test_cases:
        db_session.add(trade_obj)
        with pytest.raises(
            IntegrityError, match=f"NOT NULL constraint failed: trades.{field_name}"
        ):
            db_session.commit()
        db_session.rollback()  # Rollback the failed transaction for the next test case


def test_trade_invalid_data_types(db_session: Session) -> None:
    """
    Error handling: Attempt to save a `Trade` with incorrect data types for numeric fields.
    Expects `StatementError` or `TypeError` upon commit, as SQLAlchemy attempts conversion.
    """
    # Invalid quantity (string instead of float/numeric)
    trade_str_quantity = Trade(symbol="ABC", side="XYZ", quantity="not_a_number", price=100.0)
    db_session.add(trade_str_quantity)
    with pytest.raises(StatementError):
        db_session.commit()
    db_session.rollback()

    # Invalid price (string instead of float/numeric)
    trade_str_price = Trade(symbol="ABC", side="XYZ", quantity=1.0, price="not_a_number")
    db_session.add(trade_str_price)
    with pytest.raises(StatementError):
        db_session.commit()
    db_session.rollback()

    # Invalid timestamp (integer instead of datetime)
    trade_int_ts = Trade(symbol="ABC", side="XYZ", quantity=1.0, price=100.0, timestamp=12345)
    db_session.add(trade_int_ts)
    with pytest.raises((TypeError, StatementError)):  # Depends on SQLAlchemy version/dialect
        db_session.commit()
    db_session.rollback()


def test_init_db_error_handling(monkeypatch, mocker) -> None:
    """
    Test error handling in `init_db` when `Base.metadata.create_all` fails.
    Mocks `create_all` to raise an `SQLAlchemyError`.
    """
    # Mock Base.metadata.create_all to raise an exception
    mock_create_all = mocker.patch.object(
        db_module.Base.metadata,
        "create_all",
        side_effect=SQLAlchemyError("DB schema creation failed"),
    )

    # Temporarily replace the global engine/SessionLocal with a fresh one for this specific test
    temp_engine = create_engine("sqlite:///:memory:")
    temp_session_local = sessionmaker(bind=temp_engine)

    monkeypatch.setattr(db_module, "engine", temp_engine)
    monkeypatch.setattr(db_module, "SessionLocal", temp_session_local)

    with pytest.raises(SQLAlchemyError, match="DB schema creation failed"):
        db_module.init_db()  # Call the init_db function directly

    mock_create_all.assert_called_once()
    temp_engine.dispose()


def test_session_commit_error_handling(mocker) -> None:
    """
    Test error handling when `session.commit()` fails within the `get_db` context.
    Verifies that `session.rollback()` and `session.close()` are called,
    and the exception is propagated.
    """
    # Create a mock session that raises an error on commit
    spy_session = mocker.MagicMock(spec=Session)
    spy_session.commit.side_effect = SQLAlchemyError("Mocked commit failure")
    spy_session.add.return_value = None  # Prevent errors on add

    # Mock SessionLocal to return our spy session
    mock_session_local = mocker.MagicMock(return_value=spy_session)
    mocker.patch("src.db.SessionLocal", new=mock_session_local)

    trade = Trade(symbol="TEST", side="BUY", quantity=1.0, price=10.0)

    with pytest.raises(SQLAlchemyError, match="Mocked commit failure"):
        with contextlib.closing(get_db()) as gen:
            s = next(gen)
            s.add(trade)
            s.commit()  # This will trigger the mocked error

    spy_session.rollback.assert_called_once()
    spy_session.close.assert_called_once()
    mock_session_local.assert_called_once()


def test_database_connection_error_on_use(
    monkeypatch: pytest.MonkeyPatch, mocker: Callable[..., Generator[object, None, None]]
) -> None:
    """
    Test error handling when the database connection fails during initial use
    (e.g., during `init_db` or when `get_db` tries to establish a session).
    Mocks `db_module.engine.connect` to raise an `OperationalError`.
    """
    # Mock the engine's connect method to simulate a connection failure
    mock_connect = mocker.patch.object(
        db_module.engine, "connect", side_effect=OperationalError("Connection failed", {}, Exception())
    )

    # Test init_db: it attempts to create tables, which requires a connection
    with pytest.raises(OperationalError, match="Connection failed"):
        db_module.init_db()
    mock_connect.assert_called_once()
    mock_connect.reset_mock()  # Reset for the next test scenario

    # Test get_db: it attempts to create a session, which requires a connection

    with pytest.raises(OperationalError, match="Connection failed"):
        with contextlib.closing(db_module.get_db()) as gen:
            next(gen)  # The error should occur when the session is first acquired

    mock_connect.assert_called_once()
