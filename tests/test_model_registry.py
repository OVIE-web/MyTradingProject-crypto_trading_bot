"""
Tests for src/model_registry.py module.
Tests model registry database operations.
"""

from unittest.mock import MagicMock, patch

import psycopg2
import pytest

from src.model_registry import ModelRegistry, create_registry


class TestModelRegistry:
    """Test cases for ModelRegistry class."""

    @patch("psycopg2.connect")
    def test_init_success(self, mock_connect: MagicMock) -> None:
        """Test successful initialization."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        registry = ModelRegistry("postgresql://test")

        mock_connect.assert_called_once_with("postgresql://test")
        assert registry.conn == mock_conn
        mock_conn.close.assert_not_called()  # Should not close on init

    @patch("psycopg2.connect")
    def test_init_failure(self, mock_connect: MagicMock) -> None:
        """Test initialization failure."""
        mock_connect.side_effect = psycopg2.OperationalError("Connection failed")

        with pytest.raises(RuntimeError, match="Could not connect to model registry database"):
            ModelRegistry("postgresql://test")

    @patch("psycopg2.connect")
    def test_init_default_dsn(
        self, mock_connect: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test initialization with default DSN from config."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://default")
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        registry = ModelRegistry()  # No DSN provided

        mock_connect.assert_called_once_with("postgresql://default")

    @patch("psycopg2.connect")
    def test_context_manager(self, mock_connect: MagicMock) -> None:
        """Test context manager functionality."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with ModelRegistry("postgresql://test") as registry:
            assert registry.conn == mock_conn

        # Connection should be closed when exiting context
        mock_conn.close.assert_called_once()

    @patch("psycopg2.connect")
    def test_close(self, mock_connect: MagicMock) -> None:
        """Test explicit close method."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        registry = ModelRegistry("postgresql://test")
        registry.close()

        mock_conn.close.assert_called_once()

    @patch("psycopg2.connect")
    def test_register_model(self, mock_connect: MagicMock) -> None:
        """Test model registration."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        registry = ModelRegistry("postgresql://test")

        # Test successful registration
        result = registry.register_model(
            model_name="test_model",
            model_path="/path/to/model.json",
            accuracy=0.85,
            params={"learning_rate": 0.1},
        )

        assert result is True
        # Verify SQL execution
        assert mock_cursor.execute.call_count == 1
        execute_args = mock_cursor.execute.call_args[0]
        assert "INSERT INTO model_registry" in execute_args[0]
        assert "test_model" in execute_args[1]
        assert "/path/to/model.json" in execute_args[1]
        assert 0.85 in execute_args[1]

    @patch("psycopg2.connect")
    def test_get_model_found(self, mock_connect: MagicMock) -> None:
        """Test retrieving an existing model."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Mock fetchone to return model data
        mock_cursor.fetchone.return_value = {
            "id": 1,
            "model_name": "test_model",
            "model_path": "/path/to/model.json",
            "accuracy": 0.85,
            "params": '{"learning_rate": 0.1}',
            "created_at": "2024-01-01T00:00:00Z",
        }

        registry = ModelRegistry("postgresql://test")
        result = registry.get_model("test_model")

        assert result is not None
        assert result["model_name"] == "test_model"
        assert result["accuracy"] == 0.85
        assert result["params"] == {"learning_rate": 0.1}

    @patch("psycopg2.connect")
    def test_get_model_not_found(self, mock_connect: MagicMock) -> None:
        """Test retrieving a non-existent model."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Mock fetchone to return None
        mock_cursor.fetchone.return_value = None

        registry = ModelRegistry("postgresql://test")
        result = registry.get_model("nonexistent_model")

        assert result is None

    @patch("psycopg2.connect")
    def test_list_models(self, mock_connect: MagicMock) -> None:
        """Test listing all models."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Mock fetchall to return multiple models
        mock_cursor.fetchall.return_value = [
            {
                "id": 1,
                "model_name": "model1",
                "model_path": "/path/model1.json",
                "accuracy": 0.85,
                "params": '{"lr": 0.1}',
                "created_at": "2024-01-01T00:00:00Z",
            },
            {
                "id": 2,
                "model_name": "model2",
                "model_path": "/path/model2.json",
                "accuracy": 0.90,
                "params": '{"lr": 0.05}',
                "created_at": "2024-01-02T00:00:00Z",
            },
        ]

        registry = ModelRegistry("postgresql://test")
        result = registry.list_models()

        assert len(result) == 2
        assert result[0]["model_name"] == "model1"
        assert result[1]["model_name"] == "model2"

    @patch("psycopg2.connect")
    def test_delete_model_success(self, mock_connect: MagicMock) -> None:
        """Test successful model deletion."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Mock rowcount to indicate deletion
        mock_cursor.rowcount = 1

        registry = ModelRegistry("postgresql://test")
        result = registry.delete_model("test_model")

        assert result is True
        # Verify DELETE SQL execution
        mock_cursor.execute.assert_called_once()
        execute_args = mock_cursor.execute.call_args[0]
        assert "DELETE FROM model_registry" in execute_args[0]
        assert "test_model" in execute_args[1]

    @patch("psycopg2.connect")
    def test_delete_model_not_found(self, mock_connect: MagicMock) -> None:
        """Test deleting a non-existent model."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Mock rowcount to indicate no deletion
        mock_cursor.rowcount = 0

        registry = ModelRegistry("postgresql://test")
        result = registry.delete_model("nonexistent_model")

        assert result is False


class TestCreateRegistry:
    """Test cases for create_registry function."""

    @patch("src.model_registry.ModelRegistry")
    def test_create_registry_with_dsn(self, mock_registry_class: MagicMock) -> None:
        """Test create_registry with explicit DSN."""
        mock_registry_instance = MagicMock()
        mock_registry_class.return_value = mock_registry_instance

        result = create_registry("postgresql://test")

        mock_registry_class.assert_called_once_with("postgresql://test")
        assert result == mock_registry_instance

    @patch("src.model_registry.ModelRegistry")
    def test_create_registry_default(self, mock_registry_class: MagicMock) -> None:
        """Test create_registry with default DSN."""
        mock_registry_instance = MagicMock()
        mock_registry_class.return_value = mock_registry_instance

        result = create_registry()

        mock_registry_class.assert_called_once_with(None)
        assert result == mock_registry_instance
