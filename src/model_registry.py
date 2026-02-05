# src/model_registry.py
"""
Model registry helper for storing trained model metadata.

Provides:
  - ModelRegistry: PostgreSQL-backed registry for trained ML models
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any

import psycopg2
from psycopg2.extensions import connection as _Connection
from psycopg2.extras import RealDictCursor

from src.config import DATABASE_URL

LOG = logging.getLogger(__name__)


class ModelRegistry:
    """
    Simple model registry backed by PostgreSQL.

    This class is intentionally minimal — it wraps a single `model_registry`
    table and provides basic CRUD operations.
    """

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn: str = dsn or DATABASE_URL

        try:
            self.conn: _Connection = psycopg2.connect(self._dsn)
            self.conn.autocommit = True
            self._ensure_table()
        except Exception as exc:  # pragma: no cover
            LOG.exception("Failed to connect to model registry DB")
            raise RuntimeError("Could not connect to model registry database") from exc

    # ------------------------------------------------------------------
    # Context manager support (PRODUCTION REQUIRED)
    # ------------------------------------------------------------------
    def __enter__(self) -> ModelRegistry:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the database connection."""
        try:
            if hasattr(self, "conn"):
                self.conn.close()
        except Exception:
            LOG.exception("Error closing model registry connection")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_table(self) -> None:
        create_sql = """
        CREATE TABLE IF NOT EXISTS model_registry (
            id SERIAL PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_path TEXT NOT NULL,
            accuracy DOUBLE PRECISION,
            params JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        """
        with self.conn.cursor() as cur:
            cur.execute(create_sql)

    def _row_to_dict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(row)
        params = out.get("params")

        if isinstance(params, (str, bytes)):
            try:
                out["params"] = json.loads(params)
            except Exception:
                LOG.debug("Failed to parse params JSON; leaving raw")

        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_model(
        self,
        model_name: str,
        model_path: str,
        accuracy: float,
        params: Mapping[str, Any],
    ) -> None:
        """
        Register a new model in the registry.

        Args:
            model_name: Name of the model.
            model_path: Path to the saved model file.
            accuracy: Accuracy of the model.
            params: Model parameters as a mapping.
        """
        insert_sql = "INSERT INTO model_registry (model_name, model_path, accuracy, params) VALUES (%s, %s, %s, %s)"
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    insert_sql,
                    (model_name, model_path, accuracy, json.dumps(dict(params))),
                )
            LOG.info(
                "Registered model '%s' at %s (accuracy=%.4f)",
                model_name,
                model_path,
                accuracy,
            )
        except Exception:
            LOG.exception("Failed to register model '%s'", model_name)
            raise

    def get_model(self, model_name: str) -> dict[str, Any] | None:
        """
        Retrieve the most recent model entry by name.
        Args:
            model_name: Name of the model to retrieve.
        Returns:
            A dictionary of model metadata, or None if not found.
        """
        query = """
        SELECT id, model_name, model_path, accuracy, params, created_at
        FROM model_registry
        WHERE model_name = %s
        ORDER BY created_at DESC
        LIMIT 1;
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (model_name,))
                row: Mapping[str, Any] | None = cur.fetchone()
            return self._row_to_dict(row) if row else None
        except Exception:
            LOG.exception("Failed to get model '%s'", model_name)
            raise

    def list_models(self) -> list[dict[str, Any]]:
        """
        List the most recent entry for each model name.
        Returns:
            A list of dictionaries with model metadata.
        """
        query = """
        SELECT DISTINCT ON (model_name)
            id, model_name, model_path, accuracy, params, created_at
        FROM model_registry
        ORDER BY model_name, created_at DESC;
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                rows: Sequence[Mapping[str, Any]] = cur.fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception:
            LOG.exception("Failed to list models")
            raise

    def delete_model(self, model_name: str) -> bool:
        """
        Delete all entries for a given model name.

        Args:
            model_name: Name of the model to delete.
        Returns:
            True if any rows were deleted, False otherwise.
        """
        delete_sql = "DELETE FROM model_registry WHERE model_name = %s;"
        try:
            with self.conn.cursor() as cur:
                cur.execute(delete_sql, (model_name,))
                deleted = cur.rowcount
            LOG.info("Deleted %d rows for model '%s'", deleted, model_name)
            return deleted > 0
        except Exception:
            LOG.exception("Failed to delete model '%s'", model_name)
            raise


# ------------------------------------------------------------------
# Factory (safe for tests / DI)
# ------------------------------------------------------------------
def create_registry(dsn: str | None = None) -> ModelRegistry:
    return ModelRegistry(dsn)
