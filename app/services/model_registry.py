"""
Model registry helper for storing trained model metadata.

Provides:
    - ModelRegistry: PostgreSQL-backed registry for trained ML models
    - create_registry: factory helper for dependency injection/tests
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

from app.core.config import DATABASE_URL

LOG = logging.getLogger(__name__)


class ModelRegistry:
    """
    Simple PostgreSQL-backed model registry.

    Stores trained model metadata such as:
    - model name
    - model file path
    - accuracy
    - params
    - creation timestamp
    """

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or DATABASE_URL

        try:
            self.conn: _Connection = psycopg2.connect(self._dsn)
            self.conn.autocommit = True
            self._ensure_table()
        except Exception as exc:
            LOG.exception("Failed to connect to model registry database.")
            raise RuntimeError("Could not connect to model registry database.") from exc

    # ------------------------------------------------------------------
    # Context manager support
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
            if hasattr(self, "conn") and not self.conn.closed:
                self.conn.close()
        except Exception:
            LOG.exception("Error closing model registry connection.")

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

        CREATE INDEX IF NOT EXISTS idx_model_registry_model_name_created_at
        ON model_registry (model_name, created_at DESC);
        """

        with self.conn.cursor() as cur:
            cur.execute(create_sql)

    def _row_to_dict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(row)
        params = output.get("params")

        if isinstance(params, (str, bytes)):
            try:
                output["params"] = json.loads(params)
            except Exception:
                LOG.debug("Failed to parse model params JSON; leaving raw.")

        return output

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_model(
        self,
        model_name: str,
        model_path: str,
        accuracy: float | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Register a trained model in the registry.
        """
        if not model_name:
            raise ValueError("model_name is required.")

        if not model_path:
            raise ValueError("model_path is required.")

        insert_sql = """
        INSERT INTO model_registry (model_name, model_path, accuracy, params)
        VALUES (%s, %s, %s, %s);
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    insert_sql,
                    (
                        model_name,
                        model_path,
                        accuracy,
                        json.dumps(dict(params or {})),
                    ),
                )

            LOG.info(
                "Registered model '%s' at %s. accuracy=%s",
                model_name,
                model_path,
                accuracy,
            )

        except Exception:
            LOG.exception("Failed to register model '%s'.", model_name)
            raise

    def get_model(self, model_name: str) -> dict[str, Any] | None:
        """
        Retrieve the most recent model entry by model name.
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
            LOG.exception("Failed to get model '%s'.", model_name)
            raise

    def list_models(self) -> list[dict[str, Any]]:
        """
        List the most recent entry for each model name.
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

            return [self._row_to_dict(row) for row in rows]

        except Exception:
            LOG.exception("Failed to list models.")
            raise

    def delete_model(self, model_name: str) -> bool:
        """
        Delete all entries for a given model name.
        """
        delete_sql = "DELETE FROM model_registry WHERE model_name = %s;"

        try:
            with self.conn.cursor() as cur:
                cur.execute(delete_sql, (model_name,))
                deleted = cur.rowcount

            LOG.info("Deleted %d rows for model '%s'.", deleted, model_name)
            return deleted > 0

        except Exception:
            LOG.exception("Failed to delete model '%s'.", model_name)
            raise


def create_registry(dsn: str | None = None) -> ModelRegistry:
    """Factory helper for tests and dependency injection."""
    return ModelRegistry(dsn)
