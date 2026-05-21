"""Create initial trading bot tables.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-05-18 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("prediction", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("model_name", sa.String(length=120), nullable=True),
        sa.Column("model_version", sa.String(length=80), nullable=True),
        sa.Column("model_path", sa.String(length=500), nullable=True),
        sa.Column("symbol", sa.String(length=20), nullable=True),
        sa.Column("source", sa.String(length=80), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_predictions_id", "predictions", ["id"], unique=False)
    op.create_index("ix_predictions_symbol", "predictions", ["symbol"], unique=False)

    op.create_table(
        "trades",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("side", sa.String(length=10), nullable=False),
        sa.Column("quantity", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("price", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=True),
        sa.Column("order_id", sa.String(length=80), nullable=True),
        sa.Column("fill_price", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("commission", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("commission_asset", sa.String(length=10), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("order_id"),
    )
    op.create_index("ix_trades_id", "trades", ["id"], unique=False)
    op.create_index("ix_trades_symbol", "trades", ["symbol"], unique=False)

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(length=80), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=40), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_superuser", sa.Boolean(), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_id", "users", ["id"], unique=False)
    op.create_index("ix_users_username", "users", ["username"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_users_username", table_name="users")
    op.drop_index("ix_users_id", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")

    op.drop_index("ix_trades_symbol", table_name="trades")
    op.drop_index("ix_trades_id", table_name="trades")
    op.drop_table("trades")

    op.drop_index("ix_predictions_symbol", table_name="predictions")
    op.drop_index("ix_predictions_id", table_name="predictions")
    op.drop_table("predictions")
