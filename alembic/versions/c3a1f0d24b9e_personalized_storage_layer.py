"""personalized storage layer for gesture recognition

Добавляет таблицы прикладного слоя пользовательских данных:
``users``, ``gesture_samples``, ``recognition_models``, ``app_sessions``,
``recognition_logs`` — а также расширяет таблицу ``gestures`` полями
``description``, ``is_active``, ``user_id``, ``updated_at``.

Revision ID: c3a1f0d24b9e
Revises: b2c4e6d8a0f1
Create Date: 2026-04-19
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "c3a1f0d24b9e"
down_revision: Union[str, Sequence[str], None] = "b2c4e6d8a0f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
    )

    with op.batch_alter_table("gestures") as batch:
        batch.add_column(sa.Column("description", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column(
                "is_active",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            )
        )
        batch.add_column(sa.Column("user_id", sa.Integer(), nullable=True))
        batch.add_column(
            sa.Column(
                "updated_at",
                sa.DateTime(),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=False,
            )
        )
        batch.create_foreign_key(
            "fk_gestures_user_id",
            "users",
            ["user_id"],
            ["id"],
        )

    op.create_table(
        "gesture_samples",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("gesture_id", sa.Integer(), nullable=False),
        sa.Column("sample_index", sa.Integer(), nullable=False),
        sa.Column("features_path", sa.Text(), nullable=True),
        sa.Column("features_json", sa.Text(), nullable=True),
        sa.Column("frames", sa.Integer(), nullable=True),
        sa.Column("hand_count", sa.Integer(), nullable=True),
        sa.Column(
            "source",
            sa.String(length=32),
            nullable=False,
            server_default="camera",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["gesture_id"], ["gestures.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "gesture_id", "sample_index", name="uq_gesture_sample_index"
        ),
    )
    op.create_index(
        "ix_gesture_samples_gesture_id",
        "gesture_samples",
        ["gesture_id"],
    )

    op.create_table(
        "recognition_models",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column(
            "algorithm",
            sa.String(length=32),
            nullable=False,
            server_default="knn",
        ),
        sa.Column("model_path", sa.Text(), nullable=True),
        sa.Column("classes_path", sa.Text(), nullable=True),
        sa.Column("feature_dim", sa.Integer(), nullable=True),
        sa.Column("n_classes", sa.Integer(), nullable=True),
        sa.Column("n_samples", sa.Integer(), nullable=True),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("hyperparams_json", sa.Text(), nullable=True),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "app_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("app_version", sa.String(length=32), nullable=True),
        sa.Column("platform", sa.String(length=32), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "recognition_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("model_id", sa.Integer(), nullable=True),
        sa.Column("gesture_id", sa.Integer(), nullable=True),
        sa.Column("session_id", sa.Integer(), nullable=True),
        sa.Column(
            "executed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "detected_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["gesture_id"], ["gestures.id"]),
        sa.ForeignKeyConstraint(["model_id"], ["recognition_models.id"]),
        sa.ForeignKeyConstraint(["session_id"], ["app_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_recognition_logs_detected_at",
        "recognition_logs",
        ["detected_at"],
    )
    op.create_index(
        "ix_recognition_logs_label",
        "recognition_logs",
        ["label"],
    )


def downgrade() -> None:
    op.drop_index("ix_recognition_logs_label", table_name="recognition_logs")
    op.drop_index(
        "ix_recognition_logs_detected_at", table_name="recognition_logs"
    )
    op.drop_table("recognition_logs")
    op.drop_table("app_sessions")
    op.drop_table("recognition_models")
    op.drop_index(
        "ix_gesture_samples_gesture_id", table_name="gesture_samples"
    )
    op.drop_table("gesture_samples")

    with op.batch_alter_table("gestures") as batch:
        batch.drop_constraint("fk_gestures_user_id", type_="foreignkey")
        batch.drop_column("updated_at")
        batch.drop_column("user_id")
        batch.drop_column("is_active")
        batch.drop_column("description")

    op.drop_table("users")
