"""add commands.action_spec JSON for UI-defined actions

Revision ID: b2c4e6d8a0f1
Revises: 4f809d88a774
Create Date: 2026-04-13

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b2c4e6d8a0f1"
down_revision: Union[str, Sequence[str], None] = "4f809d88a774"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "commands",
        sa.Column("action_spec", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("commands", "action_spec")
