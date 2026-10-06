"""keyword index over search_text instead of the timestamped transcript

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _replace_tsv(source_column: str) -> None:
    op.drop_index("ix_conversations_tsv", table_name="conversations", postgresql_using="gin")
    op.drop_column("conversations", "tsv")
    op.add_column(
        "conversations",
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed(f"to_tsvector('english', {source_column})", persisted=True),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_conversations_tsv", "conversations", ["tsv"], unique=False, postgresql_using="gin"
    )


def upgrade() -> None:
    # Conversations are derived data; rebuild them with
    # `python -m threadlight.processing.worker --reindex --drain`.
    op.execute("DELETE FROM conversations")
    op.add_column("conversations", sa.Column("search_text", sa.Text(), nullable=False))
    _replace_tsv("search_text")


def downgrade() -> None:
    _replace_tsv("text")
    op.drop_column("conversations", "search_text")
