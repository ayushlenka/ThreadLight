"""Database schema.

Discord snowflake IDs are stored as BIGINT and used directly as primary keys.
Threads are stored as channels (with parent_id set), matching how Discord models them,
so every message belongs to exactly one channel and permission filtering is a single
`channel_id = ANY(:visible)` predicate.

Every derived record (conversation, decision) is scoped to exactly one channel. Answers
that span channels are composed at query time from records the asking user can see, so
nothing derived from a hidden channel is ever stored alongside visible content.
"""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from threadlight.db.base import Base

# voyage-3.5 default output dimension.
EMBEDDING_DIM = 1024


class Guild(Base):
    __tablename__ = "guilds"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(Text)
    owner_id: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Role(Base):
    """Guild role. The @everyone role's id equals the guild id."""

    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    guild_id: Mapped[int] = mapped_column(ForeignKey("guilds.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(Text)
    # Discord permission bitfield (fits in a signed 64-bit integer).
    permissions: Mapped[int] = mapped_column(BigInteger)


class Member(Base):
    __tablename__ = "members"

    guild_id: Mapped[int] = mapped_column(
        ForeignKey("guilds.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    display_name: Mapped[str] = mapped_column(Text)
    # Excludes @everyone, which every member implicitly has.
    role_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger), server_default="{}")
    # Set when a member leaves; former members can see nothing.
    left_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ChannelOverwrite(Base):
    """Per-channel permission overwrite for a role or a member. Threads have none."""

    __tablename__ = "channel_overwrites"

    channel_id: Mapped[int] = mapped_column(
        ForeignKey("channels.id", ondelete="CASCADE"), primary_key=True
    )
    target_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    target_type: Mapped[str] = mapped_column(Text)
    allow: Mapped[int] = mapped_column(BigInteger)
    deny: Mapped[int] = mapped_column(BigInteger)

    __table_args__ = (CheckConstraint("target_type IN ('role', 'member')", name="target_type"),)


class Channel(Base):
    __tablename__ = "channels"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    guild_id: Mapped[int] = mapped_column(ForeignKey("guilds.id", ondelete="CASCADE"), index=True)
    # Category for text channels, parent channel for threads.
    parent_id: Mapped[int | None] = mapped_column(BigInteger)
    # discord.ChannelType name: text, public_thread, private_thread, news, forum, ...
    type: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text)
    # Backfill checkpoint (newest message id ingested so far), so a crashed or restarted
    # backfill resumes where it stopped instead of starting over.
    backfill_cursor: Mapped[int | None] = mapped_column(BigInteger)
    backfill_done: Mapped[bool] = mapped_column(Boolean, server_default="false")
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"))
    author_id: Mapped[int] = mapped_column(BigInteger)
    author_name: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    # Not a foreign key: the referenced message may predate the backfill or be deleted.
    reply_to_id: Mapped[int | None] = mapped_column(BigInteger)
    attachments: Mapped[list[Any]] = mapped_column(JSONB, server_default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    edited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index(None, "channel_id", "created_at"),)


class Conversation(Base):
    """A segment of related messages in one channel: the unit we embed and retrieve."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    message_count: Mapped[int] = mapped_column(Integer)
    # Used for Discord jump links.
    first_message_id: Mapped[int] = mapped_column(BigInteger)
    # sha256 of `text`. Re-segmenting a channel keeps conversations whose hash is unchanged,
    # so only new or modified conversations are re-embedded (and re-extracted).
    content_hash: Mapped[str] = mapped_column(Text)
    # Rendered transcript ("[2024-02-13 14:02] alice: ...") used for embedding and prompts.
    text: Mapped[str] = mapped_column(Text)
    # Speaker + content only (no timestamps/channel header), for the keyword index.
    search_text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', search_text)", persisted=True)
    )
    # Set once decision extraction has run (finding no decisions is a valid outcome).
    extracted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index(None, "channel_id", "started_at"),
        Index(None, "tsv", postgresql_using="gin"),
        Index(
            None,
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True
    )
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True, index=True
    )


class Decision(Base):
    """A decision extracted from a conversation.

    `status` is what the conversation itself established ('decided' or 'proposed').
    Whether a decision has since been superseded is not stored: it's derived from
    `supersedes_id` links at read time, filtered to what the viewer can see, so it can't go
    stale when the superseding decision is deleted, and a hidden channel's decision never
    reveals itself by marking a visible one as replaced.
    """

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    channel_id: Mapped[int] = mapped_column(
        ForeignKey("channels.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    rationale: Mapped[list[Any]] = mapped_column(JSONB, server_default="[]")
    alternatives: Mapped[list[Any]] = mapped_column(JSONB, server_default="[]")
    status: Mapped[str] = mapped_column(Text)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # The message where it was settled; the decision's citation / jump link.
    decided_message_id: Mapped[int] = mapped_column(BigInteger)
    # The earlier decision this one replaces.
    supersedes_id: Mapped[int | None] = mapped_column(
        ForeignKey("decisions.id", ondelete="SET NULL"), index=True
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', title || ' ' || summary)", persisted=True),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("status IN ('proposed', 'decided')", name="status"),
        Index(None, "tsv", postgresql_using="gin"),
        Index(
            None,
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class DecisionSource(Base):
    """Messages that support a decision. Powers citations and delete propagation."""

    __tablename__ = "decision_sources"

    decision_id: Mapped[int] = mapped_column(
        ForeignKey("decisions.id", ondelete="CASCADE"), primary_key=True
    )
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True, index=True
    )


class Job(Base):
    """Postgres-backed work queue, claimed with SELECT ... FOR UPDATE SKIP LOCKED."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    kind: Mapped[str] = mapped_column(Text)
    # At most one *pending* job per (kind, dedupe_key): a burst of messages in a channel
    # collapses into a single re-segment job.
    dedupe_key: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    status: Mapped[str] = mapped_column(Text, server_default="pending")
    attempts: Mapped[int] = mapped_column(Integer, server_default="0")
    last_error: Mapped[str | None] = mapped_column(Text)
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint("status IN ('pending', 'running', 'done', 'failed')", name="status"),
        Index(None, "status", "run_after"),
        Index(
            "uq_jobs_pending_dedupe",
            "kind",
            "dedupe_key",
            unique=True,
            postgresql_where=text("status = 'pending'"),
        ),
    )


class ApiUsage(Base):
    """One paid API call (Claude or Voyage) with its tokens and list-price cost."""

    __tablename__ = "api_usage"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    provider: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    # answer | extract | supersede | embed_document | embed_query
    purpose: Mapped[str] = mapped_column(Text, index=True)
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer, server_default="0")
    cache_read_tokens: Mapped[int] = mapped_column(Integer, server_default="0")
    cache_write_tokens: Mapped[int] = mapped_column(Integer, server_default="0")
    # NULL when the model has no entry in usage.PRICES.
    cost_usd: Mapped[Any | None] = mapped_column(Numeric(12, 6))
    # guild_id, user_id, conversation_id, ... (see usage.usage_tags)
    tags: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
