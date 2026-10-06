"""Turn stored messages into embedded conversations."""

import logging
from dataclasses import dataclass

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.ext.asyncio import AsyncSession

from threadlight.db.models import Channel, Conversation, ConversationMessage, Message
from threadlight.processing.embedder import Embedder
from threadlight.processing.segmenter import Names, SegMessage, segment

log = logging.getLogger(__name__)


@dataclass
class ResegmentResult:
    created: int
    kept: int
    removed: int


async def resegment_channel(session: AsyncSession, channel_id: int) -> ResegmentResult:
    """Rebuild a channel's conversations from its current messages.

    The whole channel is re-segmented (cheap: pure Python over a few thousand rows), then
    diffed against what's stored by content hash. Unchanged conversations keep their row,
    embedding, and any extracted decisions; changed ones are replaced. Edits and deletes
    therefore propagate: a deleted message changes its conversation's text, so the old
    conversation (and anything derived from it) is removed.
    """
    channel = await session.get(Channel, channel_id)
    if channel is None:
        return ResegmentResult(0, 0, 0)

    rows = await session.execute(
        select(Message)
        .where(Message.channel_id == channel_id, Message.deleted_at.is_(None))
        .order_by(Message.created_at, Message.id)
    )
    messages = [to_seg_message(m) for m in rows.scalars()]
    segments = segment(channel.name, messages, await load_names(session, channel.guild_id))

    existing = dict(
        (
            await session.execute(
                select(Conversation.content_hash, Conversation.id).where(
                    Conversation.channel_id == channel_id
                )
            )
        ).all()
    )
    wanted = {s.content_hash for s in segments}

    stale_ids = [cid for h, cid in existing.items() if h not in wanted]
    if stale_ids:
        await session.execute(delete(Conversation).where(Conversation.id.in_(stale_ids)))

    created = 0
    for seg in segments:
        if seg.content_hash in existing:
            continue
        conv = Conversation(
            channel_id=channel_id,
            started_at=seg.first.created_at,
            ended_at=seg.last.created_at,
            message_count=len(seg.messages),
            first_message_id=seg.first.id,
            content_hash=seg.content_hash,
            text=seg.text,
            search_text=seg.search_text,
        )
        session.add(conv)
        await session.flush()
        session.add_all(
            ConversationMessage(conversation_id=conv.id, message_id=m.id) for m in seg.messages
        )
        created += 1

    return ResegmentResult(created=created, kept=len(segments) - created, removed=len(stale_ids))


def to_seg_message(m: Message) -> SegMessage:
    return SegMessage(
        id=m.id,
        author_name=m.author_name,
        content=m.content,
        created_at=m.created_at,
        reply_to_id=m.reply_to_id,
        attachment_names=tuple(a["filename"] for a in m.attachments),
    )


async def load_names(session: AsyncSession, guild_id: int) -> Names:
    """Display names for resolving mentions: each author's most recent name, and channels.

    Users who were mentioned but never posted render as "@someone".
    """
    users = await session.execute(
        select(Message.author_id, Message.author_name)
        .join(Channel, Channel.id == Message.channel_id)
        .where(Channel.guild_id == guild_id)
        .ext(distinct_on(Message.author_id))
        .order_by(Message.author_id, Message.created_at.desc())
    )
    channels = await session.execute(
        select(Channel.id, Channel.name).where(Channel.guild_id == guild_id)
    )
    return Names(users=dict(users.all()), channels=dict(channels.all()))


async def embed_pending(
    session: AsyncSession,
    embedder: Embedder,
    channel_id: int | None = None,
    limit: int | None = None,
) -> int:
    """Embed up to `limit` conversations that don't have an embedding yet. Returns how many.

    Callers embedding a large backlog should call this repeatedly in separate transactions
    so progress is committed as it goes (see worker.embed_channel).
    """
    stmt = select(Conversation.id, Conversation.text).where(Conversation.embedding.is_(None))
    if channel_id is not None:
        stmt = stmt.where(Conversation.channel_id == channel_id)
    pending = (await session.execute(stmt.order_by(Conversation.id).limit(limit))).all()
    if not pending:
        return 0

    vectors = await embedder.embed_documents([text for _, text in pending])
    for (conv_id, _), vec in zip(pending, vectors, strict=True):
        await session.execute(
            update(Conversation).where(Conversation.id == conv_id).values(embedding=vec)
        )
    return len(pending)
