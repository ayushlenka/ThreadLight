"""Idempotent writes for ingestion. Backfill and live events can safely overlap."""

from collections.abc import Iterable
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from threadlight.db.models import Channel, ChannelOverwrite, Guild, Member, Message, Role


async def upsert_guild(
    session: AsyncSession, guild_id: int, name: str, owner_id: int | None = None
) -> None:
    stmt = insert(Guild).values(id=guild_id, name=name, owner_id=owner_id)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Guild.id],
            set_={
                "name": stmt.excluded.name,
                "owner_id": func.coalesce(stmt.excluded.owner_id, Guild.owner_id),
            },
        )
    )


# Permission state


async def upsert_roles(session: AsyncSession, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    stmt = insert(Role).values(rows)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Role.id],
            set_={"name": stmt.excluded.name, "permissions": stmt.excluded.permissions},
        )
    )


async def replace_roles(session: AsyncSession, guild_id: int, rows: list[dict[str, Any]]) -> None:
    """Make the stored roles exactly `rows` (drops roles deleted while offline)."""
    await session.execute(
        delete(Role).where(Role.guild_id == guild_id, Role.id.not_in([r["id"] for r in rows]))
    )
    await upsert_roles(session, rows)


async def delete_role(session: AsyncSession, role_id: int) -> None:
    await session.execute(delete(Role).where(Role.id == role_id))


async def upsert_members(session: AsyncSession, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    stmt = insert(Member).values(rows)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Member.guild_id, Member.user_id],
            set_={
                "display_name": stmt.excluded.display_name,
                "role_ids": stmt.excluded.role_ids,
                "left_at": None,
            },
        )
    )


async def replace_members(session: AsyncSession, guild_id: int, rows: list[dict[str, Any]]) -> None:
    """Upsert current members and mark everyone else as having left."""
    await upsert_members(session, rows)
    await session.execute(
        update(Member)
        .where(
            Member.guild_id == guild_id,
            Member.user_id.not_in([r["user_id"] for r in rows]),
            Member.left_at.is_(None),
        )
        .values(left_at=func.now())
    )


async def mark_member_left(session: AsyncSession, guild_id: int, user_id: int) -> None:
    await session.execute(
        update(Member)
        .where(Member.guild_id == guild_id, Member.user_id == user_id)
        .values(left_at=func.now())
    )


async def replace_overwrites(
    session: AsyncSession, channel_id: int, rows: list[dict[str, Any]]
) -> None:
    await session.execute(delete(ChannelOverwrite).where(ChannelOverwrite.channel_id == channel_id))
    if rows:
        await session.execute(insert(ChannelOverwrite).values(rows))


async def upsert_channels(session: AsyncSession, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    stmt = insert(Channel).values(rows)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Channel.id],
            set_={
                "name": stmt.excluded.name,
                "parent_id": stmt.excluded.parent_id,
                "type": stmt.excluded.type,
                "deleted_at": None,
            },
        )
    )


async def mark_channel_deleted(session: AsyncSession, channel_id: int) -> None:
    await session.execute(
        update(Channel).where(Channel.id == channel_id).values(deleted_at=func.now())
    )


async def upsert_messages(session: AsyncSession, rows: list[dict[str, Any]]) -> None:
    """Insert new messages and apply edits to existing ones.

    Deleted messages are never resurrected: a backfill page fetched just before a delete
    event must not restore the scrubbed content.
    """
    if not rows:
        return
    stmt = insert(Message).values(rows)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Message.id],
            set_={
                "content": stmt.excluded.content,
                "attachments": stmt.excluded.attachments,
                "author_name": stmt.excluded.author_name,
                "edited_at": stmt.excluded.edited_at,
            },
            where=Message.deleted_at.is_(None),
        )
    )


async def mark_messages_deleted(session: AsyncSession, message_ids: Iterable[int]) -> None:
    """Scrub content but keep the row so reply chains and conversation boundaries survive."""
    ids = list(message_ids)
    if not ids:
        return
    await session.execute(
        update(Message)
        .where(Message.id.in_(ids), Message.deleted_at.is_(None))
        .values(content="", attachments=[], deleted_at=func.now())
    )


async def get_backfill_cursor(session: AsyncSession, channel_id: int) -> int | None:
    return await session.scalar(select(Channel.backfill_cursor).where(Channel.id == channel_id))


async def save_backfill_page(
    session: AsyncSession,
    channel_id: int,
    rows: list[dict[str, Any]],
    cursor: int,
) -> None:
    """Write a page of history and advance the checkpoint in the same transaction."""
    await upsert_messages(session, rows)
    await session.execute(
        update(Channel).where(Channel.id == channel_id).values(backfill_cursor=cursor)
    )


async def mark_backfill_done(session: AsyncSession, channel_id: int) -> None:
    await session.execute(
        update(Channel).where(Channel.id == channel_id).values(backfill_done=True)
    )
