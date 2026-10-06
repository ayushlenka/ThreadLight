"""History sync: walk each channel oldest-to-newest from its checkpoint.

The same walk serves as the initial backfill and as catch-up after downtime: on every
startup each channel resumes from its stored cursor, so messages posted while the bot was
offline are picked up. Writes are idempotent, so overlap with live events is harmless.

Known gap: edits and deletes that happen while the bot is offline are not caught by the
walk (it only sees messages newer than the cursor). A periodic reconciliation pass can
close that later.
"""

import logging

import discord
from sqlalchemy import func, select

from threadlight.db.models import Channel, Message
from threadlight.db.session import SessionLocal
from threadlight.ingest import store
from threadlight.ingest.convert import (
    IngestableChannel,
    channel_row,
    member_row,
    message_row,
    overwrite_rows,
    role_row,
)
from threadlight.processing import jobs

log = logging.getLogger(__name__)

PAGE_SIZE = 100  # Discord returns at most 100 messages per history request.


async def discover_channels(guild: discord.Guild) -> list[IngestableChannel]:
    """Text channels and their threads (active and archived) the bot can read."""
    me = guild.me
    readable_parents: list[discord.TextChannel | discord.ForumChannel] = []
    for channel in guild.channels:
        if not isinstance(channel, discord.TextChannel | discord.ForumChannel):
            continue
        if channel.permissions_for(me).read_message_history:
            readable_parents.append(channel)
        else:
            log.warning("skipping #%s: bot lacks Read Message History", channel.name)

    found: dict[int, IngestableChannel] = {}
    parent_ids = {p.id for p in readable_parents}
    for parent in readable_parents:
        if isinstance(parent, discord.TextChannel):
            found[parent.id] = parent
        try:
            async for thread in parent.archived_threads(limit=None):
                found[thread.id] = thread
        except discord.Forbidden:
            log.warning("cannot list archived threads in #%s", parent.name)

    for thread in await guild.active_threads():
        if thread.parent_id in parent_ids:
            found[thread.id] = thread

    return list(found.values())


async def sync_channel(channel: IngestableChannel) -> int:
    """Ingest everything newer than the channel's checkpoint. Returns messages seen."""
    async with SessionLocal() as session:
        cursor = await store.get_backfill_cursor(session, channel.id)

    after = discord.Object(id=cursor) if cursor else None
    rows: list[dict] = []
    last_id: int | None = None
    seen = 0

    async def flush() -> None:
        async with SessionLocal.begin() as session:
            await store.save_backfill_page(session, channel.id, rows, last_id)
        rows.clear()

    try:
        async for message in channel.history(limit=None, after=after, oldest_first=True):
            seen += 1
            last_id = message.id
            if (row := message_row(message)) is not None:
                rows.append(row)
            if seen % PAGE_SIZE == 0:
                await flush()
    except discord.Forbidden:
        log.warning("no access to history in #%s", channel.name)
        return seen

    if seen % PAGE_SIZE:
        await flush()
    async with SessionLocal.begin() as session:
        await store.mark_backfill_done(session, channel.id)
        if seen:
            await jobs.enqueue_segment(session, channel.id)
    return seen


async def sync_guild(guild: discord.Guild) -> None:
    channels = await discover_channels(guild)
    # Permission state first, in one transaction with the channel list, so access checks
    # are correct before any new history is searchable.
    async with SessionLocal.begin() as session:
        await store.upsert_guild(session, guild.id, guild.name, guild.owner_id)
        await store.replace_roles(session, guild.id, [role_row(r) for r in guild.roles])
        await store.replace_members(session, guild.id, [member_row(m) for m in guild.members])
        await store.upsert_channels(session, [channel_row(c) for c in channels])
        for channel in channels:
            if not isinstance(channel, discord.Thread):
                await store.replace_overwrites(session, channel.id, overwrite_rows(channel))
    log.info("permission state: %d roles, %d members", len(guild.roles), len(guild.members))

    log.info("syncing %d channels/threads in %s", len(channels), guild.name)
    for channel in channels:
        seen = await sync_channel(channel)
        log.info("#%s: %d new messages fetched", channel.name, seen)

    await log_summary(guild.id)


async def log_summary(guild_id: int) -> None:
    async with SessionLocal() as session:
        result = await session.execute(
            select(Channel.name, func.count(Message.id))
            .join(Message, Message.channel_id == Channel.id, isouter=True)
            .where(Channel.guild_id == guild_id, Channel.deleted_at.is_(None))
            .group_by(Channel.name)
            .order_by(func.count(Message.id).desc())
        )
        counts = result.all()
    total = sum(n for _, n in counts)
    log.info("sync complete: %d messages stored", total)
    for name, n in counts:
        log.info("  #%-30s %d", name, n)
