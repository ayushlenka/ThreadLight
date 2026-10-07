"""Discord bot: syncs history on startup, mirrors live changes into Postgres, and serves /ask.

Run with `python -m threadlight.ingest.bot`.

Raw events are used for edits and deletes because discord.py only caches recent messages;
the non-raw variants silently miss anything older than the cache.
"""

import asyncio
import logging
from datetime import timedelta

import discord
from discord import app_commands

from threadlight.answer.discord_ask import register_ask_command, register_decisions_command
from threadlight.config import get_settings
from threadlight.db.session import SessionLocal
from threadlight.ingest import store
from threadlight.ingest.backfill import sync_guild
from threadlight.ingest.convert import (
    channel_row,
    member_row,
    message_row,
    overwrite_rows,
    role_row,
)
from threadlight.processing import jobs

log = logging.getLogger(__name__)

# Debounce: re-segment a channel a minute after its first change rather than per message.
RESEGMENT_DELAY = timedelta(seconds=60)


class IngestBot(discord.Client):
    def __init__(self, guild_id: int) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True  # member roles feed per-user channel permissions
        super().__init__(intents=intents)
        self.guild_id = guild_id
        self._known_channels: set[int] = set()
        self._sync_task: asyncio.Task | None = None
        self.tree = app_commands.CommandTree(self)
        register_ask_command(self.tree, guild_id)
        register_decisions_command(self.tree, guild_id)

    async def setup_hook(self) -> None:
        # Guild-scoped commands update instantly (global ones can take up to an hour).
        await self.tree.sync(guild=discord.Object(id=self.guild_id))

    def _ours(self, guild_id: int | None) -> bool:
        return guild_id == self.guild_id

    async def on_ready(self) -> None:
        # on_ready fires again after reconnects; only sync once per process.
        if self._sync_task is not None:
            return
        guild = self.get_guild(self.guild_id)
        if guild is None:
            joined = ", ".join(f"{g.name} ({g.id})" for g in self.guilds) or "none"
            invite = discord.utils.oauth_url(
                self.user.id,
                permissions=discord.Permissions(view_channel=True, read_message_history=True),
                scopes=("bot", "applications.commands"),
            )
            log.error(
                "bot is not in guild %s; check DISCORD_GUILD_ID. Bot is in: %s. Invite: %s",
                self.guild_id,
                joined,
                invite,
            )
            await self.close()
            return
        log.info("connected as %s to %s", self.user, guild.name)
        self._sync_task = asyncio.create_task(sync_guild(guild))
        self._sync_task.add_done_callback(self._on_sync_done)

    def _on_sync_done(self, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            log.error("history sync failed", exc_info=task.exception())

    async def _write_channel(self, channel: discord.abc.GuildChannel | discord.Thread) -> None:
        # Live events can arrive before on_ready (while members are still being chunked),
        # i.e. before the history sync has created the guild row, so upsert it here too.
        # Overwrites are written in the same transaction as the channel: a channel row
        # without its overwrites would look readable by everyone.
        async with SessionLocal.begin() as session:
            await store.upsert_guild(
                session, channel.guild.id, channel.guild.name, channel.guild.owner_id
            )
            await store.upsert_channels(session, [channel_row(channel)])
            if not isinstance(channel, discord.Thread):
                await store.replace_overwrites(session, channel.id, overwrite_rows(channel))
        self._known_channels.add(channel.id)

    async def _ensure_channel(self, channel: discord.abc.GuildChannel | discord.Thread) -> None:
        if channel.id not in self._known_channels:
            await self._write_channel(channel)

    # Messages

    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None or not self._ours(message.guild.id):
            return
        # Skip voice-channel text chat and other channel kinds the history sync doesn't cover.
        if not isinstance(message.channel, discord.TextChannel | discord.Thread):
            return
        if (row := message_row(message)) is None:
            return
        await self._ensure_channel(message.channel)
        async with SessionLocal.begin() as session:
            await store.upsert_messages(session, [row])
            await jobs.enqueue_segment(session, row["channel_id"], RESEGMENT_DELAY)

    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent) -> None:
        if not self._ours(payload.guild_id):
            return
        if (row := message_row(payload.message)) is None:
            return
        await self._ensure_channel(payload.message.channel)
        async with SessionLocal.begin() as session:
            await store.upsert_messages(session, [row])
            await jobs.enqueue_segment(session, row["channel_id"], RESEGMENT_DELAY)

    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent) -> None:
        if not self._ours(payload.guild_id):
            return
        async with SessionLocal.begin() as session:
            await store.mark_messages_deleted(session, [payload.message_id])
            await jobs.enqueue_segment(session, payload.channel_id, RESEGMENT_DELAY)

    async def on_raw_bulk_message_delete(self, payload: discord.RawBulkMessageDeleteEvent) -> None:
        if not self._ours(payload.guild_id):
            return
        async with SessionLocal.begin() as session:
            await store.mark_messages_deleted(session, payload.message_ids)
            await jobs.enqueue_segment(session, payload.channel_id, RESEGMENT_DELAY)

    # Channels and threads

    async def _upsert_channel(self, channel: discord.abc.GuildChannel | discord.Thread) -> None:
        if not self._ours(channel.guild.id):
            return
        if not isinstance(channel, discord.TextChannel | discord.Thread):
            return
        await self._write_channel(channel)

    async def _delete_channel(self, guild_id: int, channel_id: int) -> None:
        if not self._ours(guild_id):
            return
        async with SessionLocal.begin() as session:
            await store.mark_channel_deleted(session, channel_id)
        self._known_channels.discard(channel_id)

    async def on_guild_channel_create(self, channel: discord.abc.GuildChannel) -> None:
        await self._upsert_channel(channel)

    async def on_guild_channel_update(
        self, before: discord.abc.GuildChannel, after: discord.abc.GuildChannel
    ) -> None:
        await self._upsert_channel(after)

    async def on_guild_channel_delete(self, channel: discord.abc.GuildChannel) -> None:
        await self._delete_channel(channel.guild.id, channel.id)

    async def on_thread_create(self, thread: discord.Thread) -> None:
        await self._upsert_channel(thread)

    async def on_thread_update(self, before: discord.Thread, after: discord.Thread) -> None:
        await self._upsert_channel(after)

    async def on_raw_thread_delete(self, payload: discord.RawThreadDeleteEvent) -> None:
        await self._delete_channel(payload.guild_id, payload.thread_id)

    # Permission state: roles, members, ownership

    async def on_guild_update(self, before: discord.Guild, after: discord.Guild) -> None:
        if not self._ours(after.id):
            return
        async with SessionLocal.begin() as session:
            await store.upsert_guild(session, after.id, after.name, after.owner_id)

    async def _write_role(self, role: discord.Role) -> None:
        if not self._ours(role.guild.id):
            return
        async with SessionLocal.begin() as session:
            await store.upsert_roles(session, [role_row(role)])

    async def on_guild_role_create(self, role: discord.Role) -> None:
        await self._write_role(role)

    async def on_guild_role_update(self, before: discord.Role, after: discord.Role) -> None:
        await self._write_role(after)

    async def on_guild_role_delete(self, role: discord.Role) -> None:
        if not self._ours(role.guild.id):
            return
        async with SessionLocal.begin() as session:
            await store.delete_role(session, role.id)

    async def _write_member(self, member: discord.Member) -> None:
        if not self._ours(member.guild.id):
            return
        async with SessionLocal.begin() as session:
            await store.upsert_members(session, [member_row(member)])

    async def on_member_join(self, member: discord.Member) -> None:
        await self._write_member(member)

    async def on_member_update(self, before: discord.Member, after: discord.Member) -> None:
        await self._write_member(after)

    async def on_raw_member_remove(self, payload: discord.RawMemberRemoveEvent) -> None:
        if not self._ours(payload.guild_id):
            return
        async with SessionLocal.begin() as session:
            await store.mark_member_left(session, payload.guild_id, payload.user.id)


def main() -> None:
    settings = get_settings()
    if not settings.discord_bot_token or settings.discord_guild_id is None:
        raise SystemExit(
            "Set DISCORD_BOT_TOKEN and DISCORD_GUILD_ID in .env "
            "(run `threadlight check`; setup: docs/self-hosting.md)"
        )
    discord.utils.setup_logging(level=logging.INFO)
    IngestBot(settings.discord_guild_id).run(settings.discord_bot_token, log_handler=None)


if __name__ == "__main__":
    main()
