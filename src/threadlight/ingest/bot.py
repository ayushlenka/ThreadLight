"""Discord bot: syncs history on startup, then mirrors live changes into Postgres.

Run with `python -m threadlight.ingest.bot`.

Raw events are used for edits and deletes because discord.py only caches recent messages;
the non-raw variants silently miss anything older than the cache.
"""

import asyncio
import logging

import discord

from threadlight.config import get_settings
from threadlight.db.session import SessionLocal
from threadlight.ingest import store
from threadlight.ingest.backfill import sync_guild
from threadlight.ingest.convert import channel_row, message_row

log = logging.getLogger(__name__)


class IngestBot(discord.Client):
    def __init__(self, guild_id: int) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True  # needed later to compute per-user channel permissions
        super().__init__(intents=intents)
        self.guild_id = guild_id
        self._known_channels: set[int] = set()
        self._sync_task: asyncio.Task | None = None

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
        async with SessionLocal.begin() as session:
            await store.upsert_guild(session, channel.guild.id, channel.guild.name)
            await store.upsert_channels(session, [channel_row(channel)])
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

    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent) -> None:
        if not self._ours(payload.guild_id):
            return
        if (row := message_row(payload.message)) is None:
            return
        await self._ensure_channel(payload.message.channel)
        async with SessionLocal.begin() as session:
            await store.upsert_messages(session, [row])

    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent) -> None:
        if not self._ours(payload.guild_id):
            return
        async with SessionLocal.begin() as session:
            await store.mark_messages_deleted(session, [payload.message_id])

    async def on_raw_bulk_message_delete(self, payload: discord.RawBulkMessageDeleteEvent) -> None:
        if not self._ours(payload.guild_id):
            return
        async with SessionLocal.begin() as session:
            await store.mark_messages_deleted(session, payload.message_ids)

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


def main() -> None:
    settings = get_settings()
    if not settings.discord_bot_token or settings.discord_guild_id is None:
        raise SystemExit("Set DISCORD_BOT_TOKEN and DISCORD_GUILD_ID in .env")
    discord.utils.setup_logging(level=logging.INFO)
    IngestBot(settings.discord_guild_id).run(settings.discord_bot_token, log_handler=None)


if __name__ == "__main__":
    main()
