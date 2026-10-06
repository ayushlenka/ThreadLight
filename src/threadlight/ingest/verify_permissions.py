"""Differential check: our permission resolution vs discord.py, on the live guild.

    python -m threadlight.ingest.verify_permissions

For every (member, channel) pair, compares `visible_channel_ids` (computed from the state
stored in Postgres) with discord.py's `permissions_for` on the gateway cache. Run it after
the bot has synced; any mismatch means either the algorithm or the stored state is wrong.

Private threads are skipped: discord.py treats them like their parent, while we
deliberately hide them from non-moderators (see retrieval/permissions.py).
"""

import logging

import discord
from sqlalchemy import select

from threadlight.config import get_settings
from threadlight.db.models import Channel
from threadlight.db.session import SessionLocal, engine
from threadlight.retrieval.permissions import visible_channel_ids

log = logging.getLogger(__name__)


def discord_can_read(channel: discord.abc.GuildChannel | discord.Thread, member) -> bool:
    perms = channel.permissions_for(member)
    return perms.view_channel and perms.read_message_history


class Verifier(discord.Client):
    def __init__(self, guild_id: int) -> None:
        intents = discord.Intents.default()
        intents.members = True
        super().__init__(intents=intents)
        self.guild_id = guild_id
        self.mismatches = 0

    async def on_ready(self) -> None:
        try:
            await self.verify()
        finally:
            await engine.dispose()
            await self.close()

    async def verify(self) -> None:
        guild = self.get_guild(self.guild_id)
        if guild is None:
            log.error("bot is not in guild %s", self.guild_id)
            self.mismatches = -1
            return

        async with SessionLocal() as session:
            stored = (
                await session.execute(
                    select(Channel.id, Channel.type).where(
                        Channel.guild_id == guild.id, Channel.deleted_at.is_(None)
                    )
                )
            ).all()

            pairs = 0
            for member in guild.members:
                ours = set(await visible_channel_ids(session, guild.id, member.id))
                for channel_id, channel_type in stored:
                    if channel_type == "private_thread":
                        continue
                    channel = guild.get_channel_or_thread(channel_id)
                    if channel is None:
                        try:
                            channel = await guild.fetch_channel(channel_id)
                        except discord.NotFound:
                            log.warning("channel %s no longer exists; skipping", channel_id)
                            continue
                    pairs += 1
                    expected = discord_can_read(channel, member)
                    if (channel_id in ours) != expected:
                        self.mismatches += 1
                        log.error(
                            "MISMATCH %s in #%s: discord.py=%s ours=%s",
                            member.display_name,
                            channel.name,
                            expected,
                            channel_id in ours,
                        )

        log.info(
            "checked %d (member, channel) pairs across %d members: %d mismatches",
            pairs,
            len(guild.members),
            self.mismatches,
        )


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("discord").setLevel(logging.WARNING)
    client = Verifier(settings.discord_guild_id)
    client.run(settings.discord_bot_token, log_handler=None)
    raise SystemExit(1 if client.mismatches else 0)


if __name__ == "__main__":
    main()
