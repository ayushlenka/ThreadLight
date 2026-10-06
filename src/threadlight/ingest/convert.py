"""Convert discord.py objects into plain row dicts for the store layer."""

from typing import Any

import discord

# User-authored message types. System messages (joins, pins, boosts, thread-starter stubs)
# carry no discussion content and would only add noise to retrieval.
INGESTED_TYPES = {discord.MessageType.default, discord.MessageType.reply}

IngestableChannel = discord.TextChannel | discord.Thread


def channel_row(channel: discord.abc.GuildChannel | discord.Thread) -> dict[str, Any]:
    if isinstance(channel, discord.Thread):
        parent_id = channel.parent_id
    else:
        parent_id = channel.category_id
    return {
        "id": channel.id,
        "guild_id": channel.guild.id,
        "parent_id": parent_id,
        "type": channel.type.name,
        "name": channel.name,
    }


def message_row(message: discord.Message) -> dict[str, Any] | None:
    """Row for a message, or None if it isn't something we index."""
    if message.type not in INGESTED_TYPES:
        return None
    if not message.content and not message.attachments:
        return None

    reply_to_id = None
    if message.type == discord.MessageType.reply and message.reference is not None:
        reply_to_id = message.reference.message_id

    return {
        "id": message.id,
        "channel_id": message.channel.id,
        "author_id": message.author.id,
        "author_name": message.author.display_name,
        "content": message.content,
        "reply_to_id": reply_to_id,
        "attachments": [
            {"filename": a.filename, "url": a.url, "content_type": a.content_type}
            for a in message.attachments
        ],
        "created_at": message.created_at,
        "edited_at": message.edited_at,
    }
