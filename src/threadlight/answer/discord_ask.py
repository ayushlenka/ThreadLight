"""The /ask and /decisions slash commands.

Replies are ephemeral (visible only to the asker). The answer is computed from the
channels *the asker* can read; posting it publicly would leak that content to everyone in
the current channel, including people without access to the source channels.
"""

import logging

import anthropic
import discord
import voyageai.error
from discord import app_commands

from threadlight.answer.rag import Answer, Answerer, ask, get_answerer, render_markdown
from threadlight.db.session import SessionLocal
from threadlight.processing.embedder import Embedder, get_embedder
from threadlight.retrieval.decisions import DecisionView, list_decisions, search_decisions
from threadlight.retrieval.permissions import visible_channel_ids

log = logging.getLogger(__name__)

EMBED_COLOR = 0x5865F2
DECISIONS_SHOWN = 10
STATUS_ICON = {"decided": "✅", "proposed": "💭", "superseded": "↩️"}


def answer_embed(question: str, answer: Answer) -> discord.Embed:
    embed = discord.Embed(description=render_markdown(answer), color=EMBED_COLOR)
    embed.set_author(name=question[:256])
    footer = "Only you can see this answer."
    if answer.model:
        footer += f" · {answer.model}"
    embed.set_footer(text=footer)
    return embed


def register_ask_command(tree: app_commands.CommandTree, guild_id: int) -> None:
    embedder: Embedder | None = None
    answerer: Answerer | None = None
    try:
        embedder, answerer = get_embedder(), get_answerer()
    except RuntimeError as exc:
        log.warning("/ask disabled: %s", exc)

    @tree.command(
        name="ask",
        description="Ask what this server discussed or decided",
        guild=discord.Object(id=guild_id),
    )
    @app_commands.describe(question="e.g. What did we decide about the deploy schedule?")
    async def ask_command(interaction: discord.Interaction, question: str) -> None:
        if interaction.guild_id != guild_id:
            await interaction.response.send_message("Not available here.", ephemeral=True)
            return
        if embedder is None or answerer is None:
            await interaction.response.send_message(
                "/ask isn't configured yet (missing API keys).", ephemeral=True
            )
            return

        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            async with SessionLocal() as session:
                answer = await ask(
                    session, embedder, answerer, guild_id, interaction.user.id, question
                )
            await interaction.followup.send(embed=answer_embed(question, answer), ephemeral=True)
        except (voyageai.error.RateLimitError, anthropic.RateLimitError):
            await interaction.followup.send(
                "I'm being rate limited right now. Try again in a minute.", ephemeral=True
            )
        except Exception:
            log.exception("/ask failed for question %r", question)
            await interaction.followup.send("Something went wrong answering that.", ephemeral=True)


def decisions_embed(title: str, views: list[DecisionView]) -> discord.Embed:
    lines = []
    for v in views[:DECISIONS_SHOWN]:
        line = (
            f"{STATUS_ICON[v.status]} **[{v.title}]({v.jump_url})**\n"
            f"-# #{v.channel_name} · {v.decided_at:%b %d, %Y} · {v.status}"
        )
        if v.replaced_by:
            line += f"\n-# replaced by: {v.replaced_by.title} ({v.replaced_by.decided_at:%b %d})"
        lines.append(line)
    body = "\n\n".join(lines) or "No decisions found in the channels you can access."
    embed = discord.Embed(title=title[:256], description=body[:4096], color=EMBED_COLOR)
    embed.set_footer(text="Only you can see this. Extracted automatically; check the source.")
    return embed


def register_decisions_command(tree: app_commands.CommandTree, guild_id: int) -> None:
    try:
        embedder: Embedder | None = get_embedder()
    except RuntimeError:
        embedder = None

    @tree.command(
        name="decisions",
        description="Show the decision log, optionally about a topic",
        guild=discord.Object(id=guild_id),
    )
    @app_commands.describe(topic="Optional, e.g. 'hosting' or 'auth'")
    async def decisions_command(interaction: discord.Interaction, topic: str | None = None) -> None:
        if interaction.guild_id != guild_id:
            await interaction.response.send_message("Not available here.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            async with SessionLocal() as session:
                visible = await visible_channel_ids(session, guild_id, interaction.user.id)
                if topic and embedder is not None:
                    vec = await embedder.embed_query(topic)
                    views = await search_decisions(session, topic, vec, visible)
                    title = f"Decisions about: {topic}"
                else:
                    views = await list_decisions(session, visible, limit=DECISIONS_SHOWN)
                    title = "Recent decisions"
            await interaction.followup.send(embed=decisions_embed(title, views), ephemeral=True)
        except voyageai.error.RateLimitError:
            await interaction.followup.send(
                "I'm being rate limited right now. Try again in a minute.", ephemeral=True
            )
        except Exception:
            log.exception("/decisions failed for topic %r", topic)
            await interaction.followup.send("Something went wrong.", ephemeral=True)
