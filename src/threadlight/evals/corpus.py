"""Load the synthetic eval corpus (evals/corpus/*.yaml) into Postgres as its own guild.

    python -m threadlight.evals.corpus            # install, then queue processing jobs

The corpus bypasses Discord so it can carry realistic historical timestamps. It goes
through the same tables and processing pipeline as real data; only ingestion differs.
IDs are deterministic snowflakes derived from timestamps, so ordering behaves like Discord.
"""

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from threadlight.db.models import Conversation, Guild
from threadlight.db.session import SessionLocal, engine
from threadlight.ingest import store
from threadlight.processing import jobs
from threadlight.retrieval.permissions import VIEW_CHANNEL

CORPUS_DIR = Path(__file__).resolve().parents[3] / "evals" / "corpus"
DEFAULT_CORPUS = CORPUS_DIR / "campus_eats.yaml"

DISCORD_EPOCH_MS = 1_420_070_400_000
# Low 22 bits of a snowflake are worker/process/sequence. A fixed worker/process pattern
# keeps synthetic ids from colliding with real ones minted at the same millisecond.
_SYNTHETIC_BITS = 0b11111_11111 << 12
_ID_EPOCH = datetime(2015, 6, 1, tzinfo=UTC)

# Plain members get read access through @everyone, like a typical small server.
EVERYONE_PERMISSIONS = VIEW_CHANNEL | (1 << 16) | (1 << 11)  # view, read history, send


def snowflake(when: datetime, seq: int = 0) -> int:
    ms = int(when.timestamp() * 1000)
    return ((ms - DISCORD_EPOCH_MS) << 22) | _SYNTHETIC_BITS | (seq & 0xFFF)


def entity_id(index: int) -> int:
    """Stable id for guild/channel/role/member number `index`."""
    return snowflake(_ID_EPOCH + timedelta(seconds=index))


@dataclass
class TruthDecision:
    key: str
    conversation_key: str
    title: str
    status: str
    message_id: int
    supersedes: str | None
    private: bool
    optional: bool = False


@dataclass
class CorpusMessage:
    id: int
    author: str
    content: str
    created_at: datetime


@dataclass
class CorpusConversation:
    key: str
    channel: str
    messages: list[CorpusMessage]
    ambiguous: bool = False
    decisions: list[TruthDecision] = field(default_factory=list)


@dataclass
class Corpus:
    guild_id: int
    guild_name: str
    owner: str
    member_ids: dict[str, int]
    role_ids: dict[str, int]
    role_members: dict[str, list[str]]
    channel_ids: dict[str, int]
    private_channels: dict[str, list[str]]  # channel -> roles that can see it
    conversations: list[CorpusConversation]

    @property
    def decisions(self) -> list[TruthDecision]:
        return [d for c in self.conversations for d in c.decisions]


def parse(path: Path = DEFAULT_CORPUS) -> Corpus:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    counter = iter(range(1, 10_000))
    guild_id = entity_id(next(counter))
    member_ids = {name: entity_id(next(counter)) for name in raw["members"]}
    role_ids = {r["name"]: entity_id(next(counter)) for r in raw.get("roles", [])}
    channel_ids = {c["name"]: entity_id(next(counter)) for c in raw["channels"]}
    private = {c["name"]: c["private_to"] for c in raw["channels"] if c.get("private_to")}

    conversations = []
    for c in raw["conversations"]:
        start = datetime.strptime(c["start"], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        messages = []
        for seq, (minutes, author, content) in enumerate(c["messages"]):
            if author not in member_ids:
                raise ValueError(f"{c['key']}: unknown author {author!r}")
            at = start + timedelta(minutes=minutes)
            messages.append(CorpusMessage(snowflake(at, seq), author, content, at))
        conv = CorpusConversation(
            key=c["key"],
            channel=c["channel"],
            messages=messages,
            ambiguous=c.get("ambiguous", False),
        )
        for d in c.get("decisions", []):
            conv.decisions.append(
                TruthDecision(
                    key=d["key"],
                    conversation_key=c["key"],
                    title=d["title"],
                    status=d["status"],
                    message_id=messages[d["message"] - 1].id,
                    supersedes=d.get("supersedes"),
                    private=c["channel"] in private,
                    optional=d.get("optional", False),
                )
            )
        conversations.append(conv)

    keys = {d.key for conv in conversations for d in conv.decisions}
    for conv in conversations:
        for d in conv.decisions:
            if d.supersedes and d.supersedes not in keys:
                raise ValueError(f"{d.key} supersedes unknown decision {d.supersedes!r}")

    return Corpus(
        guild_id=guild_id,
        guild_name=raw["guild"]["name"],
        owner=raw["guild"]["owner"],
        member_ids=member_ids,
        role_ids=role_ids,
        role_members={r["name"]: r["members"] for r in raw.get("roles", [])},
        channel_ids=channel_ids,
        private_channels=private,
        conversations=conversations,
    )


async def install(session: AsyncSession, corpus: Corpus) -> None:
    """Replace any previous copy of the corpus guild (cascades to all derived data)."""
    await session.execute(delete(Guild).where(Guild.id == corpus.guild_id))
    await store.upsert_guild(
        session, corpus.guild_id, corpus.guild_name, corpus.member_ids[corpus.owner]
    )
    await store.replace_roles(
        session,
        corpus.guild_id,
        [
            {
                "id": corpus.guild_id,
                "guild_id": corpus.guild_id,
                "name": "@everyone",
                "permissions": EVERYONE_PERMISSIONS,
            },
            *(
                {"id": rid, "guild_id": corpus.guild_id, "name": name, "permissions": 0}
                for name, rid in corpus.role_ids.items()
            ),
        ],
    )
    await store.replace_members(
        session,
        corpus.guild_id,
        [
            {
                "guild_id": corpus.guild_id,
                "user_id": uid,
                "display_name": name,
                "role_ids": [
                    corpus.role_ids[r]
                    for r, members in corpus.role_members.items()
                    if name in members
                ],
            }
            for name, uid in corpus.member_ids.items()
        ],
    )
    await store.upsert_channels(
        session,
        [
            {
                "id": cid,
                "guild_id": corpus.guild_id,
                "parent_id": None,
                "type": "text",
                "name": name,
            }
            for name, cid in corpus.channel_ids.items()
        ],
    )
    for channel, roles in corpus.private_channels.items():
        cid = corpus.channel_ids[channel]
        await store.replace_overwrites(
            session,
            cid,
            [
                {
                    "channel_id": cid,
                    "target_id": corpus.guild_id,
                    "target_type": "role",
                    "allow": 0,
                    "deny": VIEW_CHANNEL,
                },
                *(
                    {
                        "channel_id": cid,
                        "target_id": corpus.role_ids[r],
                        "target_type": "role",
                        "allow": VIEW_CHANNEL,
                        "deny": 0,
                    }
                    for r in roles
                ),
            ],
        )
    await store.upsert_messages(
        session,
        [
            {
                "id": m.id,
                "channel_id": corpus.channel_ids[conv.channel],
                "author_id": corpus.member_ids[m.author],
                "author_name": m.author,
                "content": m.content,
                "reply_to_id": None,
                "attachments": [],
                "created_at": m.created_at,
                "edited_at": None,
            }
            for conv in corpus.conversations
            for m in conv.messages
        ],
    )


async def segmentation_mismatches(session: AsyncSession, corpus: Corpus) -> list[str]:
    """Corpus conversations whose boundaries the segmenter didn't reproduce."""
    stored = set(
        await session.scalars(
            select(Conversation.first_message_id).where(
                Conversation.channel_id.in_(corpus.channel_ids.values())
            )
        )
    )
    expected = {c.messages[0].id: c.key for c in corpus.conversations}
    problems = [
        f"missing conversation {key!r}" for mid, key in expected.items() if mid not in stored
    ]
    problems += [
        f"unexpected conversation starting at message {mid}" for mid in stored - expected.keys()
    ]
    return problems


async def main(path: Path) -> None:
    corpus = parse(path)
    async with SessionLocal.begin() as session:
        await install(session, corpus)
        for cid in corpus.channel_ids.values():
            await jobs.enqueue_segment(session, cid)
    await engine.dispose()
    print(
        f"Installed {corpus.guild_name}: {len(corpus.conversations)} conversations, "
        f"{len(corpus.decisions)} labeled decisions. Guild id {corpus.guild_id}.\n"
        "Processing jobs queued; run: python -m threadlight.processing.worker --drain"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_CORPUS)
    asyncio.run(main(parser.parse_args().path))
