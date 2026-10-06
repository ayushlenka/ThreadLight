from datetime import UTC, datetime
from types import SimpleNamespace

import discord
from sqlalchemy import select

from threadlight.db.models import Channel, Message
from threadlight.db.session import SessionLocal
from threadlight.ingest import store
from threadlight.ingest.convert import message_row

GUILD_ID = 1
CHANNEL_ID = 10
T0 = datetime(2024, 2, 13, 14, 0, tzinfo=UTC)


def fake_message(**overrides):
    """Duck-typed stand-in for discord.Message with only the fields message_row reads."""
    fields = {
        "id": 100,
        "type": discord.MessageType.default,
        "content": "let's move auth to OAuth",
        "attachments": [],
        "reference": None,
        "channel": SimpleNamespace(id=CHANNEL_ID),
        "author": SimpleNamespace(id=7, display_name="alice"),
        "created_at": T0,
        "edited_at": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def row(message_id: int, content: str = "hello", **extra):
    return {
        "id": message_id,
        "channel_id": CHANNEL_ID,
        "author_id": 7,
        "author_name": "alice",
        "content": content,
        "reply_to_id": None,
        "attachments": [],
        "created_at": T0,
        "edited_at": None,
    } | extra


# message_row


def test_message_row_maps_fields():
    r = message_row(fake_message())
    assert r["id"] == 100
    assert r["channel_id"] == CHANNEL_ID
    assert r["author_name"] == "alice"
    assert r["content"] == "let's move auth to OAuth"
    assert r["reply_to_id"] is None


def test_message_row_captures_reply_target():
    msg = fake_message(type=discord.MessageType.reply, reference=SimpleNamespace(message_id=99))
    assert message_row(msg)["reply_to_id"] == 99


def test_message_row_skips_system_messages():
    assert message_row(fake_message(type=discord.MessageType.pins_add)) is None
    assert message_row(fake_message(type=discord.MessageType.new_member)) is None


def test_message_row_skips_empty_messages():
    assert message_row(fake_message(content="")) is None


def test_message_row_keeps_attachment_only_messages():
    attachment = SimpleNamespace(
        filename="arch.png", url="https://x/arch.png", content_type="image/png"
    )
    r = message_row(fake_message(content="", attachments=[attachment]))
    assert r["attachments"] == [
        {"filename": "arch.png", "url": "https://x/arch.png", "content_type": "image/png"}
    ]


# store


async def seed_channel():
    async with SessionLocal.begin() as s:
        await store.upsert_guild(s, GUILD_ID, "test guild")
        await store.upsert_channels(
            s,
            [
                {
                    "id": CHANNEL_ID,
                    "guild_id": GUILD_ID,
                    "parent_id": None,
                    "type": "text",
                    "name": "engineering",
                }
            ],
        )


async def get_message(message_id: int) -> Message:
    async with SessionLocal() as s:
        return await s.scalar(select(Message).where(Message.id == message_id))


async def test_upsert_is_idempotent_and_applies_edits(db):
    await seed_channel()
    async with SessionLocal.begin() as s:
        await store.upsert_messages(s, [row(1, "first draft")])
    async with SessionLocal.begin() as s:
        await store.upsert_messages(s, [row(1, "edited", edited_at=T0)])

    msg = await get_message(1)
    assert msg.content == "edited"
    assert msg.edited_at == T0


async def test_delete_scrubs_content_and_is_not_resurrected(db):
    await seed_channel()
    async with SessionLocal.begin() as s:
        await store.upsert_messages(s, [row(1, "secret plan")])
    async with SessionLocal.begin() as s:
        await store.mark_messages_deleted(s, [1])
    # A backfill page fetched before the delete arrives afterwards.
    async with SessionLocal.begin() as s:
        await store.upsert_messages(s, [row(1, "secret plan")])

    msg = await get_message(1)
    assert msg.deleted_at is not None
    assert msg.content == ""


async def test_backfill_page_advances_cursor(db):
    await seed_channel()
    async with SessionLocal.begin() as s:
        await store.save_backfill_page(s, CHANNEL_ID, [row(1), row(2)], cursor=2)
    async with SessionLocal.begin() as s:
        await store.mark_backfill_done(s, CHANNEL_ID)

    async with SessionLocal() as s:
        channel = await s.get(Channel, CHANNEL_ID)
        assert channel.backfill_cursor == 2
        assert channel.backfill_done is True
        assert await store.get_backfill_cursor(s, CHANNEL_ID) == 2


async def test_channel_delete_and_recreate(db):
    await seed_channel()
    async with SessionLocal.begin() as s:
        await store.mark_channel_deleted(s, CHANNEL_ID)
    async with SessionLocal() as s:
        assert (await s.get(Channel, CHANNEL_ID)).deleted_at is not None

    await seed_channel()  # e.g. permissions restored / channel update event
    async with SessionLocal() as s:
        assert (await s.get(Channel, CHANNEL_ID)).deleted_at is None
