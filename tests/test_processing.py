from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from tests.fakes import HashingEmbedder
from threadlight.db.models import Conversation, ConversationMessage, Job
from threadlight.db.session import SessionLocal
from threadlight.ingest import store
from threadlight.processing import jobs
from threadlight.processing.pipeline import embed_pending, resegment_channel

GUILD_ID = 1
CHANNEL_ID = 10
T0 = datetime(2024, 2, 13, 14, 0, tzinfo=UTC)


def row(message_id: int, minutes: float, content: str):
    return {
        "id": message_id,
        "channel_id": CHANNEL_ID,
        "author_id": 7,
        "author_name": "alice",
        "content": content,
        "reply_to_id": None,
        "attachments": [],
        "created_at": T0 + timedelta(minutes=minutes),
        "edited_at": None,
    }


async def seed(messages):
    async with SessionLocal.begin() as s:
        await store.upsert_guild(s, GUILD_ID, "g")
        await store.upsert_channels(
            s,
            [
                {
                    "id": CHANNEL_ID,
                    "guild_id": GUILD_ID,
                    "parent_id": None,
                    "type": "text",
                    "name": "eng",
                }
            ],
        )
        await store.upsert_messages(s, messages)


async def resegment():
    async with SessionLocal.begin() as s:
        return await resegment_channel(s, CHANNEL_ID)


async def conversations() -> list[Conversation]:
    async with SessionLocal() as s:
        return list(await s.scalars(select(Conversation).order_by(Conversation.started_at)))


# pipeline


async def test_resegment_creates_conversations_with_links(db):
    await seed([row(1, 0, "auth plan"), row(2, 5, "use OAuth"), row(3, 300, "lunch?")])

    result = await resegment()

    assert (result.created, result.kept, result.removed) == (2, 0, 0)
    first, second = await conversations()
    assert first.message_count == 2 and first.first_message_id == 1
    assert second.first_message_id == 3
    async with SessionLocal() as s:
        assert await s.scalar(select(func.count()).select_from(ConversationMessage)) == 3


async def test_resegment_keeps_unchanged_and_replaces_changed(db):
    await seed([row(1, 0, "auth plan"), row(2, 300, "lunch?")])
    await resegment()
    before = {c.first_message_id: c.id for c in await conversations()}

    # A new message joins the second conversation; the first is untouched.
    async with SessionLocal.begin() as s:
        await store.upsert_messages(s, [row(3, 305, "tacos")])
    result = await resegment()

    assert (result.created, result.kept, result.removed) == (1, 1, 1)
    after = {c.first_message_id: c.id for c in await conversations()}
    assert after[1] == before[1]  # same row: embedding and decisions survive
    assert after[2] != before[2]


async def test_deleted_message_drops_out_of_conversation_text(db):
    await seed([row(1, 0, "public"), row(2, 1, "secret token abc")])
    await resegment()

    async with SessionLocal.begin() as s:
        await store.mark_messages_deleted(s, [2])
    await resegment()

    [conv] = await conversations()
    assert "secret" not in conv.text
    assert conv.message_count == 1


async def test_resegment_is_idempotent(db):
    await seed([row(1, 0, "a"), row(2, 300, "b")])
    await resegment()
    result = await resegment()
    assert (result.created, result.kept, result.removed) == (0, 2, 0)


async def test_embed_pending_only_embeds_new_conversations(db):
    await seed([row(1, 0, "auth plan")])
    await resegment()
    embedder = HashingEmbedder()

    async with SessionLocal.begin() as s:
        assert await embed_pending(s, embedder) == 1
    async with SessionLocal.begin() as s:
        assert await embed_pending(s, embedder) == 0

    assert len(embedder.document_calls) == 1
    [conv] = await conversations()
    assert conv.embedding is not None


# job queue


async def pending_jobs() -> list[Job]:
    async with SessionLocal() as s:
        return list(await s.scalars(select(Job).where(Job.status == "pending")))


async def test_enqueue_dedupes_pending_jobs(db):
    async with SessionLocal.begin() as s:
        await jobs.enqueue_segment(s, CHANNEL_ID)
        await jobs.enqueue_segment(s, CHANNEL_ID)
        await jobs.enqueue_segment(s, 99)
    assert len(await pending_jobs()) == 2


async def test_new_job_can_queue_while_one_is_running(db):
    async with SessionLocal.begin() as s:
        await jobs.enqueue_segment(s, CHANNEL_ID)
    async with SessionLocal.begin() as s:
        job = await jobs.claim(s)
    assert job.status == "running" and job.attempts == 1

    async with SessionLocal.begin() as s:
        await jobs.enqueue_segment(s, CHANNEL_ID)
    assert len(await pending_jobs()) == 1


async def test_claim_respects_run_after(db):
    async with SessionLocal.begin() as s:
        await jobs.enqueue_segment(s, CHANNEL_ID, delay=timedelta(minutes=5))
    async with SessionLocal.begin() as s:
        assert await jobs.claim(s) is None


async def test_fail_requeues_with_backoff_then_gives_up(db):
    async with SessionLocal.begin() as s:
        await jobs.enqueue_segment(s, CHANNEL_ID)
    async with SessionLocal.begin() as s:
        job = await jobs.claim(s)
    async with SessionLocal.begin() as s:
        await jobs.fail(s, job, "boom")

    [requeued] = await pending_jobs()
    assert requeued.last_error == "boom"
    assert requeued.run_after > datetime.now(UTC)

    job.attempts = jobs.MAX_ATTEMPTS
    async with SessionLocal.begin() as s:
        await jobs.fail(s, job, "boom again")
        assert (await s.get(Job, job.id)).status == "failed"


async def test_fail_retires_job_superseded_by_newer_pending(db):
    async with SessionLocal.begin() as s:
        await jobs.enqueue_segment(s, CHANNEL_ID)
    async with SessionLocal.begin() as s:
        job = await jobs.claim(s)
    async with SessionLocal.begin() as s:
        await jobs.enqueue_segment(s, CHANNEL_ID)  # more messages arrived meanwhile
    async with SessionLocal.begin() as s:
        await jobs.fail(s, job, "boom")  # must not violate the pending-dedupe index

    async with SessionLocal() as s:
        assert (await s.get(Job, job.id)).status == "failed"
    assert len(await pending_jobs()) == 1
