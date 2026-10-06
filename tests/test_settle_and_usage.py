from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

from sqlalchemy import select

from tests.fakes import HashingEmbedder
from threadlight.db.models import ApiUsage, Job
from threadlight.db.session import SessionLocal
from threadlight.ingest import store
from threadlight.processing import jobs
from threadlight.processing.decisions import (
    SETTLE_AFTER,
    ExtractedDecision,
    ExtractionResult,
    SupersessionResult,
    extract_channel,
)
from threadlight.processing.pipeline import resegment_channel
from threadlight.processing.worker import WorkerContext, handle_segment_channel
from threadlight.usage import (
    cost_usd,
    record_claude_usage,
    record_usage,
    summarize,
    usage_tags,
)

GUILD, CHANNEL = 1, 10
NOW = datetime(2026, 10, 6, 18, 0, tzinfo=UTC)


class CountingLLM:
    def __init__(self):
        self.extract_calls = 0

    async def extract(self, transcript):
        self.extract_calls += 1
        return ExtractionResult(
            decisions=[
                ExtractedDecision(
                    title="Ship it",
                    summary="Ship it.",
                    status="decided",
                    rationale=[],
                    alternatives=[],
                    decided_message=1,
                    source_messages=[1],
                )
            ]
        )

    async def find_replacements(self, listing):
        return SupersessionResult(replacements=[])


async def seed_channel():
    async with SessionLocal.begin() as s:
        await store.upsert_guild(s, GUILD, "g")
        await store.upsert_channels(
            s,
            [{"id": CHANNEL, "guild_id": GUILD, "parent_id": None, "type": "text", "name": "eng"}],
        )


async def post(message_id: int, at: datetime, content: str = "let's ship it"):
    async with SessionLocal.begin() as s:
        await store.upsert_messages(
            s,
            [
                {
                    "id": message_id,
                    "channel_id": CHANNEL,
                    "author_id": 7,
                    "author_name": "alice",
                    "content": content,
                    "reply_to_id": None,
                    "attachments": [],
                    "created_at": at,
                    "edited_at": None,
                }
            ],
        )
        await resegment_channel(s, CHANNEL)


# Settled-only extraction


async def test_active_conversation_is_not_extracted_until_settled(db):
    await seed_channel()
    llm, embedder = CountingLLM(), HashingEmbedder()

    # A conversation that keeps growing: each message re-segments it into a new row.
    for i in range(5):
        at = NOW - timedelta(minutes=10 - i)
        await post(100 + i, at)
        stats = await extract_channel(SessionLocal, llm, embedder, CHANNEL, now=NOW)
        assert stats.conversations == 0
        assert stats.next_settle_at == at + SETTLE_AFTER

    assert llm.extract_calls == 0

    # Once quiet for longer than the gap, it's extracted exactly once.
    later = NOW + SETTLE_AFTER + timedelta(minutes=1)
    stats = await extract_channel(SessionLocal, llm, embedder, CHANNEL, now=later)
    assert (stats.conversations, stats.decisions, stats.next_settle_at) == (1, 1, None)
    await extract_channel(SessionLocal, llm, embedder, CHANNEL, now=later)
    assert llm.extract_calls == 1


async def test_settled_conversations_extract_while_active_ones_wait(db):
    await seed_channel()
    await post(100, NOW - timedelta(days=2), "old: let's use postgres")
    await post(101, NOW - timedelta(minutes=5), "new: let's ship friday")
    llm = CountingLLM()

    stats = await extract_channel(SessionLocal, llm, HashingEmbedder(), CHANNEL, now=NOW)
    assert stats.conversations == 1
    assert stats.next_settle_at == NOW - timedelta(minutes=5) + SETTLE_AFTER


async def test_worker_schedules_follow_up_extraction_for_active_conversation(db):
    await seed_channel()
    recent = datetime.now(UTC) - timedelta(minutes=1)
    await post(100, recent)
    ctx = WorkerContext(embedder=HashingEmbedder(), decision_llm=CountingLLM())

    await handle_segment_channel({"channel_id": CHANNEL}, ctx)

    async with SessionLocal() as s:
        job = await s.scalar(select(Job).where(Job.kind == jobs.EXTRACT_CHANNEL))
    assert job is not None and job.status == "pending"
    expected = recent + SETTLE_AFTER
    assert expected <= job.run_after <= expected + timedelta(minutes=2)
    assert ctx.decision_llm.extract_calls == 0


# Usage recording


def test_cost_is_computed_from_list_prices():
    # Opus 5.5: $4 in / $20 out / $0.20 cache read per million tokens.
    assert cost_usd("claude-opus-5-5", 1_000_000, 0) == Decimal("4.0")
    assert cost_usd("claude-opus-5-5", 1000, 500, cache_read_tokens=10_000) == Decimal("0.016")
    assert cost_usd("voyage-3.5", 1_000_000, 0) == Decimal("0.06")
    assert cost_usd("some-unknown-model", 1000, 1000) is None


async def test_record_usage_writes_tagged_row(db):
    with usage_tags(guild_id=GUILD), usage_tags(user_id=42):
        await record_usage(
            provider="anthropic",
            model="claude-opus-5-5",
            purpose="answer",
            input_tokens=5000,
            output_tokens=800,
        )
    await record_usage(
        provider="voyage", model="voyage-3.5", purpose="embed_query", input_tokens=12
    )

    async with SessionLocal() as s:
        rows = {r.purpose: r for r in await s.scalars(select(ApiUsage))}
    assert rows["answer"].tags == {"guild_id": GUILD, "user_id": 42}
    assert rows["answer"].cost_usd == Decimal("0.036000")
    assert rows["embed_query"].tags == {}


async def test_record_claude_usage_reads_response_usage(db):
    response = SimpleNamespace(
        model="claude-sonnet-5-5",
        usage=SimpleNamespace(
            input_tokens=2000,
            output_tokens=300,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
        ),
    )
    await record_claude_usage(response, "extract")
    async with SessionLocal() as s:
        row = await s.scalar(select(ApiUsage))
    assert (row.model, row.purpose, row.input_tokens, row.output_tokens) == (
        "claude-sonnet-5-5",
        "extract",
        2000,
        300,
    )
    assert row.cost_usd == Decimal("0.007000")


async def test_recording_failure_never_raises(db, monkeypatch):
    import threadlight.db.session as session_module

    class Broken:
        def begin(self):
            raise RuntimeError("database down")

    monkeypatch.setattr(session_module, "SessionLocal", Broken())
    await record_usage(provider="voyage", model="voyage-3.5", purpose="embed_query", input_tokens=1)


async def test_summarize_groups_by_purpose_and_model(db):
    for _ in range(3):
        await record_usage(
            provider="anthropic",
            model="claude-opus-5-5",
            purpose="answer",
            input_tokens=1000,
            output_tokens=100,
        )
    await record_usage(
        provider="voyage", model="voyage-3.5", purpose="embed_query", input_tokens=10
    )

    async with SessionLocal() as s:
        lines = {line.purpose: line for line in await summarize(s, NOW - timedelta(days=30))}
    answer = lines["answer"]
    assert (answer.calls, answer.input_tokens, answer.output_tokens) == (3, 3000, 300)
    assert answer.cost_per_call == Decimal("0.006")
