from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient

from tests.fakes import HashingEmbedder
from threadlight.answer.rag import (
    NO_RESULTS,
    REFUSED,
    Answer,
    Source,
    ask,
    get_answerer,
    render_markdown,
)
from threadlight.api.main import app
from threadlight.config import get_settings
from threadlight.db.session import SessionLocal
from threadlight.ingest import store
from threadlight.processing.embedder import get_embedder
from threadlight.processing.pipeline import embed_pending, resegment_channel
from threadlight.retrieval.permissions import READ, VIEW_CHANNEL

GUILD = 1
ENG, LEADS = 10, 11
ALICE, BOB = 900, 901
T0 = datetime(2024, 2, 13, 14, 0, tzinfo=UTC)
NOW = datetime(2024, 3, 1, tzinfo=UTC)

ENG_MESSAGES = [
    (100, ALICE, "alice", 0, "should we move auth to oauth?"),
    (101, BOB, "bob", 2, "yes, oauth fixes the password reset mess"),
    (102, ALICE, "alice", 5, "decided: migrate to google oauth in february"),
    (103, BOB, "bob", 60 * 24 * 3, "anyone want lunch"),
]
LEADS_MESSAGES = [(200, ALICE, "alice", 0, "the oauth vendor budget is confidential")]


def text_block(text, citations=None):
    return SimpleNamespace(type="text", text=text, citations=citations)


def cite(doc, start, end=None, cited="..."):
    return SimpleNamespace(
        type="content_block_location",
        document_index=doc,
        start_block_index=start,
        end_block_index=end if end is not None else start + 1,
        cited_text=cited,
    )


class FakeAnswerer:
    """Records what would be sent to Claude; replies via a scripted function."""

    def __init__(self, reply=None):
        self.calls = []
        self.reply = reply or (
            lambda docs: SimpleNamespace(
                model="fake-model",
                stop_reason="end_turn",
                content=[
                    SimpleNamespace(type="thinking", thinking=""),
                    text_block("The team decided to migrate to Google OAuth.", [cite(0, 2)]),
                    text_block(" It fixes password resets.", [cite(0, 1), cite(0, 2)]),
                ],
            )
        )

    async def generate(self, documents, question, now):
        self.calls.append({"documents": documents, "question": question, "now": now})
        return self.reply(documents)


async def seed():
    async with SessionLocal.begin() as s:
        await store.upsert_guild(s, GUILD, "g")
        await store.replace_roles(
            s, GUILD, [{"id": GUILD, "guild_id": GUILD, "name": "@everyone", "permissions": READ}]
        )
        await store.upsert_members(
            s,
            [
                {"guild_id": GUILD, "user_id": uid, "display_name": n, "role_ids": []}
                for uid, n in [(ALICE, "alice"), (BOB, "bob")]
            ],
        )
        await store.upsert_channels(
            s,
            [
                {"id": ENG, "guild_id": GUILD, "parent_id": None, "type": "text", "name": "eng"},
                {
                    "id": LEADS,
                    "guild_id": GUILD,
                    "parent_id": None,
                    "type": "text",
                    "name": "leads",
                },
            ],
        )
        # #leads is hidden from @everyone; only Alice can see it.
        await store.replace_overwrites(
            s,
            LEADS,
            [
                {
                    "channel_id": LEADS,
                    "target_id": GUILD,
                    "target_type": "role",
                    "allow": 0,
                    "deny": VIEW_CHANNEL,
                },
                {
                    "channel_id": LEADS,
                    "target_id": ALICE,
                    "target_type": "member",
                    "allow": VIEW_CHANNEL,
                    "deny": 0,
                },
            ],
        )
        for channel, msgs in [(ENG, ENG_MESSAGES), (LEADS, LEADS_MESSAGES)]:
            await store.upsert_messages(
                s,
                [
                    {
                        "id": mid,
                        "channel_id": channel,
                        "author_id": uid,
                        "author_name": name,
                        "content": content,
                        "reply_to_id": None,
                        "attachments": [],
                        "created_at": T0 + timedelta(minutes=minutes),
                        "edited_at": None,
                    }
                    for mid, uid, name, minutes, content in msgs
                ],
            )
            await resegment_channel(s, channel)
    embedder = HashingEmbedder()
    async with SessionLocal.begin() as s:
        await embed_pending(s, embedder)
    return embedder


async def run_ask(embedder, answerer, user_id, question="what did we decide about oauth?"):
    async with SessionLocal.begin() as s:
        return await ask(s, embedder, answerer, GUILD, user_id, question, now=NOW)


def doc_titles(call):
    return [d["title"] for d in call["documents"]]


async def test_sends_one_content_block_per_message(db):
    embedder = await seed()
    answerer = FakeAnswerer()
    await run_ask(embedder, answerer, BOB)

    [call] = answerer.calls
    eng_doc = next(d for d in call["documents"] if d["title"].startswith("#eng (2024-02-13)"))
    blocks = [b["text"] for b in eng_doc["source"]["content"]]
    assert blocks == [
        "[2024-02-13 14:00] alice: should we move auth to oauth?",
        "[2024-02-13 14:02] bob: yes, oauth fixes the password reset mess",
        "[2024-02-13 14:05] alice: decided: migrate to google oauth in february",
    ]
    assert eng_doc["citations"] == {"enabled": True}
    assert call["now"] == NOW


async def test_hidden_channels_never_reach_the_model(db):
    embedder = await seed()
    bob, alice = FakeAnswerer(), FakeAnswerer()
    await run_ask(embedder, bob, BOB, "oauth vendor budget")
    await run_ask(embedder, alice, ALICE, "oauth vendor budget")

    assert not any(t.startswith("#leads") for t in doc_titles(bob.calls[0]))
    assert any(t.startswith("#leads") for t in doc_titles(alice.calls[0]))


async def test_citations_map_to_exact_messages(db):
    embedder = await seed()
    answer = await run_ask(embedder, FakeAnswerer(), BOB)

    assert answer.text == (
        "The team decided to migrate to Google OAuth.[1] It fixes password resets.[1][2]"
    )
    first, second = answer.sources
    assert (first.number, first.author_name) == (1, "alice")
    assert first.jump_url == f"https://discord.com/channels/{GUILD}/{ENG}/102"
    assert (second.number, second.author_name) == (2, "bob")
    assert second.jump_url.endswith("/101")
    assert answer.model == "fake-model"


async def test_deleted_message_is_excluded_before_resegmentation(db):
    embedder = await seed()
    async with SessionLocal.begin() as s:
        await store.mark_messages_deleted(s, [101])  # no resegment yet

    answerer = FakeAnswerer(
        lambda docs: SimpleNamespace(model="m", stop_reason="end_turn", content=[text_block("ok")])
    )
    await run_ask(embedder, answerer, BOB)
    all_text = str(answerer.calls[0]["documents"])
    assert "password reset" not in all_text
    assert "migrate to google oauth" in all_text


async def test_non_member_gets_no_results_without_calling_the_model(db):
    embedder = await seed()
    answerer = FakeAnswerer()
    answer = await run_ask(embedder, answerer, 12345)
    assert answer.text == NO_RESULTS and not answer.answered
    assert answerer.calls == []


async def test_refusal_and_truncation(db):
    embedder = await seed()
    refused = await run_ask(
        embedder,
        FakeAnswerer(lambda d: SimpleNamespace(model="m", stop_reason="refusal", content=[])),
        BOB,
    )
    assert refused.text == REFUSED and not refused.answered

    truncated = await run_ask(
        embedder,
        FakeAnswerer(
            lambda d: SimpleNamespace(
                model="m", stop_reason="max_tokens", content=[text_block("partial")]
            )
        ),
        BOB,
    )
    assert truncated.text.startswith("partial") and "cut off" in truncated.text


async def test_out_of_range_citations_are_ignored(db):
    embedder = await seed()
    answerer = FakeAnswerer(
        lambda d: SimpleNamespace(
            model="m",
            stop_reason="end_turn",
            content=[text_block("x", [cite(99, 0), cite(0, 99)])],
        )
    )
    answer = await run_ask(embedder, answerer, BOB)
    assert answer.text == "x" and answer.sources == []


def test_render_markdown_drops_sources_that_do_not_fit():
    sources = [
        Source(n, "eng", "alice", T0, f"https://discord.com/channels/1/2/{n}", "...")
        for n in range(1, 4)
    ]
    full = render_markdown(Answer(text="Answer.", sources=sources))
    assert full.startswith("Answer.\n\n**Sources**\n`[1]` [#eng · Feb 13, 2024 · alice]")
    assert full.count("`[") == 3

    one_line = len(full.splitlines()[-1])
    short = render_markdown(Answer(text="Answer.", sources=sources), max_chars=len(full) - one_line)
    assert short.count("`[") == 2


async def test_ask_endpoint(db, monkeypatch):
    embedder = await seed()
    monkeypatch.setenv("DISCORD_GUILD_ID", str(GUILD))
    get_settings.cache_clear()
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_answerer] = lambda: FakeAnswerer()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/ask", json={"question": "oauth?", "user_id": BOB})
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["answered"] is True
    assert [s["jump_url"].rsplit("/", 1)[1] for s in body["sources"]] == ["102", "101"]
