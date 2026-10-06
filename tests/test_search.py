from datetime import UTC, datetime, timedelta

from httpx import ASGITransport, AsyncClient

from tests.fakes import HashingEmbedder
from threadlight.api.main import app
from threadlight.config import get_settings
from threadlight.db.session import SessionLocal
from threadlight.ingest import store
from threadlight.processing.embedder import get_embedder
from threadlight.processing.pipeline import embed_pending, resegment_channel
from threadlight.retrieval.hybrid import hybrid_search, rrf
from threadlight.retrieval.permissions import READ, VIEW_CHANNEL

GUILD_ID = 1
ENG, RANDOM, SECRET = 10, 11, 12
ALICE = 900
T0 = datetime(2024, 2, 13, 14, 0, tzinfo=UTC)


def test_rrf_rewards_agreement_between_lists():
    scores = rrf([[1, 2, 3], [3, 1, 4]])
    assert max(scores, key=scores.get) == 1  # ranks 1 and 2
    assert scores[3] > scores[2]  # in both lists beats one list
    assert scores[4] < scores[2]


async def seed_corpus():
    embedder = HashingEmbedder()
    convs = {
        ENG: ["we should migrate authentication to google oauth", "the deploy pipeline is broken"],
        RANDOM: ["anyone want pizza for lunch"],
        SECRET: ["oauth migration budget is confidential"],
    }
    async with SessionLocal.begin() as s:
        await store.upsert_guild(s, GUILD_ID, "g")
        await store.upsert_channels(
            s,
            [
                {"id": cid, "guild_id": GUILD_ID, "parent_id": None, "type": "text", "name": name}
                for cid, name in [(ENG, "eng"), (RANDOM, "random"), (SECRET, "leadership")]
            ],
        )
        # Permission state: everyone can read, except #leadership, which is hidden.
        await store.replace_roles(
            s,
            GUILD_ID,
            [{"id": GUILD_ID, "guild_id": GUILD_ID, "name": "@everyone", "permissions": READ}],
        )
        await store.upsert_members(
            s, [{"guild_id": GUILD_ID, "user_id": ALICE, "display_name": "alice", "role_ids": []}]
        )
        await store.replace_overwrites(
            s,
            SECRET,
            [
                {
                    "channel_id": SECRET,
                    "target_id": GUILD_ID,
                    "target_type": "role",
                    "allow": 0,
                    "deny": VIEW_CHANNEL,
                }
            ],
        )
        msg_id = 100
        for cid, texts in convs.items():
            for i, content in enumerate(texts):
                await store.upsert_messages(
                    s,
                    [
                        {
                            "id": msg_id,
                            "channel_id": cid,
                            "author_id": 7,
                            "author_name": "alice",
                            "content": content,
                            "reply_to_id": None,
                            "attachments": [],
                            "created_at": T0 + timedelta(days=i),
                            "edited_at": None,
                        }
                    ],
                )
                msg_id += 1
        for cid in convs:
            await resegment_channel(s, cid)
    async with SessionLocal.begin() as s:
        await embed_pending(s, embedder)
    return embedder


async def test_hybrid_search_finds_relevant_conversation(db):
    embedder = await seed_corpus()
    async with SessionLocal.begin() as s:
        hits = await hybrid_search(s, embedder, "what did we decide about oauth?", [ENG, RANDOM])

    assert hits[0].channel_name == "eng"
    assert "oauth" in hits[0].text
    assert hits[0].keyword_rank == 1
    assert hits[0].jump_url.startswith(f"https://discord.com/channels/{GUILD_ID}/{ENG}/")


async def test_hybrid_search_never_returns_invisible_channels(db):
    embedder = await seed_corpus()
    async with SessionLocal.begin() as s:
        hits = await hybrid_search(s, embedder, "oauth migration budget", [ENG, RANDOM])
    assert hits, "expected visible matches"
    assert all(h.channel_id != SECRET for h in hits)


async def test_hybrid_search_with_no_visible_channels_returns_nothing(db):
    embedder = await seed_corpus()
    async with SessionLocal.begin() as s:
        assert await hybrid_search(s, embedder, "oauth", []) == []


async def test_keyword_only_match_via_or_semantics(db):
    """Only one of the query's terms appears; AND semantics would find nothing."""
    embedder = await seed_corpus()
    async with SessionLocal.begin() as s:
        hits = await hybrid_search(s, embedder, "is the deploy pipeline fixed yet", [ENG])
    assert "deploy pipeline" in hits[0].text


async def search_as(embedder, monkeypatch, **params):
    monkeypatch.setenv("DISCORD_GUILD_ID", str(GUILD_ID))
    get_settings.cache_clear()
    app.dependency_overrides[get_embedder] = lambda: embedder
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            return await c.get("/search", params=params)
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()


async def test_search_endpoint(db, monkeypatch):
    embedder = await seed_corpus()
    resp = await search_as(embedder, monkeypatch, q="pizza lunch", limit=3, user_id=ALICE)

    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"][0]["channel"] == "random"
    assert "pizza" in body["hits"][0]["snippet"]


async def test_search_endpoint_enforces_channel_permissions(db, monkeypatch):
    embedder = await seed_corpus()
    resp = await search_as(embedder, monkeypatch, q="oauth migration budget", user_id=ALICE)
    channels = {h["channel"] for h in resp.json()["hits"]}
    assert "eng" in channels
    assert "leadership" not in channels


async def test_search_endpoint_returns_nothing_for_non_members(db, monkeypatch):
    embedder = await seed_corpus()
    resp = await search_as(embedder, monkeypatch, q="oauth", user_id=12345)
    assert resp.status_code == 200
    assert resp.json()["hits"] == []


async def test_search_endpoint_requires_user_id(db, monkeypatch):
    embedder = await seed_corpus()
    resp = await search_as(embedder, monkeypatch, q="oauth")
    assert resp.status_code == 422
