import re
from types import SimpleNamespace

from sqlalchemy import select, update

from tests.fakes import HashingEmbedder
from threadlight.answer.rag import DECISION_LOG_TITLE, ask
from threadlight.db.models import Conversation, Decision
from threadlight.db.session import SessionLocal
from threadlight.evals.corpus import install, parse, segmentation_mismatches
from threadlight.evals.extraction import score
from threadlight.ingest import store
from threadlight.processing.decisions import (
    ExtractedDecision,
    ExtractionResult,
    Replacement,
    SupersessionResult,
    extract_channel,
    extract_conversation,
    link_supersessions,
)
from threadlight.processing.pipeline import embed_pending, resegment_channel
from threadlight.retrieval.decisions import list_decisions, search_decisions
from threadlight.retrieval.permissions import visible_channel_ids

CORPUS = parse()


class OracleLLM:
    """Answers from the corpus labels: a perfect extractor and supersession judge."""

    def __init__(self, extra=None, drop=()):
        self.extra = extra or {}  # conversation key -> extra (spurious) decisions
        self.drop = set(drop)  # truth keys to "miss"
        self.transcripts = []

    def _conversation(self, transcript):
        return next(c for c in CORPUS.conversations if c.messages[0].content in transcript)

    async def extract(self, transcript):
        self.transcripts.append(transcript)
        conv = self._conversation(transcript)
        ids = [m.id for m in conv.messages]
        decisions = [
            ExtractedDecision(
                title=d.title,
                summary=d.title,
                status=d.status,
                rationale=["because"],
                alternatives=[],
                decided_message=ids.index(d.message_id) + 1,
                source_messages=[1, ids.index(d.message_id) + 1],
            )
            for d in conv.decisions
            if d.key not in self.drop
        ]
        return ExtractionResult(decisions=decisions + self.extra.get(conv.key, []))

    async def find_replacements(self, listing):
        titles = {
            int(n): t for n, t in re.findall(r"^(\d+)\. \[[^\]]+\] #\S+: (.+?)\. ", listing, re.M)
        }
        by_title = {d.title: d for d in CORPUS.decisions}
        number = {by_title[t].key: n for n, t in titles.items() if t in by_title}
        return SupersessionResult(
            replacements=[
                Replacement(newer=number[d.key], older=number[d.supersedes])
                for d in CORPUS.decisions
                if d.supersedes and d.key in number and d.supersedes in number
            ]
        )


async def load_corpus():
    embedder = HashingEmbedder()
    async with SessionLocal.begin() as s:
        await install(s, CORPUS)
        for cid in CORPUS.channel_ids.values():
            await resegment_channel(s, cid)
    async with SessionLocal.begin() as s:
        await embed_pending(s, embedder)
    return embedder


async def extract_all(llm, embedder, newest_first=False):
    """Run extraction conversation by conversation, optionally in reverse time order."""
    async with SessionLocal() as s:
        conv_ids = list(
            await s.scalars(
                select(Conversation.id)
                .where(Conversation.channel_id.in_(CORPUS.channel_ids.values()))
                .order_by(
                    Conversation.started_at.desc() if newest_first else Conversation.started_at
                )
            )
        )
    for conv_id in conv_ids:
        async with SessionLocal.begin() as s:
            conv = await s.get(Conversation, conv_id)
            for d in await extract_conversation(s, llm, embedder, conv):
                await link_supersessions(s, llm, d)


def member(name):
    return CORPUS.member_ids[name]


async def visible_for(name):
    async with SessionLocal() as s:
        return await visible_channel_ids(s, CORPUS.guild_id, member(name))


async def decision_id(title):
    async with SessionLocal() as s:
        return await s.scalar(select(Decision.id).where(Decision.title == title))


# Corpus


async def test_segmenter_reproduces_corpus_conversations(db):
    await load_corpus()
    async with SessionLocal() as s:
        assert await segmentation_mismatches(s, CORPUS) == []


async def test_private_channel_visibility(db):
    await load_corpus()
    leads = CORPUS.channel_ids["leads"]
    assert leads in await visible_for("maya")
    assert leads not in await visible_for("sam")


# Extraction and scoring


async def test_perfect_extractor_scores_perfectly_in_any_order(db):
    embedder = await load_corpus()
    await extract_all(OracleLLM(), embedder, newest_first=True)

    async with SessionLocal() as s:
        report = await score(s, CORPUS)
    assert (report.precision, report.recall, report.status_accuracy) == (1.0, 1.0, 1.0)
    assert (report.link_precision, report.link_recall) == (1.0, 1.0)
    assert report.links_correct == 3


async def test_scorer_counts_misses_and_spurious_decisions(db):
    embedder = await load_corpus()
    spurious = ExtractedDecision(
        title="Rewrite the backend in Rust",
        summary="joke",
        status="decided",
        rationale=[],
        alternatives=[],
        decided_message=1,
        source_messages=[1],
    )
    deferral = spurious.model_copy(update={"title": "Defer dark mode"})
    llm = OracleLLM(extra={"rust-joke": [spurious], "dark-mode": [deferral]}, drop={"api-rest"})
    await extract_all(llm, embedder)

    async with SessionLocal() as s:
        report = await score(s, CORPUS)
    assert report.false_negatives == ["api-rest: Keep the API REST instead of GraphQL"]
    # The dark-mode conversation is labeled ambiguous, so it doesn't count either way.
    assert report.false_positives == ["[rust-joke] Rewrite the backend in Rust (decided)"]


async def test_extraction_maps_message_numbers_and_marks_conversation(db):
    embedder = await load_corpus()
    llm = OracleLLM()
    await extract_all(llm, embedder)

    db_choice = next(t for t in llm.transcripts if "postgres or mongo" in t)
    assert db_choice.startswith("Channel: #backend\n\n1. [2025-01-08 19:00] priya:")
    async with SessionLocal() as s:
        d = await s.scalar(
            select(Decision).where(Decision.title == "Use PostgreSQL as the database")
        )
        truth = next(t for t in CORPUS.decisions if t.key == "db-postgres")
        assert d.decided_message_id == truth.message_id
        assert d.embedding is not None
        pending = await s.scalar(
            select(Conversation.id).where(
                Conversation.channel_id.in_(CORPUS.channel_ids.values()),
                Conversation.extracted_at.is_(None),
            )
        )
        assert pending is None


async def test_out_of_range_message_numbers_are_dropped(db):
    embedder = await load_corpus()

    class Broken(OracleLLM):
        async def extract(self, transcript):
            bad = ExtractedDecision(
                title="x",
                summary="x",
                status="decided",
                rationale=[],
                alternatives=[],
                decided_message=999,
                source_messages=[999],
            )
            return ExtractionResult(decisions=[bad])

    stats = await extract_channel(SessionLocal, Broken(), embedder, CORPUS.channel_ids["general"])
    assert stats.decisions == 0 and stats.conversations > 0


async def test_supersession_never_links_backwards_or_unrelated_pairs(db):
    embedder = await load_corpus()
    await extract_all(OracleLLM(), embedder)
    async with SessionLocal.begin() as s:
        await s.execute(update(Decision).values(supersedes_id=None))

    jwt, oauth = await decision_id("Use email/password auth with JWTs, no OAuth for now"), None
    oauth = await decision_id("Migrate auth to Google OAuth and drop email/password")

    class Backwards(OracleLLM):
        async def find_replacements(self, listing):
            # #1 is the JWT decision and #2 its nearest neighbor, the later OAuth decision.
            # Claims JWT replaced OAuth (backwards in time), plus a pair without #1.
            return SupersessionResult(
                replacements=[Replacement(newer=1, older=2), Replacement(newer=3, older=2)]
            )

    async with SessionLocal.begin() as s:
        jwt_decision = await s.get(Decision, jwt)
        await link_supersessions(s, Backwards(), jwt_decision)
    async with SessionLocal() as s:
        assert (await s.get(Decision, oauth)).supersedes_id is None
        assert (await s.get(Decision, jwt)).supersedes_id is None


async def test_resegmenting_a_conversation_drops_its_decisions(db):
    embedder = await load_corpus()
    await extract_all(OracleLLM(), embedder)
    truth = next(t for t in CORPUS.decisions if t.key == "pr-policy")

    async with SessionLocal.begin() as s:
        await store.mark_messages_deleted(s, [truth.message_id])
        await resegment_channel(s, CORPUS.channel_ids["general"])
    assert await decision_id(truth.title) is None


# Per-viewer reads


async def test_superseded_status_and_chains(db):
    embedder = await load_corpus()
    await extract_all(OracleLLM(), embedder)

    async with SessionLocal() as s:
        visible = await visible_channel_ids(s, CORPUS.guild_id, member("sam"))
        vec = await embedder.embed_query("auth oauth jwt")
        views = await search_decisions(s, "auth oauth jwt", vec, visible)
    by_title = {v.title: v for v in views}
    jwt = by_title["Use email/password auth with JWTs, no OAuth for now"]
    oauth = by_title["Migrate auth to Google OAuth and drop email/password"]
    assert jwt.status == "superseded" and jwt.replaced_by.id == oauth.id
    assert oauth.status == "decided" and oauth.replaces.id == jwt.id
    assert views.index(jwt) < views.index(oauth)  # oldest first


async def test_hidden_successor_does_not_mark_decision_superseded(db):
    embedder = await load_corpus()
    await extract_all(OracleLLM(), embedder)
    postgres = await decision_id("Use PostgreSQL as the database")
    cut = await decision_id("Cut live delivery tracking from the MVP")  # in private #leads
    async with SessionLocal.begin() as s:
        await s.execute(update(Decision).where(Decision.id == cut).values(supersedes_id=postgres))

    async with SessionLocal() as s:
        sam = {v.id: v for v in await list_decisions(s, await visible_for("sam"), limit=100)}
        maya = {v.id: v for v in await list_decisions(s, await visible_for("maya"), limit=100)}
    assert cut not in sam
    assert sam[postgres].status == "decided" and sam[postgres].replaced_by is None
    assert maya[postgres].status == "superseded" and maya[postgres].replaced_by.id == cut


# Decision log in /ask


class CapturingAnswerer:
    def __init__(self, cite_block_containing=None):
        self.documents = None
        self.cite_text = cite_block_containing

    async def generate(self, documents, question, now):
        self.documents = documents
        citations = []
        if self.cite_text is not None:
            blocks = [b["text"] for b in documents[0]["source"]["content"]]
            self.cite = next(i for i, b in enumerate(blocks) if self.cite_text in b)
            citations = [
                SimpleNamespace(
                    type="content_block_location",
                    document_index=0,
                    start_block_index=self.cite,
                    end_block_index=self.cite + 1,
                    cited_text="...",
                )
            ]
        return SimpleNamespace(
            model="fake",
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="Answer.", citations=citations)],
        )


async def test_ask_includes_decision_log_with_chain(db):
    embedder = await load_corpus()
    await extract_all(OracleLLM(), embedder)

    answerer = CapturingAnswerer(cite_block_containing="DECIDED: Migrate auth to Google OAuth")
    async with SessionLocal() as s:
        answer = await ask(
            s,
            embedder,
            answerer,
            CORPUS.guild_id,
            member("sam"),
            "what did we decide about auth oauth jwt?",
        )

    log = answerer.documents[0]
    assert log["title"] == DECISION_LOG_TITLE
    blocks = [b["text"] for b in log["source"]["content"]]
    jwt_block = next(b for b in blocks if "JWTs" in b)
    assert "SUPERSEDED" in jwt_block and "Later replaced by" in jwt_block
    oauth_truth = next(t for t in CORPUS.decisions if t.key == "auth-oauth")
    # Citing the OAuth decision's block jumps to the message where it was settled.
    assert answer.sources[0].jump_url.endswith(f"/{oauth_truth.message_id}")


async def test_decision_with_deleted_deciding_message_is_left_out(db):
    embedder = await load_corpus()
    await extract_all(OracleLLM(), embedder)
    truth = next(t for t in CORPUS.decisions if t.key == "auth-oauth")
    async with SessionLocal.begin() as s:
        await store.mark_messages_deleted(s, [truth.message_id])  # no resegment yet

    answerer = CapturingAnswerer()
    async with SessionLocal() as s:
        await ask(s, embedder, answerer, CORPUS.guild_id, member("sam"), "auth oauth decision")
    assert "Google OAuth and drop" not in str(answerer.documents)
