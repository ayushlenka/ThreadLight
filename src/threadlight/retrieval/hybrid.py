"""Hybrid retrieval over conversations: vector similarity + full-text, fused with RRF.

Vector search finds paraphrases ("login" ~ "auth"); keyword search finds exact names,
error strings, and jargon that embeddings blur. Reciprocal Rank Fusion combines the two
ranked lists without having to calibrate their very different score scales.

Every query takes `visible_channel_ids` and filters inside SQL, so conversations from
channels the asker can't see are never retrieved at all.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Text, func, literal_column, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from threadlight.db.models import Channel, Conversation
from threadlight.processing.embedder import Embedder

RRF_K = 60  # Standard constant from Cormack et al.; damps the influence of top ranks.
CANDIDATES = 50  # Depth of each ranked list before fusion.


@dataclass
class SearchHit:
    conversation_id: int
    channel_id: int
    channel_name: str
    guild_id: int
    started_at: datetime
    ended_at: datetime
    first_message_id: int
    message_count: int
    text: str
    score: float
    vector_rank: int | None
    keyword_rank: int | None

    @property
    def jump_url(self) -> str:
        return (
            f"https://discord.com/channels/{self.guild_id}/{self.channel_id}/"
            f"{self.first_message_id}"
        )


def rrf(ranked_lists: Sequence[Sequence[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank(d))."""
    scores: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def _or_tsquery(query: str):
    """Match any query term instead of all of them.

    plainto_tsquery ANDs terms, which is too strict for natural-language questions
    ("what did we decide about oauth" would require 'decid' AND 'oauth'). OR-ing them and
    ranking with ts_rank_cd rewards conversations that match more terms.
    """
    english = literal_column("'english'::regconfig")
    terms = func.plainto_tsquery(english, query).cast(Text)
    return func.to_tsquery(english, func.replace(terms, "&", "|"))


# Rankings work on any model with id, channel_id, embedding, and tsv columns
# (Conversation, Decision).


async def vector_ranking(
    session: AsyncSession,
    query_vec: list[float],
    visible_channel_ids: Collection[int],
    limit: int = CANDIDATES,
    model=Conversation,
) -> list[int]:
    # With a WHERE filter, a plain HNSW scan can return fewer than `limit` rows (it filters
    # after the index walk). Iterative scans (pgvector >= 0.8) keep walking until filled.
    await session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
    rows = await session.scalars(
        select(model.id)
        .where(model.channel_id.in_(visible_channel_ids), model.embedding.is_not(None))
        .order_by(model.embedding.cosine_distance(query_vec))
        .limit(limit)
    )
    return list(rows)


async def keyword_ranking(
    session: AsyncSession,
    query: str,
    visible_channel_ids: Collection[int],
    limit: int = CANDIDATES,
    model=Conversation,
) -> list[int]:
    tsq = _or_tsquery(query)
    rows = await session.scalars(
        select(model.id)
        .where(model.channel_id.in_(visible_channel_ids), model.tsv.op("@@")(tsq))
        .order_by(func.ts_rank_cd(model.tsv, tsq).desc(), model.id)
        .limit(limit)
    )
    return list(rows)


async def hybrid_search(
    session: AsyncSession,
    embedder: Embedder,
    query: str,
    visible_channel_ids: Collection[int],
    limit: int = 10,
    query_vec: list[float] | None = None,
) -> list[SearchHit]:
    if not visible_channel_ids or not query.strip():
        return []

    if query_vec is None:
        query_vec = await embedder.embed_query(query)
    vec_ids = await vector_ranking(session, query_vec, visible_channel_ids)
    kw_ids = await keyword_ranking(session, query, visible_channel_ids)

    scores = rrf([vec_ids, kw_ids])
    top = sorted(scores, key=lambda cid: (-scores[cid], cid))[:limit]
    if not top:
        return []

    vec_rank = {cid: i for i, cid in enumerate(vec_ids, start=1)}
    kw_rank = {cid: i for i, cid in enumerate(kw_ids, start=1)}
    rows = await session.execute(
        select(Conversation, Channel.name, Channel.guild_id)
        .join(Channel, Channel.id == Conversation.channel_id)
        .where(Conversation.id.in_(top))
    )
    by_id = {conv.id: (conv, name, guild_id) for conv, name, guild_id in rows}

    hits = []
    for cid in top:
        conv, name, guild_id = by_id[cid]
        hits.append(
            SearchHit(
                conversation_id=cid,
                channel_id=conv.channel_id,
                channel_name=name,
                guild_id=guild_id,
                started_at=conv.started_at,
                ended_at=conv.ended_at,
                first_message_id=conv.first_message_id,
                message_count=conv.message_count,
                text=conv.text,
                score=scores[cid],
                vector_rank=vec_rank.get(cid),
                keyword_rank=kw_rank.get(cid),
            )
        )
    return hits
