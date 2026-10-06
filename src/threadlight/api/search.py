from datetime import datetime
from typing import Annotated

import voyageai.error
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from threadlight.config import get_settings
from threadlight.db.session import get_session
from threadlight.processing.embedder import Embedder, get_embedder
from threadlight.retrieval.hybrid import hybrid_search
from threadlight.retrieval.permissions import visible_channel_ids

router = APIRouter()

SNIPPET_CHARS = 600


class Hit(BaseModel):
    conversation_id: int
    channel: str
    started_at: datetime
    ended_at: datetime
    message_count: int
    score: float
    vector_rank: int | None
    keyword_rank: int | None
    jump_url: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    hits: list[Hit]


@router.get("/search", response_model=SearchResponse)
async def search(
    q: Annotated[str, Query(min_length=1, max_length=500)],
    # Discord user id of the person asking. Results are limited to channels they can read.
    # TRUSTED INPUT: there is no end-user auth yet, so this must only be set by an internal
    # caller that verified the identity (the bot's /ask in M4; Discord OAuth for a web UI).
    user_id: Annotated[int, Query()],
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> SearchResponse:
    guild_id = get_settings().discord_guild_id
    if guild_id is None:
        raise HTTPException(503, "DISCORD_GUILD_ID is not configured")

    visible = await visible_channel_ids(session, guild_id, user_id)

    try:
        results = await hybrid_search(session, embedder, q, visible, limit=limit)
    except voyageai.error.RateLimitError:
        raise HTTPException(503, "Embedding provider rate limit hit; retry shortly") from None
    return SearchResponse(
        query=q,
        hits=[
            Hit(
                conversation_id=h.conversation_id,
                channel=h.channel_name,
                started_at=h.started_at,
                ended_at=h.ended_at,
                message_count=h.message_count,
                score=h.score,
                vector_rank=h.vector_rank,
                keyword_rank=h.keyword_rank,
                jump_url=h.jump_url,
                snippet=h.text[:SNIPPET_CHARS],
            )
            for h in results
        ],
    )
