from datetime import datetime
from typing import Annotated

import voyageai.error
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from threadlight.config import get_settings
from threadlight.db.session import get_session
from threadlight.processing.embedder import Embedder, get_embedder
from threadlight.retrieval.decisions import DecisionRef, list_decisions, search_decisions
from threadlight.retrieval.permissions import visible_channel_ids

router = APIRouter()


class RefOut(BaseModel):
    id: int
    title: str
    decided_at: datetime
    channel: str


class DecisionOut(BaseModel):
    id: int
    title: str
    summary: str
    status: str
    decided_at: datetime
    channel: str
    jump_url: str
    rationale: list[str]
    alternatives: list[str]
    replaces: RefOut | None
    replaced_by: RefOut | None


def _ref(ref: DecisionRef | None) -> RefOut | None:
    if ref is None:
        return None
    return RefOut(id=ref.id, title=ref.title, decided_at=ref.decided_at, channel=ref.channel_name)


@router.get("/decisions", response_model=list[DecisionOut])
async def decisions_endpoint(
    # TRUSTED INPUT, same as /search.
    user_id: Annotated[int, Query()],
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    q: Annotated[str | None, Query(max_length=500)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[DecisionOut]:
    """The asker's decision log (newest first), or decisions matching `q` with their chains."""
    guild_id = get_settings().discord_guild_id
    if guild_id is None:
        raise HTTPException(503, "DISCORD_GUILD_ID is not configured")
    visible = await visible_channel_ids(session, guild_id, user_id)
    if q:
        try:
            query_vec = await embedder.embed_query(q)
        except voyageai.error.RateLimitError:
            raise HTTPException(503, "Embedding provider rate limit hit; retry shortly") from None
        views = await search_decisions(session, q, query_vec, visible, limit=min(limit, 20))
    else:
        views = await list_decisions(session, visible, limit=limit)
    return [
        DecisionOut(
            id=v.id,
            title=v.title,
            summary=v.summary,
            status=v.status,
            decided_at=v.decided_at,
            channel=v.channel_name,
            jump_url=v.jump_url,
            rationale=v.rationale,
            alternatives=v.alternatives,
            replaces=_ref(v.replaces),
            replaced_by=_ref(v.replaced_by),
        )
        for v in views
    ]
