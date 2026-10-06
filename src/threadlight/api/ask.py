from datetime import datetime
from typing import Annotated

import anthropic
import voyageai.error
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from threadlight.answer.rag import Answerer, ask, get_answerer
from threadlight.config import get_settings
from threadlight.db.session import get_session
from threadlight.processing.embedder import Embedder, get_embedder

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    # TRUSTED INPUT, same as /search: only internal callers that verified the identity.
    user_id: int


class SourceOut(BaseModel):
    number: int
    channel: str
    author: str
    created_at: datetime
    jump_url: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    answered: bool
    model: str | None
    sources: list[SourceOut]


@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(
    body: AskRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    answerer: Annotated[Answerer, Depends(get_answerer)],
) -> AskResponse:
    guild_id = get_settings().discord_guild_id
    if guild_id is None:
        raise HTTPException(503, "DISCORD_GUILD_ID is not configured")
    try:
        answer = await ask(session, embedder, answerer, guild_id, body.user_id, body.question)
    except (voyageai.error.RateLimitError, anthropic.RateLimitError):
        raise HTTPException(503, "Rate limited by a model provider; retry shortly") from None
    return AskResponse(
        answer=answer.text,
        answered=answer.answered,
        model=answer.model,
        sources=[
            SourceOut(
                number=s.number,
                channel=s.channel_name,
                author=s.author_name,
                created_at=s.created_at,
                jump_url=s.jump_url,
                excerpt=s.excerpt,
            )
            for s in answer.sources
        ],
    )
