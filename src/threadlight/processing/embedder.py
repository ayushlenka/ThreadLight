import asyncio
import logging
from typing import Protocol

import voyageai
import voyageai.error

from threadlight.config import get_settings
from threadlight.db.models import EMBEDDING_DIM
from threadlight.usage import record_usage

log = logging.getLogger(__name__)

VOYAGE_MODEL = "voyage-3.5"
# Keep each request under Voyage's free-tier limit (10K tokens/minute without a payment
# method). ~4 chars per token puts a full batch around 6K tokens.
MAX_BATCH_CHARS = 24_000
MAX_BATCH_TEXTS = 128

# Free-tier limits are per minute, so back off on the order of a minute. Background
# document embedding can afford to wait; interactive queries should fail fast.
RATE_LIMIT_BACKOFF_SECONDS = 20
DOCUMENT_RETRIES = 6
QUERY_RETRIES = 1


class Embedder(Protocol):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class VoyageEmbedder:
    def __init__(self, api_key: str) -> None:
        self._client = voyageai.AsyncClient(api_key=api_key)

    async def _embed(self, texts: list[str], input_type: str, retries: int) -> list[list[float]]:
        for attempt in range(retries + 1):
            try:
                result = await self._client.embed(
                    texts,
                    model=VOYAGE_MODEL,
                    input_type=input_type,
                    output_dimension=EMBEDDING_DIM,
                )
                await record_usage(
                    provider="voyage",
                    model=VOYAGE_MODEL,
                    purpose=f"embed_{input_type}",
                    input_tokens=result.total_tokens,
                )
                return result.embeddings
            except voyageai.error.RateLimitError:
                if attempt == retries:
                    raise
                wait = RATE_LIMIT_BACKOFF_SECONDS * (attempt + 1)
                log.warning("Voyage rate limit hit; retrying in %ds", wait)
                await asyncio.sleep(wait)
        raise AssertionError("unreachable")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for batch in _batches(texts):
            out.extend(await self._embed(batch, "document", DOCUMENT_RETRIES))
        return out

    async def embed_query(self, text: str) -> list[float]:
        [vec] = await self._embed([text], "query", QUERY_RETRIES)
        return vec


def _batches(texts: list[str]):
    batch: list[str] = []
    chars = 0
    for t in texts:
        if batch and (chars + len(t) > MAX_BATCH_CHARS or len(batch) >= MAX_BATCH_TEXTS):
            yield batch
            batch, chars = [], 0
        batch.append(t)
        chars += len(t)
    if batch:
        yield batch


def get_embedder() -> Embedder:
    key = get_settings().voyage_api_key
    if not key:
        raise RuntimeError("VOYAGE_API_KEY is not set")
    return VoyageEmbedder(key)
