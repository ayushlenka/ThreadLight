"""Processing worker: drains the job queue.

python -m threadlight.processing.worker              # run forever
python -m threadlight.processing.worker --drain      # exit when the queue is empty
python -m threadlight.processing.worker --reindex    # queue every channel first
"""

import argparse
import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select

from threadlight.db.models import Channel
from threadlight.db.session import SessionLocal, engine
from threadlight.processing import jobs
from threadlight.processing.decisions import (
    DecisionLLM,
    ExtractStats,
    extract_channel,
    get_decision_llm,
)
from threadlight.processing.embedder import Embedder, get_embedder
from threadlight.processing.pipeline import embed_pending, resegment_channel

log = logging.getLogger(__name__)

POLL_INTERVAL = 2.0
EMBED_CHUNK = 64  # conversations per committed transaction
# Slack after a conversation settles before the follow-up extraction runs.
SETTLE_SLACK = timedelta(seconds=30)


@dataclass
class WorkerContext:
    embedder: Embedder
    decision_llm: DecisionLLM | None  # None: decision extraction disabled (no API key)


Handler = Callable[[dict[str, Any], WorkerContext], Awaitable[None]]


async def handle_segment_channel(payload: dict[str, Any], ctx: WorkerContext) -> None:
    channel_id = payload["channel_id"]
    async with SessionLocal.begin() as session:
        result = await resegment_channel(session, channel_id)
    # Separate transactions throughout: if a later step fails, earlier work is saved and
    # the retry picks up where this left off (embedding and extraction are incremental).
    embedded = await embed_channel(ctx.embedder, channel_id)
    extracted = None
    if ctx.decision_llm is not None:
        extracted = await extract_and_schedule(ctx, channel_id)
    log.info(
        "channel %s: %d created, %d kept, %d removed, %d embedded%s",
        channel_id,
        result.created,
        result.kept,
        result.removed,
        embedded,
        ""
        if extracted is None
        else f", {extracted.conversations} extracted -> {extracted.decisions} decisions, "
        f"{extracted.links} supersession links",
    )


async def extract_and_schedule(ctx: WorkerContext, channel_id: int) -> ExtractStats:
    """Extract settled conversations; schedule a follow-up for ones still active."""
    stats = await extract_channel(SessionLocal, ctx.decision_llm, ctx.embedder, channel_id)
    if stats.next_settle_at is not None:
        delay = max(stats.next_settle_at - datetime.now(UTC), timedelta(0)) + SETTLE_SLACK
        async with SessionLocal.begin() as session:
            await jobs.enqueue_extract(session, channel_id, delay)
    return stats


async def handle_extract_channel(payload: dict[str, Any], ctx: WorkerContext) -> None:
    if ctx.decision_llm is None:
        return
    stats = await extract_and_schedule(ctx, payload["channel_id"])
    log.info(
        "channel %s: %d extracted -> %d decisions, %d supersession links",
        payload["channel_id"],
        stats.conversations,
        stats.decisions,
        stats.links,
    )


async def embed_channel(embedder: Embedder, channel_id: int) -> int:
    total = 0
    while True:
        async with SessionLocal.begin() as session:
            n = await embed_pending(session, embedder, channel_id=channel_id, limit=EMBED_CHUNK)
        total += n
        if n < EMBED_CHUNK:
            return total


HANDLERS: dict[str, Handler] = {
    jobs.SEGMENT_CHANNEL: handle_segment_channel,
    jobs.EXTRACT_CHANNEL: handle_extract_channel,
}


async def run_one(ctx: WorkerContext) -> bool:
    """Process one job. Returns False if there was nothing to do."""
    async with SessionLocal.begin() as session:
        job = await jobs.claim(session)
    if job is None:
        return False

    try:
        await HANDLERS[job.kind](job.payload, ctx)
    except Exception as exc:
        log.exception("job %s (%s) failed", job.id, job.kind)
        async with SessionLocal.begin() as session:
            await jobs.fail(session, job, f"{type(exc).__name__}: {exc}")
    else:
        async with SessionLocal.begin() as session:
            await jobs.complete(session, job.id)
    return True


async def reindex_all() -> int:
    async with SessionLocal.begin() as session:
        channel_ids = (
            await session.scalars(select(Channel.id).where(Channel.deleted_at.is_(None)))
        ).all()
        for cid in channel_ids:
            await jobs.enqueue_segment(session, cid)
    return len(channel_ids)


async def main(drain: bool, reindex: bool) -> None:
    try:
        decision_llm = get_decision_llm()
    except RuntimeError as exc:
        log.warning("decision extraction disabled: %s", exc)
        decision_llm = None
    ctx = WorkerContext(embedder=get_embedder(), decision_llm=decision_llm)
    if reindex:
        log.info("queued %d channels for re-segmentation", await reindex_all())
    try:
        while True:
            if not await run_one(ctx):
                if drain:
                    async with SessionLocal() as session:
                        unfinished = await jobs.has_unfinished(session)
                    if not unfinished:
                        log.info("queue drained")
                        return
                await asyncio.sleep(POLL_INTERVAL)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drain", action="store_true", help="exit when the queue is empty")
    parser.add_argument("--reindex", action="store_true", help="queue all channels first")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("voyage").setLevel(logging.WARNING)  # logs every request at INFO
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main(drain=args.drain, reindex=args.reindex))
