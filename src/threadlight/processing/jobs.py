"""Postgres-backed job queue.

Workers claim jobs with SELECT ... FOR UPDATE SKIP LOCKED, so several workers can run
concurrently without double-processing. A job left 'running' by a crashed worker is
reclaimed after STALE_AFTER.
"""

from datetime import timedelta
from typing import Any

from sqlalchemy import func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from threadlight.db.models import Job

SEGMENT_CHANNEL = "segment_channel"
EXTRACT_CHANNEL = "extract_channel"

MAX_ATTEMPTS = 5
STALE_AFTER = timedelta(minutes=10)


async def enqueue(
    session: AsyncSession,
    kind: str,
    payload: dict[str, Any],
    *,
    dedupe_key: str | None = None,
    delay: timedelta = timedelta(0),
) -> None:
    """Add a job. If a pending job with the same (kind, dedupe_key) exists, do nothing."""
    stmt = insert(Job).values(
        kind=kind,
        payload=payload,
        dedupe_key=dedupe_key,
        run_after=func.now() + delay,
    )
    await session.execute(
        stmt.on_conflict_do_nothing(
            index_elements=[Job.kind, Job.dedupe_key],
            index_where=text("status = 'pending'"),
        )
    )


async def enqueue_segment(
    session: AsyncSession, channel_id: int, delay: timedelta = timedelta(0)
) -> None:
    await enqueue(
        session,
        SEGMENT_CHANNEL,
        {"channel_id": channel_id},
        dedupe_key=str(channel_id),
        delay=delay,
    )


async def enqueue_extract(session: AsyncSession, channel_id: int, delay: timedelta) -> None:
    await enqueue(
        session,
        EXTRACT_CHANNEL,
        {"channel_id": channel_id},
        dedupe_key=str(channel_id),
        delay=delay,
    )


async def claim(session: AsyncSession) -> Job | None:
    """Claim the next runnable job, marking it running. Commit to release the row lock."""
    next_id = (
        select(Job.id)
        .where(
            or_(
                (Job.status == "pending") & (Job.run_after <= func.now()),
                (Job.status == "running") & (Job.updated_at < func.now() - STALE_AFTER),
            )
        )
        .order_by(Job.run_after)
        .limit(1)
        .with_for_update(skip_locked=True)
        .scalar_subquery()
    )
    return await session.scalar(
        update(Job)
        .where(Job.id == next_id)
        .values(status="running", attempts=Job.attempts + 1, updated_at=func.now())
        .returning(Job)
    )


async def has_unfinished(session: AsyncSession) -> bool:
    """Any job pending (including ones scheduled for later) or running."""
    return bool(
        await session.scalar(select(Job.id).where(Job.status.in_(("pending", "running"))).limit(1))
    )


async def complete(session: AsyncSession, job_id: int) -> None:
    await session.execute(update(Job).where(Job.id == job_id).values(status="done"))


async def fail(session: AsyncSession, job: Job, error: str) -> None:
    """Retry with exponential backoff, or give up after MAX_ATTEMPTS.

    If a newer pending job with the same dedupe key exists (more messages arrived while this
    one ran), it covers the same work, so this one is retired rather than re-queued.
    """
    superseded = job.dedupe_key is not None and await session.scalar(
        select(Job.id).where(
            Job.kind == job.kind,
            Job.dedupe_key == job.dedupe_key,
            Job.status == "pending",
            Job.id != job.id,
        )
    )
    if superseded:
        values = {"status": "failed", "last_error": f"{error} (superseded by job {superseded})"}
    elif job.attempts >= MAX_ATTEMPTS:
        values = {"status": "failed", "last_error": error}
    else:
        backoff = timedelta(seconds=30 * 2 ** (job.attempts - 1))
        values = {"status": "pending", "last_error": error, "run_after": func.now() + backoff}
    await session.execute(update(Job).where(Job.id == job.id).values(**values))
