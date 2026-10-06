"""Read decisions as a particular viewer sees them.

A decision is "superseded" for a viewer only if the decision that replaced it is in a
channel they can read. Otherwise they see it as still standing: a #leads decision must not
reveal itself by marking a #backend decision as changed. Replacement links pointing at
invisible decisions are dropped the same way.

A decision whose deciding message has been deleted is invisible too, as a result and as a
reference from other decisions, from the moment of deletion (before re-segmentation
removes the decision itself).
"""

from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from threadlight.db.models import Channel, Decision, Message
from threadlight.retrieval.hybrid import keyword_ranking, rrf, vector_ranking

CHAIN_DEPTH = 3


@dataclass
class DecisionRef:
    id: int
    title: str
    decided_at: datetime
    channel_name: str


@dataclass
class DecisionView:
    id: int
    title: str
    summary: str
    status: str  # decided | proposed | superseded (as this viewer sees it)
    decided_at: datetime
    channel_id: int
    channel_name: str
    guild_id: int
    decided_message_id: int
    rationale: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    replaces: DecisionRef | None = None
    replaced_by: DecisionRef | None = None

    @property
    def jump_url(self) -> str:
        return (
            f"https://discord.com/channels/{self.guild_id}/{self.channel_id}/"
            f"{self.decided_message_id}"
        )


async def load_views(
    session: AsyncSession, ids: Collection[int], visible_channel_ids: Collection[int]
) -> dict[int, DecisionView]:
    if not ids or not visible_channel_ids:
        return {}
    visible = list(visible_channel_ids)

    def visible_decisions():
        return (
            select(Decision, Channel.name, Channel.guild_id)
            .join(Channel, Channel.id == Decision.channel_id)
            .join(Message, Message.id == Decision.decided_message_id)
            .where(Decision.channel_id.in_(visible), Message.deleted_at.is_(None))
        )

    rows = (await session.execute(visible_decisions().where(Decision.id.in_(ids)))).all()
    if not rows:
        return {}

    # Visible successors (the latest one, if a decision was replaced more than once) and
    # visible predecessors.
    found = [d.id for d, _, _ in rows]
    successors: dict[int, DecisionRef] = {}
    for d, name, _ in (
        await session.execute(
            visible_decisions()
            .where(Decision.supersedes_id.in_(found))
            .order_by(Decision.decided_at)
        )
    ).all():
        successors[d.supersedes_id] = DecisionRef(d.id, d.title, d.decided_at, name)

    pred_ids = {d.supersedes_id for d, _, _ in rows if d.supersedes_id}
    predecessors = {
        d.id: DecisionRef(d.id, d.title, d.decided_at, name)
        for d, name, _ in (
            await session.execute(visible_decisions().where(Decision.id.in_(pred_ids)))
        ).all()
    }

    views = {}
    for d, name, guild_id in rows:
        replaced_by = successors.get(d.id)
        views[d.id] = DecisionView(
            id=d.id,
            title=d.title,
            summary=d.summary,
            status="superseded" if replaced_by else d.status,
            decided_at=d.decided_at,
            channel_id=d.channel_id,
            channel_name=name,
            guild_id=guild_id,
            decided_message_id=d.decided_message_id,
            rationale=list(d.rationale),
            alternatives=list(d.alternatives),
            replaces=predecessors.get(d.supersedes_id) if d.supersedes_id else None,
            replaced_by=replaced_by,
        )
    return views


async def search_decisions(
    session: AsyncSession,
    query: str,
    query_vec: list[float],
    visible_channel_ids: Collection[int],
    limit: int = 5,
) -> list[DecisionView]:
    """Best-matching decisions plus their visible replacement chains, oldest first.

    Chains matter for "what did we decide about X": the match might be the original
    decision, and the answer is whatever later replaced it.
    """
    if not visible_channel_ids or not query.strip():
        return []
    vec = await vector_ranking(session, query_vec, visible_channel_ids, limit=20, model=Decision)
    kw = await keyword_ranking(session, query, visible_channel_ids, limit=20, model=Decision)
    scores = rrf([vec, kw])
    top = sorted(scores, key=lambda i: (-scores[i], i))[:limit]

    views = await load_views(session, top, visible_channel_ids)
    frontier = set(views)
    for _ in range(CHAIN_DEPTH):
        linked = {
            ref.id
            for v in (views[i] for i in frontier)
            for ref in (v.replaces, v.replaced_by)
            if ref is not None and ref.id not in views
        }
        if not linked:
            break
        new = await load_views(session, linked, visible_channel_ids)
        views.update(new)
        frontier = set(new)
    return sorted(views.values(), key=lambda v: v.decided_at)


async def list_decisions(
    session: AsyncSession, visible_channel_ids: Collection[int], limit: int = 50
) -> list[DecisionView]:
    """The viewer's decision log, newest first."""
    if not visible_channel_ids:
        return []
    ids = list(
        await session.scalars(
            select(Decision.id)
            .where(Decision.channel_id.in_(list(visible_channel_ids)))
            .order_by(Decision.decided_at.desc())
            .limit(limit)
        )
    )
    views = await load_views(session, ids, visible_channel_ids)
    return [views[i] for i in ids if i in views]
