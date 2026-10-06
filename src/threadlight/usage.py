"""Record tokens and cost for every paid API call, and report on them.

    python -m threadlight.usage            # totals for the last 30 days
    python -m threadlight.usage --days 7

Each call is written in its own short transaction, so a call that was paid for is logged
even if the work around it later fails and rolls back. Recording never raises: losing a
usage row is better than failing a user's request.

Callers attach context (guild, user, conversation) with `usage_tags(...)`; it flows to
every call made inside that block via a context variable, so API wrappers don't need
extra parameters.
"""

import argparse
import asyncio
import contextvars
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select

from threadlight.db.models import ApiUsage

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Price:
    """USD per million tokens (list prices; update when pricing changes)."""

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(4.00, 20.00, cache_read=0.20, cache_write=5.00),
    "claude-opus-5": Price(5.00, 25.00, cache_read=0.50, cache_write=6.25),
    "claude-sonnet-5-5": Price(2.00, 10.00, cache_read=0.20, cache_write=2.50),
    "claude-haiku-4-5": Price(1.00, 5.00, cache_read=0.10, cache_write=1.25),
    # Embeddings are input-only. Voyage's free allowance isn't subtracted here: this is
    # what the traffic would cost at list price.
    "voyage-3.5": Price(0.06, 0.0),
}

_tags: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "usage_tags", default=None
)


@contextmanager
def usage_tags(**tags: Any) -> Iterator[None]:
    """Attach tags (guild_id, user_id, conversation_id, ...) to calls made in this block."""
    token = _tags.set({**(_tags.get() or {}), **tags})
    try:
        yield
    finally:
        _tags.reset(token)


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Decimal | None:
    price = PRICES.get(model)
    if price is None:
        return None
    total = (
        input_tokens * price.input
        + output_tokens * price.output
        + cache_read_tokens * price.cache_read
        + cache_write_tokens * price.cache_write
    ) / 1_000_000
    return Decimal(str(round(total, 6)))


async def record_usage(
    *,
    provider: str,
    model: str,
    purpose: str,
    input_tokens: int,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> None:
    from threadlight.db.session import SessionLocal  # avoid import cycle at module load

    cost = cost_usd(model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)
    if cost is None:
        log.warning("no price for model %r; recording usage without cost", model)
    try:
        async with SessionLocal.begin() as session:
            session.add(
                ApiUsage(
                    provider=provider,
                    model=model,
                    purpose=purpose,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    cost_usd=cost,
                    tags=_tags.get() or {},
                )
            )
    except Exception:
        log.exception("failed to record %s usage", purpose)


async def record_claude_usage(response: Any, purpose: str) -> None:
    """Record a Messages API response's usage. Thinking tokens are part of output_tokens."""
    usage = response.usage
    await record_usage(
        provider="anthropic",
        # On a server-side fallback this is the model that produced the final answer; any
        # declined attempt before it isn't broken out here.
        model=response.model,
        purpose=purpose,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
    )


@dataclass
class UsageLine:
    purpose: str
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal

    @property
    def cost_per_call(self) -> Decimal:
        return self.cost_usd / self.calls if self.calls else Decimal(0)


async def summarize(session, since: datetime) -> list[UsageLine]:
    rows = await session.execute(
        select(
            ApiUsage.purpose,
            ApiUsage.model,
            func.count(),
            func.sum(ApiUsage.input_tokens),
            func.sum(ApiUsage.output_tokens),
            func.coalesce(func.sum(ApiUsage.cost_usd), 0),
        )
        .where(ApiUsage.created_at >= since)
        .group_by(ApiUsage.purpose, ApiUsage.model)
        .order_by(func.coalesce(func.sum(ApiUsage.cost_usd), 0).desc())
    )
    return [UsageLine(*row) for row in rows]


def format_summary(lines: list[UsageLine], days: int) -> str:
    out = [
        f"API usage, last {days} days",
        f"  {'purpose':<18}{'model':<20}{'calls':>7}{'in tok':>11}{'out tok':>10}"
        f"{'cost':>11}{'per call':>11}",
    ]
    for line in lines:
        out.append(
            f"  {line.purpose:<18}{line.model:<20}{line.calls:>7}{line.input_tokens:>11,}"
            f"{line.output_tokens:>10,}{float(line.cost_usd):>11.4f}"
            f"{float(line.cost_per_call):>11.5f}"
        )
    total = sum((line.cost_usd for line in lines), Decimal(0))
    out.append(
        f"  {'total':<45}{sum(line.calls for line in lines):>7}{'':>21}{float(total):>11.4f}"
    )
    return "\n".join(out)


async def main(days: int) -> None:
    from threadlight.db.session import SessionLocal, engine

    async with SessionLocal() as session:
        lines = await summarize(session, datetime.now(UTC) - timedelta(days=days))
    await engine.dispose()
    print(format_summary(lines, days))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    asyncio.run(main(parser.parse_args().days))
