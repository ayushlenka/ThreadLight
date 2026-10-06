"""Score decision extraction against the labeled corpus.

    python -m threadlight.evals.extraction             # corpus scores
    python -m threadlight.evals.extraction --real      # also list real-server decisions

Prerequisites: `python -m threadlight.evals.corpus` and a drained worker.

Matching: an extracted decision belongs to the corpus conversation that contains its
deciding message. Each labeled decision is matched to at most one extracted decision from
the same conversation; leftovers are false positives. Required labels are matched first;
`optional` labels (borderline decisions) then absorb leftovers without counting toward
precision or recall either way. Conversations marked `ambiguous` are excluded entirely.
"""

import argparse
import asyncio
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from threadlight.config import get_settings
from threadlight.db.models import Channel, Decision
from threadlight.db.session import SessionLocal, engine
from threadlight.evals.corpus import Corpus, parse


@dataclass
class ExtractionReport:
    true_positives: int = 0
    false_negatives: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)
    status_correct: int = 0
    status_wrong: list[str] = field(default_factory=list)
    links_correct: int = 0
    links_missed: list[str] = field(default_factory=list)
    links_wrong: list[str] = field(default_factory=list)
    optional_found: int = 0
    optional_total: int = 0

    @property
    def precision(self) -> float:
        found = self.true_positives + len(self.false_positives)
        return self.true_positives / found if found else 1.0

    @property
    def recall(self) -> float:
        expected = self.true_positives + len(self.false_negatives)
        return self.true_positives / expected if expected else 1.0

    @property
    def status_accuracy(self) -> float:
        total = self.status_correct + len(self.status_wrong)
        return self.status_correct / total if total else 1.0

    @property
    def link_precision(self) -> float:
        found = self.links_correct + len(self.links_wrong)
        return self.links_correct / found if found else 1.0

    @property
    def link_recall(self) -> float:
        expected = self.links_correct + len(self.links_missed)
        return self.links_correct / expected if expected else 1.0


async def score(session: AsyncSession, corpus: Corpus) -> ExtractionReport:
    msg_to_conv = {m.id: c.key for c in corpus.conversations for m in c.messages}
    ambiguous = {c.key for c in corpus.conversations if c.ambiguous}

    extracted = list(
        await session.scalars(
            select(Decision)
            .where(Decision.channel_id.in_(corpus.channel_ids.values()))
            .order_by(Decision.decided_at, Decision.id)
        )
    )
    unmatched: dict[str, list[Decision]] = {}
    for d in extracted:
        unmatched.setdefault(msg_to_conv.get(d.decided_message_id, "?"), []).append(d)

    report = ExtractionReport()
    truth_to_db: dict[str, int] = {}
    db_to_truth: dict[int, str] = {}
    ordered = sorted(corpus.decisions, key=lambda t: t.optional)  # required first
    for truth in ordered:
        pool = unmatched.get(truth.conversation_key, [])
        report.optional_total += truth.optional
        if not pool:
            if not truth.optional:
                report.false_negatives.append(f"{truth.key}: {truth.title}")
            continue
        # Prefer the extracted decision settled closest to the labeled message.
        best = min(pool, key=lambda d: abs(d.decided_message_id - truth.message_id))
        pool.remove(best)
        truth_to_db[truth.key] = best.id
        db_to_truth[best.id] = truth.key
        if truth.optional:
            report.optional_found += 1
            continue
        report.true_positives += 1
        if best.status == truth.status:
            report.status_correct += 1
        else:
            report.status_wrong.append(f"{truth.key}: expected {truth.status}, got {best.status}")

    for conv_key, leftovers in unmatched.items():
        if conv_key in ambiguous:
            continue
        report.false_positives += [f"[{conv_key}] {d.title} ({d.status})" for d in leftovers]

    expected_links = {(t.key, t.supersedes) for t in corpus.decisions if t.supersedes}
    found_links = {
        (
            db_to_truth.get(d.id, f"db:{d.id}"),
            db_to_truth.get(d.supersedes_id, f"db:{d.supersedes_id}"),
        )
        for d in extracted
        if d.supersedes_id is not None
    }
    report.links_correct = len(expected_links & found_links)
    report.links_missed = [f"{a} -> {b}" for a, b in sorted(expected_links - found_links)]
    report.links_wrong = [f"{a} -> {b}" for a, b in sorted(found_links - expected_links)]
    return report


def format_report(r: ExtractionReport) -> str:
    lines = [
        "Decision extraction (synthetic corpus)",
        f"  precision        {r.precision:6.1%}  ({r.true_positives} correct, "
        f"{len(r.false_positives)} spurious)",
        f"  recall           {r.recall:6.1%}  ({r.true_positives} of "
        f"{r.true_positives + len(r.false_negatives)} found)",
        f"  status accuracy  {r.status_accuracy:6.1%}",
        f"  optional found   {r.optional_found} of {r.optional_total} (not scored)",
        f"  link precision   {r.link_precision:6.1%}",
        f"  link recall      {r.link_recall:6.1%}  ({r.links_correct} of "
        f"{r.links_correct + len(r.links_missed)} supersessions)",
    ]
    for label, items in [
        ("missed", r.false_negatives),
        ("spurious", r.false_positives),
        ("wrong status", r.status_wrong),
        ("missed links", r.links_missed),
        ("wrong links", r.links_wrong),
    ]:
        if items:
            lines.append(f"\n  {label}:")
            lines += [f"    - {item}" for item in items]
    return "\n".join(lines)


async def real_server_decisions(session: AsyncSession, guild_id: int) -> list[str]:
    rows = await session.execute(
        select(Decision.decided_at, Channel.name, Decision.status, Decision.title)
        .join(Channel, Channel.id == Decision.channel_id)
        .where(Channel.guild_id == guild_id)
        .order_by(Decision.decided_at)
    )
    return [f"{at:%Y-%m-%d} #{ch} [{status}] {title}" for at, ch, status, title in rows]


async def main(include_real: bool) -> None:
    corpus = parse()
    async with SessionLocal() as session:
        print(format_report(await score(session, corpus)))
        guild_id = get_settings().discord_guild_id
        if include_real and guild_id:
            found = await real_server_decisions(session, guild_id)
            print(f"\nReal server: {len(found)} decisions extracted (review for false positives)")
            for line in found:
                print(f"  {line}")
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="list real-server decisions")
    asyncio.run(main(parser.parse_args().real))
