"""Configuration and database checks, and migrations, for `threadlight check|migrate`."""

import asyncio
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text

from alembic import command
from threadlight.config import get_settings

SETUP_GUIDE = "docs/self-hosting.md"


def _alembic_config() -> Config:
    # alembic.ini lives at the repo root locally and at /app in the container image.
    for base in (Path.cwd(), Path(__file__).resolve().parents[2]):
        ini = base / "alembic.ini"
        if ini.exists():
            return Config(str(ini))
    raise SystemExit("alembic.ini not found; run from the project root")


def run_migrations() -> None:
    command.upgrade(_alembic_config(), "head")


async def _database_status() -> tuple[bool, str]:
    from threadlight.db.session import engine

    head = ScriptDirectory.from_config(_alembic_config()).get_current_head()
    try:
        async with engine.connect() as conn:
            try:
                current = await conn.scalar(text("SELECT version_num FROM alembic_version"))
            except Exception:
                current = None
    except Exception as exc:
        return False, f"cannot connect ({type(exc).__name__}). Is the database running?"
    finally:
        await engine.dispose()
    if current != head:
        return (
            False,
            f"schema at {current or 'nothing'}, expected {head}: run `threadlight migrate`",
        )
    return True, f"connected, schema up to date ({head})"


def main() -> int:
    s = get_settings()
    problems = 0

    def line(ok: bool | None, label: str, detail: str) -> None:
        nonlocal problems
        mark = {True: "ok  ", False: "FAIL", None: "warn"}[ok]
        problems += ok is False
        print(f"  [{mark}] {label:<22} {detail}")

    print("ThreadLight configuration")
    line(
        bool(s.discord_bot_token), "DISCORD_BOT_TOKEN", "set" if s.discord_bot_token else "missing"
    )
    line(
        s.discord_guild_id is not None,
        "DISCORD_GUILD_ID",
        str(s.discord_guild_id) if s.discord_guild_id else "missing (your server's ID)",
    )
    line(
        bool(s.voyage_api_key),
        "VOYAGE_API_KEY",
        "set" if s.voyage_api_key else "missing: needed for search, /ask, and the worker",
    )
    if s.anthropic_api_key:
        line(
            True,
            "ANTHROPIC_API_KEY",
            f"set (answers: {s.answer_model}, extraction: {s.extract_model})",
        )
    else:
        line(None, "ANTHROPIC_API_KEY", "missing: /ask and decision extraction disabled")

    ok, detail = asyncio.run(_database_status())
    line(ok, "database", detail)

    if problems:
        print(f"\n{problems} problem(s). Setup guide: {SETUP_GUIDE}")
    else:
        print("\nAll good.")
    return 1 if problems else 0
