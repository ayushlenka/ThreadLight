"""Tests run against a separate `threadlight_test` database so they never touch real data.

The env var must be set before any threadlight module creates its engine.
"""

import asyncio
import os
import subprocess
import sys

import pytest

TEST_DB = "threadlight_test"
_ADMIN_DSN = "postgresql://threadlight:threadlight@localhost:5433/postgres"
os.environ["DATABASE_URL"] = (
    f"postgresql+asyncpg://threadlight:threadlight@localhost:5433/{TEST_DB}"
)

from sqlalchemy import text  # noqa: E402

from threadlight.db.session import engine  # noqa: E402

_TABLES = (
    "guilds, channels, messages, conversations, conversation_messages, "
    "decisions, decision_sources, jobs"
)


async def _create_test_db() -> None:
    import asyncpg

    conn = await asyncpg.connect(_ADMIN_DSN)
    try:
        if not await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", TEST_DB):
            await conn.execute(f"CREATE DATABASE {TEST_DB}")
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def db_available() -> bool:
    try:
        asyncio.run(_create_test_db())
    except OSError:
        return False
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"], check=True, capture_output=True
    )
    return True


@pytest.fixture
async def db(db_available: bool):
    """Empty, migrated test database. Skips the test if Postgres isn't running."""
    if not db_available:
        pytest.skip("database not running (docker compose up -d)")
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {_TABLES} RESTART IDENTITY CASCADE"))
    yield
