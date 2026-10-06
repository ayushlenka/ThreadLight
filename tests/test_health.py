from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from threadlight.api.main import app
from threadlight.db.session import engine


async def test_health_reports_database_ok(db):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "database": "ok"}


async def test_schema_is_migrated(db):
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        tables = {r[0] for r in rows}

    assert {
        "guilds",
        "channels",
        "messages",
        "conversations",
        "conversation_messages",
        "decisions",
        "decision_sources",
        "jobs",
    } <= tables
