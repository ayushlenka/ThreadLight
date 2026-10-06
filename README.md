# ThreadLight

Ask your Discord server what it decided, and get a cited answer instead of a pile of search results.

> **"What did we decide about moving auth to OAuth?"**
> The team decided to migrate from JWT-only auth to Google OAuth in February, citing onboarding
> friction and password-reset issues. Sources: #engineering Feb 13, #backend Feb 16.

ThreadLight ingests a server's message history, groups messages into conversations, extracts
decisions with their rationale, and answers questions with hybrid (keyword + vector) retrieval
and Claude. Answers only draw from channels **the asking user can see**.

## Architecture

```
Discord (bot: backfill + live events)
        |
     Postgres  <-- jobs table (SELECT ... FOR UPDATE SKIP LOCKED)
        |
   Processing workers: segment -> embed -> extract decisions
        |
   Postgres + pgvector (full-text + HNSW vector indexes)
        |
   Permission filter -> hybrid search -> rerank -> Claude answer with citations
        |
   FastAPI  /  Discord /ask
```

One database on purpose: full-text search, vectors, and the job queue all live in Postgres, so
there is no second store to keep in sync.

## Status

- [x] M0: scaffold, schema, migrations
- [ ] M1: Discord ingestion (resumable backfill + live create/edit/delete)
- [ ] M2: conversation segmentation, embeddings, hybrid search
- [ ] M3: permission-aware retrieval
- [ ] M4: `/ask` with cited answers
- [ ] M5: decision extraction and supersession
- [ ] M6: eval harness and metrics

## Local development

Requires Python 3.11+ and Docker.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
cp .env.example .env

docker compose up -d          # Postgres 16 + pgvector on localhost:5433
alembic upgrade head
uvicorn threadlight.api.main:app --reload
```

Then `GET http://localhost:8000/health` should return `{"status": "ok", "database": "ok"}`.

Run tests with `pytest` and lint with `ruff check .`.
