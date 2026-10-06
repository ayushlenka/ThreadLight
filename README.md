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
- [x] M1: Discord ingestion (resumable backfill + live create/edit/delete)
- [x] M2: conversation segmentation, embeddings, hybrid search
- [x] M3: permission-aware retrieval
- [x] M4: `/ask` with cited answers
- [x] M5: decision extraction and supersession
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

### Ingesting a Discord server

1. Create a bot in the Discord Developer Portal and enable the **Message Content** and
   **Server Members** privileged intents.
2. Set `DISCORD_BOT_TOKEN` and `DISCORD_GUILD_ID` in `.env`.
3. Run `python -m threadlight.ingest.bot`. If the bot isn't in the server yet, it logs an
   invite link. On every start it syncs history from each channel's checkpoint, then mirrors
   new messages, edits, and deletes live.

### Processing and search

Set `VOYAGE_API_KEY` in `.env`, then run the worker alongside the bot:

```bash
python -m threadlight.processing.worker --reindex --drain   # one-off: process everything
python -m threadlight.processing.worker                     # long-running: follow the bot
```

The bot queues a re-segmentation job whenever a channel changes (debounced to one job per
channel per minute). The worker splits the channel into conversations, keeps unchanged ones
by content hash, and embeds only what's new. Then query `GET /search?q=...&user_id=...`.

### Permissions

Results only come from channels the asking user can read. Rather than asking the bot's
cache, ThreadLight stores the permission state (roles, member roles, channel overwrites,
guild owner) in Postgres, keeps it current from gateway events, and resolves access with
Discord's documented algorithm (`retrieval/permissions.py`). The visible channel set is
applied inside the SQL of both the vector and keyword queries, so hidden content never
reaches ranking or the LLM.

To check the implementation against discord.py on a live server:

```bash
python -m threadlight.ingest.verify_permissions
```

It compares every (member, channel) pair and exits non-zero on any mismatch.

`user_id` on `/search` is trusted input: there is no end-user auth yet, so the API must
stay internal until callers verify identity (the bot's `/ask`, or Discord OAuth).

### Asking questions

With `ANTHROPIC_API_KEY` set, the bot registers a `/ask` slash command in your server.

- Each retrieved conversation is sent to Claude as a document with **one content block per
  message**, with citations enabled. Claude's citations come back as block ranges, which map
  directly to Discord message ids, so every source in the answer is a jump link to the exact
  message that supports it.
- Messages are loaded live at question time, so a message deleted a moment ago is excluded
  even before its conversation is re-segmented.
- Answers are ephemeral (visible only to the asker): they're built from the asker's visible
  channels, and posting them publicly would leak that content into a shared channel.
- Retrieved messages are treated as untrusted data in the prompt, so instructions inside
  chat messages aren't followed.

The same pipeline is available over HTTP as `POST /ask` with `{"question", "user_id"}`.
The model defaults to `claude-opus-5-5` (override with `ANSWER_MODEL` / `ANSWER_EFFORT`).

### Decisions

The worker extracts decisions from each conversation once (structured output from
Claude: title, summary, `decided` or `proposed`, reasons, alternatives, and the message
where it was settled). When a decision is stored, its nearest existing decisions are
compared and Claude names pairs where a newer decision replaces an older one, which works
regardless of the order conversations are processed in.

"Superseded" is never stored. It's derived per viewer from replacement links: a decision
shows as superseded only if the asker can see the decision that replaced it, so a private
channel's decision can't leak by marking a public one as changed. Decisions whose deciding
message was deleted disappear immediately, including from other decisions' references.

Extraction only runs on settled conversations: ones quiet for longer than the segmenter's
45-minute gap, after which they can't grow. An active conversation is re-segmented on every
message, so extracting it early would mean paying for it again and again; instead the
worker schedules a follow-up job for when it settles, and each conversation is extracted
once.

`/ask` includes matching decisions and their replacement chains as a citable decision log,
`/decisions [topic]` shows the log in Discord, and `GET /decisions?user_id=...&q=...`
serves it over HTTP.

### Evals

`evals/corpus/campus_eats.yaml` is a hand-written, labeled history of a student team
(23 conversations, 16 decisions including 3 reversals, a private channel, and hard
negatives like jokes and status updates). It loads as its own guild with realistic dates:

```bash
python -m threadlight.evals.corpus          # install and queue processing
python -m threadlight.processing.worker --drain
python -m threadlight.evals.extraction      # precision / recall / supersession links
```

### Cost tracking

Every Claude and Voyage call is logged to `api_usage` with tokens, list-price cost, and
context (guild, user, conversation). Each row is written in its own transaction, so paid
calls are recorded even when the surrounding work fails, and logging never fails a request.

```bash
python -m threadlight.usage --days 30
```

Measured so far (Opus 5.5): about $0.026 per `/ask` answer and $0.008 per extracted
conversation.

Run tests with `pytest` and lint with `ruff check .`.
