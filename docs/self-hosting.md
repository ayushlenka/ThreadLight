# Self-hosting ThreadLight

Run your own ThreadLight bot for your Discord server. Everything runs on your machine
or server with Docker; your messages are stored in your own database.

About 20 minutes the first time.

## What you need

- **Docker** with Compose (Docker Desktop on Windows/macOS, or Docker Engine on Linux).
- **Manage Server** permission on the Discord server you want to index.
- A **Voyage AI** API key for search embeddings ([dash.voyageai.com](https://dash.voyageai.com)).
  The free allowance covers small servers. Without a payment method on file Voyage limits
  you to 3 requests per minute, which makes the first sync slow and can briefly rate-limit
  `/ask`; adding one lifts that.
- An **Anthropic** API key for `/ask` answers and decision extraction
  ([console.anthropic.com](https://console.anthropic.com)). Optional: without it, history
  sync and search still work, but `/ask` and decisions are disabled.

### What it costs

You pay the AI providers directly. With the default models (Claude Opus 5.5), measured:

| | Cost |
|---|---|
| Each `/ask` question | about $0.026 |
| Decision extraction, per conversation | about $0.008, once per conversation |
| Embeddings | effectively free at small scale |

A small, casual server is typically a few dollars for the initial history and a few
dollars a month after that. `threadlight usage` shows exactly what you've spent (below).

## 1. Create the Discord bot

1. Open the [Discord Developer Portal](https://discord.com/developers/applications) and
   click **New Application**. The name is what members will see (e.g. "ThreadLight").
2. Go to **Bot**:
   - Click **Reset Token** and copy it. This is your `DISCORD_BOT_TOKEN`. Treat it like a
     password.
   - Under **Privileged Gateway Intents**, turn on **Message Content Intent** and
     **Server Members Intent**. ThreadLight needs message content to index history and
     member roles to enforce channel permissions.
3. Go to **General Information** and copy the **Application ID**.
4. Invite the bot by opening this URL with your Application ID filled in, choosing your
   server, and clicking **Authorize**:

   ```
   https://discord.com/oauth2/authorize?client_id=YOUR_APPLICATION_ID&scope=bot+applications.commands&permissions=66560
   ```

   That grants only **View Channels** and **Read Message History**, plus the slash
   commands. (If you skip this step, the bot prints this link in its logs when it starts.)

5. Get your server's ID: in Discord, enable **User Settings > Advanced > Developer Mode**,
   then right-click your server icon and choose **Copy Server ID**. This is
   `DISCORD_GUILD_ID`.

## 2. Configure

```bash
git clone https://github.com/ayushlenka/ThreadLight.git
cd ThreadLight
cp .env.example .env
```

Edit `.env` and fill in `DISCORD_BOT_TOKEN`, `DISCORD_GUILD_ID`, `VOYAGE_API_KEY`, and
`ANTHROPIC_API_KEY`. Also change `POSTGRES_PASSWORD` to something of your own **before
the first start**; it's applied when the database is created.

## 3. Start it

```bash
docker compose up -d --build
```

This starts Postgres, applies database migrations, then starts the bot and the
background worker. Check everything is configured correctly:

```bash
docker compose run --rm bot check
```

Then watch the bot sync your history:

```bash
docker compose logs -f bot
```

You'll see each channel's message count and `sync complete`. The worker then groups
messages into conversations, embeds them, and extracts decisions in the background
(`docker compose logs -f worker`). On the free Voyage tier the first pass over a large
history takes a while because of rate limits; it resumes automatically if interrupted.

## 4. Use it

In your server:

- `/ask question:` asks about the server's history. Answers cite the exact messages,
  with jump links.
- `/decisions topic:` shows the decision log, optionally about a topic, including which
  decisions were later changed.

Answers are only visible to the person who asked, and only draw from channels **that
person** can read. Someone without access to a private channel never gets answers based
on it.

## Tell your members

ThreadLight reads and stores the history of every channel the bot can see. Let your
members know before you turn it on, for example with a pinned message. Things worth
mentioning:

- Messages are stored in your database, and message text is sent to Anthropic (answers,
  decision extraction) and Voyage AI (search embeddings) for processing.
- Deleting a message removes it from ThreadLight right away.
- Private channels stay private: answers respect each person's channel access.

**To keep a channel out of ThreadLight entirely**, deny the bot's role the
**View Channel** permission on that channel. The bot never reads channels it can't see.

## Day-to-day operations

| Task | Command |
|---|---|
| See status | `docker compose ps` |
| Logs | `docker compose logs -f bot worker` |
| What you've spent | `docker compose run --rm worker usage --days 30` |
| Update to the latest version | `git pull && docker compose up -d --build` |
| Stop | `docker compose stop` |
| Back up the database | `docker compose exec db pg_dump -U threadlight threadlight > backup.sql` |

Change models in `.env` (for example `EXTRACT_MODEL=claude-sonnet-5-5` for cheaper
extraction) and run `docker compose up -d` to apply.

### Optional: the HTTP API

`docker compose --profile api up -d` also starts the HTTP API on `localhost:8000`
(`/search`, `/ask`, `/decisions`). It trusts the `user_id` it's given, so it's bound to
localhost and meant for your own tools. **Don't expose it to the internet.**

## Removing ThreadLight

1. Kick the bot from your server (Server Settings > Integrations, or right-click it in
   the member list).
2. Delete all stored data:

   ```bash
   docker compose down -v
   ```

   `-v` removes the database volume. Without it, the data stays on disk.

## Troubleshooting

`docker compose run --rm bot check` diagnoses most problems.

| Symptom | Fix |
|---|---|
| Bot logs `bot is not in guild ...` | It isn't in the server, or `DISCORD_GUILD_ID` is wrong. The log line includes an invite link and the servers the bot is in. |
| Bot exits with `PrivilegedIntentsRequired` | Turn on **Message Content** and **Server Members** intents (step 1.2). |
| `/ask` and `/decisions` don't appear | Make sure the invite included `applications.commands` (use the URL in step 1.4), then restart the bot. |
| `/ask` says it's rate limited | Voyage free tier (3 requests/minute). Wait a minute, or add a payment method on Voyage. |
| `check` reports the database can't connect | Is the `db` container running (`docker compose ps`)? If you changed `POSTGRES_PASSWORD` after the first start, it no longer matches; change it back or start fresh with `docker compose down -v`. |
| A channel isn't being indexed | The bot needs **View Channel** and **Read Message History** there. |
