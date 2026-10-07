"""ThreadLight command line: one entry point for every process.

    threadlight check       validate configuration and database
    threadlight migrate     apply database migrations
    threadlight bot         Discord bot: history sync, live mirroring, /ask, /decisions
    threadlight worker      background processing (segment, embed, extract decisions)
    threadlight api         HTTP API (internal: trusts the user_id it is given)
    threadlight usage       API usage and cost report

Also runnable as `python -m threadlight ...`.
"""

import sys
from collections.abc import Callable

COMMANDS: dict[str, tuple[str, Callable[[list[str]], None]]] = {}


def command(name: str, help_text: str):
    def register(fn: Callable[[list[str]], None]):
        COMMANDS[name] = (help_text, fn)
        return fn

    return register


@command("check", "validate configuration and database")
def _check(argv: list[str]) -> None:
    from threadlight.check import main

    raise SystemExit(main())


@command("migrate", "apply database migrations")
def _migrate(argv: list[str]) -> None:
    from threadlight.check import run_migrations

    run_migrations()


@command("bot", "run the Discord bot")
def _bot(argv: list[str]) -> None:
    from threadlight.ingest.bot import main

    main()


@command("worker", "run the background worker")
def _worker(argv: list[str]) -> None:
    from threadlight.processing.worker import cli

    cli(argv)


@command("api", "run the HTTP API")
def _api(argv: list[str]) -> None:
    import os

    import uvicorn

    uvicorn.run(
        "threadlight.api.main:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
    )


@command("usage", "API usage and cost report")
def _usage(argv: list[str]) -> None:
    import argparse
    import asyncio

    from threadlight.usage import main

    parser = argparse.ArgumentParser(prog="threadlight usage")
    parser.add_argument("--days", type=int, default=30)
    asyncio.run(main(parser.parse_args(argv).days))


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        width = max(len(n) for n in COMMANDS)
        lines = [f"  {name:<{width}}  {help_text}" for name, (help_text, _) in COMMANDS.items()]
        print("usage: threadlight <command> [options]\n\ncommands:\n" + "\n".join(lines))
        raise SystemExit(0 if argv and argv[0] in ("-h", "--help") else 2)
    COMMANDS[argv[0]][1](argv[1:])


if __name__ == "__main__":
    main()
