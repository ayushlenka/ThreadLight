"""Group a channel's messages into conversations: the unit we embed and retrieve.

Single chat messages are too short to retrieve well ("yeah let's do that" means nothing on
its own), so we split each channel's timeline into bursts of activity. Chat is strongly
bursty (on the dev server: median gap ~3 minutes, but most gaps over 30 minutes are hours
or days), so a time-gap rule captures most conversation boundaries. Size caps keep very
long bursts from becoming one unfocused embedding.

Each conversation is rendered twice:
- `text`: the transcript with channel and timestamps, for embeddings and LLM prompts.
- `search_text`: speaker and content only, for the keyword index, so timestamps and the
  channel header don't match every query.

Replies that reach back into an earlier conversation keep their context by quoting the
message they reply to.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from urllib.parse import urlsplit

DEFAULT_GAP = timedelta(minutes=45)
DEFAULT_MAX_MESSAGES = 40
DEFAULT_MAX_CHARS = 6000
QUOTE_CHARS = 80

_USER_MENTION = re.compile(r"<@!?(\d+)>")
_ROLE_MENTION = re.compile(r"<@&\d+>")
_CHANNEL_MENTION = re.compile(r"<#(\d+)>")
_CUSTOM_EMOJI = re.compile(r"<a?:(\w+):\d+>")
_URL = re.compile(r"https?://\S+")
_GIF_HOSTS = ("tenor.com", "giphy.com", "klipy.com")
_DISCORD_CDN_HOSTS = ("cdn.discordapp.com", "media.discordapp.net")


@dataclass(frozen=True)
class SegMessage:
    id: int
    author_name: str
    content: str
    created_at: datetime
    reply_to_id: int | None = None
    attachment_names: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.content) + sum(len(a) for a in self.attachment_names)


@dataclass(frozen=True)
class Names:
    """ID -> display name lookups for resolving Discord mention markup."""

    users: dict[int, str] = field(default_factory=dict)
    channels: dict[int, str] = field(default_factory=dict)


@dataclass
class Segment:
    messages: list[SegMessage]
    text: str
    search_text: str

    @property
    def first(self) -> SegMessage:
        return self.messages[0]

    @property
    def last(self) -> SegMessage:
        return self.messages[-1]

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


def split(
    messages: list[SegMessage],
    *,
    gap: timedelta = DEFAULT_GAP,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[list[SegMessage]]:
    """Split a chronologically sorted list of messages into conversation groups."""
    groups: list[list[SegMessage]] = []
    current: list[SegMessage] = []
    chars = 0
    for msg in messages:
        if current and (
            msg.created_at - current[-1].created_at > gap
            or len(current) >= max_messages
            or chars + msg.size > max_chars
        ):
            groups.append(current)
            current, chars = [], 0
        current.append(msg)
        chars += msg.size
    if current:
        groups.append(current)
    return groups


def clean_content(content: str, names: Names) -> str:
    """Make raw Discord markup readable for embeddings, search, and LLM prompts.

    Mentions become names, and media links (GIFs, CDN uploads) collapse to short markers.
    Long random URL tokens are pure noise for both embeddings and keyword search, while
    other links (game invites, server addresses, docs) are kept because they're often what
    someone is looking for.
    """
    content = _USER_MENTION.sub(lambda m: "@" + names.users.get(int(m[1]), "someone"), content)
    content = _ROLE_MENTION.sub("@role", content)
    content = _CHANNEL_MENTION.sub(
        lambda m: "#" + names.channels.get(int(m[1]), "channel"), content
    )
    content = _CUSTOM_EMOJI.sub(r":\1:", content)
    return _URL.sub(lambda m: _clean_url(m[0]), content)


def _clean_url(url: str) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").removeprefix("www.")
    path = PurePosixPath(parts.path)
    if host.endswith(_GIF_HOSTS) or path.suffix.lower() == ".gif":
        return "[gif]"
    if host in _DISCORD_CDN_HOSTS:
        return f"[attachment: {path.name}]"
    return url


def _speaker_and_body(
    m: SegMessage, in_group: set[int], by_id: dict[int, SegMessage], names: Names
) -> tuple[str, str]:
    speaker = m.author_name
    target = by_id.get(m.reply_to_id) if m.reply_to_id else None
    if target is not None:
        if target.id in in_group:
            speaker += f" (replying to {target.author_name})"
        else:
            quote = " ".join(clean_content(target.content, names).split())
            if len(quote) > QUOTE_CHARS:
                quote = quote[:QUOTE_CHARS].rstrip() + "..."
            speaker += f' (replying to {target.author_name}: "{quote}")'
    parts = [clean_content(m.content, names), *(f"[attachment: {a}]" for a in m.attachment_names)]
    return speaker, " ".join(p for p in parts if p)


def message_lines(
    group: list[SegMessage], by_id: dict[int, SegMessage], names: Names | None = None
) -> list[tuple[str, str]]:
    """(timestamped line, search line) for each message, in order."""
    names = names or Names()
    in_group = {m.id for m in group}
    out = []
    for m in group:
        speaker, body = _speaker_and_body(m, in_group, by_id, names)
        out.append((f"[{m.created_at:%Y-%m-%d %H:%M}] {speaker}: {body}", f"{speaker}: {body}"))
    return out


def render(
    channel_name: str,
    group: list[SegMessage],
    by_id: dict[int, SegMessage],
    names: Names | None = None,
) -> tuple[str, str]:
    """Returns (text, search_text) for a conversation."""
    lines = message_lines(group, by_id, names)
    text = "\n".join([f"#{channel_name}", *(line for line, _ in lines)])
    return text, "\n".join(search for _, search in lines)


def segment(
    channel_name: str,
    messages: list[SegMessage],
    names: Names | None = None,
    **split_opts,
) -> list[Segment]:
    by_id = {m.id: m for m in messages}
    segments = []
    for group in split(messages, **split_opts):
        text, search_text = render(channel_name, group, by_id, names)
        segments.append(Segment(messages=group, text=text, search_text=search_text))
    return segments
