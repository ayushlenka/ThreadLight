"""Which channels can a member read? Discord's permission algorithm, computed from Postgres.

This mirrors Discord's documented resolution order
(https://discord.com/developers/docs/topics/permissions#permission-overwrites):

1. Base permissions: the guild owner gets everything. Otherwise OR together @everyone's
   permissions and the member's roles'. ADMINISTRATOR grants everything.
2. Channel overwrites, in order: the @everyone overwrite, then all role overwrites
   combined (deny first, then allow), then the member's own overwrite.

A member can read a channel's history only with both VIEW_CHANNEL and
READ_MESSAGE_HISTORY. Threads have no overwrites of their own and inherit their parent's.
Private threads are restricted further: we only grant them to members with MANAGE_THREADS
on the parent. Discord also shows them to invited thread members, which we don't track, so
this errs toward hiding.

`ingest/verify_permissions.py` checks this implementation against discord.py's
`permissions_for` for every (member, channel) pair in a live guild.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from threadlight.db.models import Channel, ChannelOverwrite, Guild, Member, Role

ADMINISTRATOR = 1 << 3
VIEW_CHANNEL = 1 << 10
READ_MESSAGE_HISTORY = 1 << 16
MANAGE_THREADS = 1 << 34
ALL_PERMISSIONS = (1 << 63) - 1

THREAD_TYPES = {"public_thread", "private_thread", "news_thread"}
READ = VIEW_CHANNEL | READ_MESSAGE_HISTORY


@dataclass(frozen=True)
class Overwrite:
    allow: int
    deny: int


@dataclass
class ChannelOverwrites:
    everyone: Overwrite | None = None
    roles: dict[int, Overwrite] = field(default_factory=dict)
    members: dict[int, Overwrite] = field(default_factory=dict)


def base_permissions(
    user_id: int,
    member_role_ids: list[int],
    *,
    guild_id: int,
    owner_id: int | None,
    role_permissions: dict[int, int],
) -> int:
    if user_id == owner_id:
        return ALL_PERMISSIONS
    perms = role_permissions.get(guild_id, 0)  # @everyone role id == guild id
    for role_id in member_role_ids:
        perms |= role_permissions.get(role_id, 0)
    if perms & ADMINISTRATOR:
        return ALL_PERMISSIONS
    return perms


def channel_permissions(
    base: int, user_id: int, member_role_ids: list[int], overwrites: ChannelOverwrites
) -> int:
    if base & ADMINISTRATOR:
        return ALL_PERMISSIONS
    perms = base
    if overwrites.everyone is not None:
        perms = (perms & ~overwrites.everyone.deny) | overwrites.everyone.allow

    allow = deny = 0
    for role_id in member_role_ids:
        if (ow := overwrites.roles.get(role_id)) is not None:
            allow |= ow.allow
            deny |= ow.deny
    perms = (perms & ~deny) | allow

    if (ow := overwrites.members.get(user_id)) is not None:
        perms = (perms & ~ow.deny) | ow.allow
    return perms


def can_read(perms: int) -> bool:
    return perms & READ == READ


async def visible_channel_ids(session: AsyncSession, guild_id: int, user_id: int) -> list[int]:
    """Channel and thread ids whose history `user_id` can read. Empty for non-members."""
    member = await session.get(Member, (guild_id, user_id))
    if member is None or member.left_at is not None:
        return []
    guild = await session.get(Guild, guild_id)
    if guild is None:
        return []

    role_permissions = dict(
        (
            await session.execute(
                select(Role.id, Role.permissions).where(Role.guild_id == guild_id)
            )
        ).all()
    )
    base = base_permissions(
        user_id,
        member.role_ids,
        guild_id=guild_id,
        owner_id=guild.owner_id,
        role_permissions=role_permissions,
    )

    channels = (
        await session.execute(
            select(Channel.id, Channel.type, Channel.parent_id).where(
                Channel.guild_id == guild_id, Channel.deleted_at.is_(None)
            )
        )
    ).all()
    overwrites: dict[int, ChannelOverwrites] = defaultdict(ChannelOverwrites)
    rows = await session.scalars(
        select(ChannelOverwrite)
        .join(Channel, Channel.id == ChannelOverwrite.channel_id)
        .where(Channel.guild_id == guild_id)
    )
    for ow in rows:
        entry = Overwrite(allow=ow.allow, deny=ow.deny)
        target = overwrites[ow.channel_id]
        if ow.target_type == "role" and ow.target_id == guild_id:
            target.everyone = entry
        elif ow.target_type == "role":
            target.roles[ow.target_id] = entry
        else:
            target.members[ow.target_id] = entry

    def perms_in(channel_id: int) -> int:
        return channel_permissions(base, user_id, member.role_ids, overwrites[channel_id])

    stored = {c.id for c in channels}
    visible = []
    for c in channels:
        if c.type in THREAD_TYPES:
            # A thread whose parent we don't store (deleted, or unreadable by the bot) has
            # no permission context, so it stays hidden.
            if c.parent_id not in stored:
                continue
            perms = perms_in(c.parent_id)
            if c.type == "private_thread" and not perms & MANAGE_THREADS:
                continue
        else:
            perms = perms_in(c.id)
        if can_read(perms):
            visible.append(c.id)
    return visible
