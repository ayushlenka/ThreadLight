from sqlalchemy import update

from threadlight.db.models import Member
from threadlight.db.session import SessionLocal
from threadlight.ingest import store
from threadlight.retrieval.permissions import (
    ADMINISTRATOR,
    ALL_PERMISSIONS,
    MANAGE_THREADS,
    READ,
    READ_MESSAGE_HISTORY,
    VIEW_CHANNEL,
    ChannelOverwrites,
    Overwrite,
    base_permissions,
    can_read,
    channel_permissions,
    visible_channel_ids,
)

GUILD = 1  # also the @everyone role id
OWNER = 900
ALICE, BOB, CAROL, MALLORY = 901, 902, 903, 904
MOD_ROLE, MEMBER_ROLE, ADMIN_ROLE, MUTED_ROLE = 50, 51, 52, 53


# Pure algorithm


def base(user_id=ALICE, roles=(), everyone=READ, extra=None):
    role_perms = {GUILD: everyone, **(extra or {})}
    return base_permissions(
        user_id, list(roles), guild_id=GUILD, owner_id=OWNER, role_permissions=role_perms
    )


def test_owner_gets_everything():
    assert base(user_id=OWNER, everyone=0) == ALL_PERMISSIONS


def test_base_is_everyone_or_roles():
    perms = base(roles=[MOD_ROLE], everyone=VIEW_CHANNEL, extra={MOD_ROLE: MANAGE_THREADS})
    assert perms == VIEW_CHANNEL | MANAGE_THREADS


def test_administrator_role_gets_everything_and_ignores_overwrites():
    perms = base(roles=[ADMIN_ROLE], everyone=0, extra={ADMIN_ROLE: ADMINISTRATOR})
    assert perms == ALL_PERMISSIONS
    deny_all = ChannelOverwrites(everyone=Overwrite(allow=0, deny=ALL_PERMISSIONS))
    assert channel_permissions(perms, ALICE, [ADMIN_ROLE], deny_all) == ALL_PERMISSIONS


def test_everyone_overwrite_can_hide_channel():
    ow = ChannelOverwrites(everyone=Overwrite(allow=0, deny=VIEW_CHANNEL))
    assert not can_read(channel_permissions(READ, ALICE, [], ow))


def test_role_allow_beats_everyone_deny():
    ow = ChannelOverwrites(
        everyone=Overwrite(allow=0, deny=VIEW_CHANNEL),
        roles={MOD_ROLE: Overwrite(allow=VIEW_CHANNEL, deny=0)},
    )
    assert can_read(channel_permissions(READ, ALICE, [MOD_ROLE], ow))
    assert not can_read(channel_permissions(READ, BOB, [], ow))


def test_role_allow_beats_role_deny_when_member_has_both():
    """Role overwrites are combined: all denies applied, then all allows."""
    ow = ChannelOverwrites(
        roles={
            MUTED_ROLE: Overwrite(allow=0, deny=VIEW_CHANNEL),
            MOD_ROLE: Overwrite(allow=VIEW_CHANNEL, deny=0),
        }
    )
    assert can_read(channel_permissions(READ, ALICE, [MUTED_ROLE, MOD_ROLE], ow))
    assert not can_read(channel_permissions(READ, ALICE, [MUTED_ROLE], ow))


def test_member_overwrite_beats_role_overwrites():
    ow = ChannelOverwrites(
        roles={MOD_ROLE: Overwrite(allow=VIEW_CHANNEL, deny=0)},
        members={ALICE: Overwrite(allow=0, deny=VIEW_CHANNEL)},
    )
    assert not can_read(channel_permissions(READ, ALICE, [MOD_ROLE], ow))


def test_view_without_history_is_not_readable():
    ow = ChannelOverwrites(everyone=Overwrite(allow=0, deny=READ_MESSAGE_HISTORY))
    assert not can_read(channel_permissions(READ, ALICE, [], ow))


# Against the database


GENERAL, MODS, SECRET_DM, PUBLIC_THREAD, MOD_THREAD, PRIVATE_THREAD, ORPHAN_THREAD = range(10, 17)


async def seed():
    """#general is public; #mods is hidden from @everyone but open to MOD_ROLE, and Bob is
    personally denied; #secret is visible only to Carol via a member overwrite."""
    channels = [
        (GENERAL, "text", None, "general"),
        (MODS, "text", None, "mods"),
        (SECRET_DM, "text", None, "secret"),
        (PUBLIC_THREAD, "public_thread", GENERAL, "general-thread"),
        (MOD_THREAD, "public_thread", MODS, "mods-thread"),
        (PRIVATE_THREAD, "private_thread", GENERAL, "private-thread"),
        (ORPHAN_THREAD, "public_thread", 999, "orphan-thread"),
    ]
    async with SessionLocal.begin() as s:
        await store.upsert_guild(s, GUILD, "g", owner_id=OWNER)
        await store.replace_roles(
            s,
            GUILD,
            [
                {"id": GUILD, "guild_id": GUILD, "name": "@everyone", "permissions": READ},
                {"id": MOD_ROLE, "guild_id": GUILD, "name": "mod", "permissions": MANAGE_THREADS},
            ],
        )
        await store.replace_members(
            s,
            GUILD,
            [
                {"guild_id": GUILD, "user_id": uid, "display_name": name, "role_ids": roles}
                for uid, name, roles in [
                    (OWNER, "owner", []),
                    (ALICE, "alice", [MOD_ROLE]),
                    (BOB, "bob", [MOD_ROLE]),
                    (CAROL, "carol", []),
                    (MALLORY, "mallory", []),
                ]
            ],
        )
        await store.upsert_channels(
            s,
            [
                {"id": cid, "guild_id": GUILD, "parent_id": parent, "type": t, "name": n}
                for cid, t, parent, n in channels
            ],
        )
        hidden = {"target_id": GUILD, "target_type": "role", "allow": 0, "deny": VIEW_CHANNEL}
        await store.replace_overwrites(
            s,
            MODS,
            [
                {"channel_id": MODS, **hidden},
                {
                    "channel_id": MODS,
                    "target_id": MOD_ROLE,
                    "target_type": "role",
                    "allow": VIEW_CHANNEL,
                    "deny": 0,
                },
                {
                    "channel_id": MODS,
                    "target_id": BOB,
                    "target_type": "member",
                    "allow": 0,
                    "deny": VIEW_CHANNEL,
                },
            ],
        )
        await store.replace_overwrites(
            s,
            SECRET_DM,
            [
                {"channel_id": SECRET_DM, **hidden},
                {
                    "channel_id": SECRET_DM,
                    "target_id": CAROL,
                    "target_type": "member",
                    "allow": VIEW_CHANNEL,
                    "deny": 0,
                },
            ],
        )


async def visible(user_id: int) -> set[int]:
    async with SessionLocal() as s:
        return set(await visible_channel_ids(s, GUILD, user_id))


async def test_visibility_per_member(db):
    await seed()
    # Alice is a mod: mods channel, its thread, and private threads (MANAGE_THREADS).
    assert await visible(ALICE) == {GENERAL, MODS, PUBLIC_THREAD, MOD_THREAD, PRIVATE_THREAD}
    # Bob is a mod but personally denied #mods, which also hides its threads.
    assert await visible(BOB) == {GENERAL, PUBLIC_THREAD, PRIVATE_THREAD}
    # Carol has a member overwrite for #secret only.
    assert await visible(CAROL) == {GENERAL, SECRET_DM, PUBLIC_THREAD}
    assert await visible(MALLORY) == {GENERAL, PUBLIC_THREAD}


async def test_owner_sees_everything_except_orphaned_threads(db):
    await seed()
    assert await visible(OWNER) == {
        GENERAL,
        MODS,
        SECRET_DM,
        PUBLIC_THREAD,
        MOD_THREAD,
        PRIVATE_THREAD,
    }


async def test_non_members_and_former_members_see_nothing(db):
    await seed()
    assert await visible(12345) == set()
    async with SessionLocal.begin() as s:
        await store.mark_member_left(s, GUILD, CAROL)
    assert await visible(CAROL) == set()


async def test_rejoining_restores_access(db):
    await seed()
    async with SessionLocal.begin() as s:
        await store.mark_member_left(s, GUILD, CAROL)
        await store.upsert_members(
            s, [{"guild_id": GUILD, "user_id": CAROL, "display_name": "carol", "role_ids": []}]
        )
    assert SECRET_DM in await visible(CAROL)


async def test_role_removal_revokes_access(db):
    await seed()
    async with SessionLocal.begin() as s:
        await s.execute(update(Member).where(Member.user_id == ALICE).values(role_ids=[]))
    assert MODS not in await visible(ALICE)


async def test_deleted_role_grants_nothing(db):
    await seed()
    async with SessionLocal.begin() as s:
        await store.delete_role(s, MOD_ROLE)
    # Alice still lists the role id, but it no longer exists; the overwrite keyed on it
    # still applies (Discord removes overwrites for deleted roles via a channel update).
    assert PRIVATE_THREAD not in await visible(ALICE)


async def test_replace_members_marks_absent_members_left(db):
    await seed()
    async with SessionLocal.begin() as s:
        await store.replace_members(
            s, GUILD, [{"guild_id": GUILD, "user_id": ALICE, "display_name": "a", "role_ids": []}]
        )
    assert await visible(BOB) == set()
    assert GENERAL in await visible(ALICE)
