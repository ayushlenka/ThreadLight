from datetime import UTC, datetime, timedelta

from threadlight.processing.segmenter import Names, SegMessage, clean_content, segment, split

T0 = datetime(2024, 2, 13, 14, 0, tzinfo=UTC)


def msg(i: int, minutes: float, content: str = "hi", author: str = "alice", **kw) -> SegMessage:
    return SegMessage(
        id=i, author_name=author, content=content, created_at=T0 + timedelta(minutes=minutes), **kw
    )


def ids(groups):
    return [[m.id for m in g] for g in groups]


def test_splits_on_time_gap():
    messages = [msg(1, 0), msg(2, 5), msg(3, 10), msg(4, 120), msg(5, 125)]
    assert ids(split(messages, gap=timedelta(minutes=45))) == [[1, 2, 3], [4, 5]]


def test_gap_equal_to_threshold_does_not_split():
    messages = [msg(1, 0), msg(2, 45)]
    assert ids(split(messages, gap=timedelta(minutes=45))) == [[1, 2]]


def test_caps_messages_per_conversation():
    messages = [msg(i, i) for i in range(5)]
    assert ids(split(messages, max_messages=2)) == [[0, 1], [2, 3], [4]]


def test_caps_characters_per_conversation():
    messages = [msg(1, 0, "a" * 60), msg(2, 1, "b" * 60), msg(3, 2, "c" * 10)]
    assert ids(split(messages, max_chars=100)) == [[1], [2, 3]]


def test_oversized_single_message_still_gets_a_conversation():
    assert ids(split([msg(1, 0, "x" * 500)], max_chars=100)) == [[1]]


def test_render_includes_channel_timestamps_and_speakers():
    [seg] = segment("engineering", [msg(1, 0, "move auth to OAuth?"), msg(2, 1, "yes", "bob")])
    assert seg.text == (
        "#engineering\n[2024-02-13 14:00] alice: move auth to OAuth?\n[2024-02-13 14:01] bob: yes"
    )


def test_reply_within_conversation_names_target():
    messages = [msg(1, 0, "OAuth?"), msg(2, 1, "agreed", "bob", reply_to_id=1)]
    [seg] = segment("eng", messages)
    assert "bob (replying to alice): agreed" in seg.text


def test_reply_to_earlier_conversation_quotes_target():
    messages = [
        msg(1, 0, "should we drop JWT auth"),
        msg(2, 600, "yes do it", "bob", reply_to_id=1),
    ]
    first, second = segment("eng", messages, gap=timedelta(minutes=45))
    assert 'bob (replying to alice: "should we drop JWT auth"): yes do it' in second.text


def test_attachments_are_rendered():
    [seg] = segment("eng", [msg(1, 0, "diagram:", attachment_names=("arch.png",))])
    assert "alice: diagram: [attachment: arch.png]" in seg.text


def test_hash_changes_with_content_and_is_stable_otherwise():
    a = segment("eng", [msg(1, 0, "one")])[0].content_hash
    b = segment("eng", [msg(1, 0, "one")])[0].content_hash
    c = segment("eng", [msg(1, 0, "two")])[0].content_hash
    assert a == b != c


# clean_content / search_text

NAMES = Names(users={42: "hugo"}, channels={7: "minecraft"})


def test_resolves_user_channel_and_role_mentions():
    raw = "<@42> <@!42> join <#7>, <@&99> too, <@1>"
    assert clean_content(raw, NAMES) == "@hugo @hugo join #minecraft, @role too, @someone"


def test_custom_emoji_become_names():
    assert clean_content("nice <:pog:123> <a:dance:456>", NAMES) == "nice :pog: :dance:"


def test_media_links_collapse_but_other_links_survive():
    raw = (
        "https://tenor.com/view/daffy-duck-gif-168 "
        "https://media.discordapp.net/attachments/1/2/image.png?ex=abc "
        "https://example.com/x.gif "
        "https://www.pokernow.club/games/abc"
    )
    assert clean_content(raw, NAMES) == (
        "[gif] [attachment: image.png] [gif] https://www.pokernow.club/games/abc"
    )


def test_search_text_omits_timestamps_and_channel_header():
    [seg] = segment("general", [msg(1, 0, "hey <@42>")], NAMES)
    assert seg.search_text == "alice: hey @hugo"
    assert seg.text == "#general\n[2024-02-13 14:00] alice: hey @hugo"
