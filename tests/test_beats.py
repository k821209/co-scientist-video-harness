"""Beat coercion: episode files already carry plain 7-tuples — keep them working."""
from vh.steps.beats import Beat


def test_tuple_is_accepted():
    b = Beat.coerce(("b1", "gfx", "본문 나레이션", "g_lineup", 0.0, None, None))
    assert (b.id, b.kind, b.visual, b.at) == ("b1", "gfx", "g_lineup", 0.0)
    assert b.caption is None and b.credit is None


def test_b0_with_caption_becomes_eyecatch():
    """The cold open's caption is the punch line, rendered big and centred."""
    b = Beat.coerce(("b0", "clip", "훅", "keynote", 11.0, "폴드7의 후속이 아니다", "영상 · 공식"))
    assert b.eyecatch and b.at == 11.0


def test_later_beat_caption_is_not_eyecatch():
    b = Beat.coerce(("b3", "clip", "본문", "film", 4.0, "메인 7.6형", "영상 · 공식"))
    assert not b.eyecatch


def test_beat_passes_through():
    b = Beat("b2", "photo", "본문", "img/x.jpg", caption="캡션")
    assert Beat.coerce(b) is b
