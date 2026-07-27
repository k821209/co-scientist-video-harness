"""cardkit's whole point is that a rule crossing text fails the build."""
import pytest

from vh.cardkit import Card


def test_rule_through_text_raises():
    c = Card(h=600)
    box = c.text((120, 300), "227만 8,100원", Card.fb(66), c.FG, "l")
    mid = int((box[1] + box[3]) / 2)
    with pytest.raises(AssertionError, match="would cross"):
        c.rule(mid, 100, 900)


def test_rule_below_text_is_fine():
    c = Card(h=600)
    box = c.text((120, 300), "227만 8,100원", Card.fb(66), c.FG, "l")
    c.rule(int(box[3]) + 30, 100, 900)          # clear of the glyphs -> no raise


def test_rule_hugging_text_raises():
    """A line 2px under the glyphs is as ugly as one through them."""
    c = Card(h=600)
    box = c.text((120, 300), "무게", Card.fr(40), c.DIM, "l")
    with pytest.raises(AssertionError):
        c.rule(int(box[3]) + 2, 100, 900)


def test_rule_beside_text_is_fine():
    """Same y, but the x spans don't overlap -> not a collision."""
    c = Card(h=600)
    box = c.text((120, 300), "무게", Card.fr(40), c.DIM, "l")
    c.rule(int((box[1] + box[3]) / 2), int(box[2]) + 40, 900)


def test_bar_sits_under_its_box():
    c = Card(h=600)
    box = c.text((540, 200), "PROFILE", Card.fr(34), c.MINT)
    assert c.bar(box) > box[3]


# ── metrics-first sizing (feedback 2c3b276c1232) ────────────────────────────

def test_ink_measures_actual_glyph_box():
    from vh.cardkit import Card
    c = Card(w=1080, h=600)
    w1, h1 = c.ink("무게", c.fr(34))
    w2, h2 = c.ink("무게", c.fr(68))
    assert w2 > w1 and h2 > h1                      # scales with the font
    assert c.ink("무게무게무게", c.fr(34))[0] > w1     # scales with the text


def test_stack_height_gives_symmetric_inner_margins():
    """box = pad + ink + gap + ink + pad, so top/bottom padding are equal."""
    from vh.cardkit import Card
    c = Card(w=1080, h=600)
    rows = [("정리", c.fr(34)), ("네이버 14.6조", c.fb(46))]
    pad, gap = 40, 20
    h = c.stack_height(rows, pad=pad, gap=gap)
    inks = [c.ink(t, f)[1] for t, f in rows]
    assert h == pad * 2 + sum(inks) + gap
    # the drawn content leaves exactly `pad` above the first row and below the last
    y0 = 100
    first_top = y0 + pad
    last_bottom = y0 + pad + inks[0] + gap + inks[1]
    assert abs((first_top - y0) - ((y0 + h) - last_bottom)) < 1e-6


def test_fit_font_shrinks_to_fit_and_respects_floor():
    from vh.cardkit import Card
    c = Card(w=1080, h=600)
    long = "아주 긴 제목 문구가 한 줄에 다 들어가야 하는 경우"
    f = c.fit_font(long, 700, 60)
    assert c.ink(long, f)[0] <= 700 and f.size < 60            # shrank to fit
    assert c.fit_font(long, 10, 60, floor=24).size == 24        # floor honoured
    short = c.fit_font("무게", 700, 40)
    assert short.size == 40                                     # already fits: unchanged


# ── reference-video window (feedback 49f3ca81469d) ──────────────────────────

def test_window_snaps_to_even_and_records_rect(capsys):
    """Odd sizes break h264_nvenc (exit 234), so they're snapped down + reported."""
    from vh.cardkit import Card
    c = Card(w=1080, h=1920)
    bottom = c.window("g_hook", 61, 321, 961, 541)
    assert c.windows["g_hook"] == [60, 320, 960, 540]
    assert bottom == 320 + 540
    assert "snapped to even" in capsys.readouterr().out


def test_window_sidecar_written_on_save(tmp_path):
    """The assembler reads coordinates from the sidecar, never from code."""
    import json
    from vh.cardkit import Card
    c = Card(w=1080, h=1920)
    c.window("g_hook", 60, 320, 960, 540)
    c.save(tmp_path / "g_hook.png")
    sc = tmp_path / "g_hook.windows.json"
    assert sc.exists() and json.loads(sc.read_text()) == {"g_hook": [60, 320, 960, 540]}


def test_no_sidecar_when_card_has_no_window(tmp_path):
    from vh.cardkit import Card
    c = Card(w=1080, h=600)
    c.text((540, 300), "제목", c.fb(60), c.FG)
    c.save(tmp_path / "plain.png")
    assert not (tmp_path / "plain.windows.json").exists()
