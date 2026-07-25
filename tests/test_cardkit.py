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
