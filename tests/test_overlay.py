"""Shared overlay provenance logic (no ffmpeg): per-shot credits vs ribbon."""
from vh.steps import news

BASE = "[Script Info]\n\n[Events]\n"
COMMON = dict(font="F", accent="&H0", eyebrow="E", source="S", card="C",
              off=6.0, total=8.0, card_in=6.5)


def _cred(a):
    return a.count(",Cred,,")


def test_per_shot_credits_merge_adjacent():
    a = news._overlay_layer(BASE, credits=["Reuters", "Reuters", "AFP"],
                            starts=[0, 2, 4, 8], **COMMON)
    assert _cred(a) == 2                      # two runs, not three


def test_ribbon_used_when_no_credits():
    a = news._overlay_layer(BASE, ribbon="AI 생성 이미지", **COMMON)
    assert _cred(a) == 1 and "AI 생성 이미지" in a


def test_asserts_nothing_by_default():
    # the safety fix: no ribbon + no credits → no provenance claim at all
    a = news._overlay_layer(BASE, **COMMON)
    assert _cred(a) == 0


def test_credits_take_precedence_over_ribbon():
    a = news._overlay_layer(BASE, ribbon="AI 생성 이미지",
                            credits=["Reuters", "Reuters"], starts=[0, 3, 8], **COMMON)
    assert "AI 생성 이미지" not in a and _cred(a) == 1


def test_disclosure_and_badge_optional():
    a = news._overlay_layer(BASE, disclosure="D", badge="B",
                            credits=[None, None, None], starts=[0, 2, 4, 8], **COMMON)
    assert ",Disc,," in a and ",Badge,," in a
