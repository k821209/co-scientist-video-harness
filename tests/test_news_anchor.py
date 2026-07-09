"""Robust shot-anchor matching for build_short (no ffmpeg needed)."""
import pytest
from vh.steps.news import _find_anchor
from vh.steps.transcribe import Word


def _w(*pairs):
    return [Word(i, i + 0.5, t) for i, t in enumerate(pairs)]


def test_single_token():
    assert _find_anchor(_w("10억", "광년"), "10억", 0) == 0


def test_multi_token_span():
    # align/edge-tts split "이 속도를" into two tokens — must still match at 0
    assert _find_anchor(_w("이", "속도를,"), "이 속도를", 0) == 0


def test_sequential_after_pos():
    # same token twice → the second search (pos past the first) finds the second
    ws = _w("6월", "말", "6월", "해제")
    assert _find_anchor(ws, "6월", 0) == 0
    assert _find_anchor(ws, "6월", 1) == 2


def test_missing_raises_with_context():
    with pytest.raises(ValueError, match="not found"):
        _find_anchor(_w("가", "나", "다"), "라라라", 0)
