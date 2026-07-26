"""contact_sheet grid auto-sizing (feedback 8587d866794a item 3)."""
from vh import qc


def _tile_arg(monkeypatch, dur, **kw):
    captured = {}
    monkeypatch.setattr(qc, "_duration", lambda src: dur)
    def fake_run(args, **k):
        vf = args[args.index("-vf") + 1]
        captured["tile"] = [p for p in vf.split(",") if p.startswith("tile=")][0]
        return None
    monkeypatch.setattr(qc.subprocess, "run", fake_run)
    qc.contact_sheet("in.mp4", "out.png", **kw)
    c, r = captured["tile"].removeprefix("tile=").split("x")
    return int(c), int(r)


def test_auto_grid_covers_all_tiles(monkeypatch):
    # 98s / every 8s = 13 tiles → grid must hold >= 13, near-square
    cols, rows = _tile_arg(monkeypatch, 98.0, every=8.0)
    assert cols * rows >= 13 and abs(cols - rows) <= 1


def test_partial_grid_fills_missing_dim(monkeypatch):
    cols, rows = _tile_arg(monkeypatch, 98.0, every=8.0, cols=7)
    assert cols == 7 and cols * rows >= 13


def test_explicit_grid_is_respected(monkeypatch):
    cols, rows = _tile_arg(monkeypatch, 98.0, every=8.0, cols=7, rows=2)
    assert (cols, rows) == (7, 2)


# ── narration_match numeral folding (feedback 7b0e9a266d7c item 3-b) ────────

def test_fold_numerals_converges_spoken_and_digit_forms():
    from vh.qc import _fold_numerals
    import difflib
    script = "약 십삼조 천칠백억 원인데"      # what the voice script must say
    heard = "약 13조 1700억 원인데"            # what whisper writes down
    norm = lambda s: "".join(c for c in _fold_numerals(s) if c.isalnum())  # noqa: E731
    raw = difflib.SequenceMatcher(None,
                                  "".join(c for c in script if c.isalnum()),
                                  "".join(c for c in heard if c.isalnum())).ratio()
    folded = difflib.SequenceMatcher(None, norm(script), norm(heard)).ratio()
    assert folded > raw and folded >= 0.93     # folding lifts it over the QC line


def test_fold_numerals_leaves_ordinary_words_alone():
    """Digit syllables inside normal words must NOT be folded (네이버 → 네2버)."""
    from vh.qc import _fold_numerals
    for word in ["네이버는 판다 가족", "이유가 있다", "사과와 배", "일본 정부", "구글"]:
        assert _fold_numerals(word) == word
    # but a real numeral run is folded, magnitude word kept as text
    assert _fold_numerals("십사조 육천억") == "14조 6000억"
