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


# ── sheet must be anchored at t=0 (feedback abd6d722a684) ───────────────────

def test_frames_are_capped_to_one_grid(monkeypatch):
    """More samples than tiles must be capped BEFORE tile, so exactly one sheet
    is produced and it starts at the first sample — not the last group."""
    seen = {}
    monkeypatch.setattr(qc, "_duration", lambda src: 86.0)
    def fake_run(args, **k):
        seen["vf"] = args[args.index("-vf") + 1]
        return None
    monkeypatch.setattr(qc.subprocess, "run", fake_run)
    qc.contact_sheet("in.mp4", "out.png", every=6, cols=5, rows=3)   # 15 tiles, 15 frames
    vf = seen["vf"]
    assert "trim=end_frame=15" in vf                       # capped to exactly one grid
    assert vf.index("trim=end_frame=") < vf.index("tile=")  # cap BEFORE tiling


def test_undersized_grid_warns_about_truncation(monkeypatch, capsys):
    monkeypatch.setattr(qc, "_duration", lambda src: 86.0)
    monkeypatch.setattr(qc.subprocess, "run", lambda *a, **k: None)
    qc.contact_sheet("in.mp4", "out.png", every=6, cols=2, rows=1)   # 15 needed, 2 shown
    msg = capsys.readouterr().out
    assert "do not fit" in msg and "FIRST 2" in msg          # loud, not silent


# ── narration_drift: reference audio vs regenerated output (534950300c1f) ───

def _drift(monkeypatch, ref_text, out_text, **kw):
    """Stub the ASR: first call returns the reference words, second the output."""
    from vh.steps import transcribe as tr

    class W:
        def __init__(self, t): self.text = t
    texts = [ref_text, out_text]
    monkeypatch.setattr(tr, "transcribe",
                        lambda path, lang=None, prompt=None: [W(t) for t in texts.pop(0).split()])
    return qc.narration_drift("ref.wav", "out.mp4", **kw)


def test_drift_flags_a_changed_date(monkeypatch):
    """The real failure: a generative lip-sync changed 7월 14일 → 10월 14일."""
    r = _drift(monkeypatch, "이란은 7월 14일 핵 역량을 공개했다",
                            "이란은 10월 14일 핵 영향을 공개했다")
    assert r["ok"] is False
    joined = " ".join(d["reference"] + "→" + d["output"] for d in r["drift"])
    assert "7월" in joined and "10월" in joined       # the date drift is reported
    assert any("역량" in d["reference"] for d in r["drift"])


def test_identical_speech_is_clean(monkeypatch):
    same = "이란은 어제 핵 시설을 공개했다"
    r = _drift(monkeypatch, same, same)
    assert r["ok"] is True and r["drift"] == [] and r["ratio"] == 1.0


def test_shared_asr_error_is_not_drift(monkeypatch):
    """Whisper mis-hearing the SAME way in both transcripts must not fire —
    that's the whole point of comparing two transcripts."""
    r = _drift(monkeypatch, "대일한 정책은 유지된다", "대일한 정책은 유지된다")
    assert r["ok"] is True and r["drift"] == []
