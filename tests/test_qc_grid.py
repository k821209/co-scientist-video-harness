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


# ── lint_vo: catch TTS mis-readings BEFORE synthesis (feedback 4c9a7f46336f) ─

def test_lint_vo_flags_digits_and_native_counter():
    """'7번' is read as '일곱 번' — and whisper transcribes it back as '7번',
    so only a pre-synthesis check can see it."""
    kinds = {w["kind"] for w in qc.lint_vo("등번호는 7번입니다")}
    assert kinds == {"native_counter", "digits_in_vo"}
    got = [w for w in qc.lint_vo("등번호는 7번입니다") if w["kind"] == "native_counter"][0]
    assert got["match"] == "7번" and "칠 번" in got["note"]     # prescription included


def test_lint_vo_flags_big_arabic_figures():
    w = qc.lint_vo("네이버 매출은 13조 원입니다")
    assert [x["kind"] for x in w] == ["digits_in_vo"]
    assert "Hangul" in w[0]["note"]


def test_lint_vo_flags_ambiguous_day_even_in_hangul():
    """십일일/이십일일 are mis-HEARD however they're spelled — drop the day."""
    w = qc.lint_vo("십이월 십일일에 개봉합니다")
    assert [x["kind"] for x in w] == ["ambiguous_day"]
    assert "card only" in w[0]["note"]
    assert qc.lint_vo("이십일일에 시작합니다")[0]["kind"] == "ambiguous_day"


def test_lint_vo_clean_for_correct_forms():
    for good in ["십삼조 천칠백억 원인데", "칠 번 등번호를 받았습니다",
                 "올해 여름은 길었습니다", "십이월에 개봉합니다"]:
        assert qc.lint_vo(good) == [], good


def test_lint_vo_allow_digits_switch():
    """allow_digits silences the blanket digit warning; the counter rule stays,
    since it has its own prescription (2026년 → '이천이십육 년')."""
    kinds = {w["kind"] for w in qc.lint_vo("2026년 기준", allow_digits=True)}
    assert kinds == {"native_counter"}                  # digit warning gone
    assert "digits_in_vo" in {w["kind"] for w in qc.lint_vo("2026년 기준")}
    assert qc.lint_vo("총 3개 지역", allow_digits=True)[0]["kind"] == "native_counter"


# ── fold: decimals + native numerals (feedback 4fcccd3197a8) ────────────────

def test_fold_decimal_point():
    f = qc._fold_numerals
    assert "7.7" in f("칠 점 칠 퍼센트")
    assert "45.4" in f("사십오 점 사 퍼센트")
    assert "8.04" in f("팔 점 영사 퍼센트")      # fraction is a DIGIT SEQUENCE
    assert "29.7" in f("이십구 점 칠 도")


def test_fold_native_numerals_with_counters():
    f = qc._fold_numerals
    assert "8번째" in f("여덟 번째")
    assert "9시" in f("아홉 시")
    assert "4개" in f("네 개")
    assert "2배" in f("두 배")


def test_fold_single_sino_digit_before_a_counter():
    assert "6분" in qc._fold_numerals("육 분").replace(" ", "")


def test_fold_leaves_ordinary_words_alone_still():
    f = qc._fold_numerals
    for w in ["이유가 있다", "네이버는", "사과와 배", "구글", "일본 정부", "세계",
              "점점 늘었다"]:
        assert f(w) == w, w


def test_narration_match_case_folds_latin(monkeypatch):
    """A script's "ETF" vs whisper's "etf" is not a mismatch."""
    from vh.steps import transcribe as tr

    class W:
        def __init__(self, t): self.text = t
    monkeypatch.setattr(tr, "transcribe",
                        lambda p, lang=None, prompt=None: [W("etf"), W("수익률은"), W("7.7%")])
    r = qc.narration_match("v.mp4", "ETF 수익률은 칠 점 칠 퍼센트")
    assert r["ratio"] > 0.9
