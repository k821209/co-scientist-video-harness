"""Output checks — because "the render succeeded" says nothing about the result.

Every bad Short shipped so far rendered cleanly and exited 0: a clip in-point that
landed on a piano cutaway, a divider line through a price, captions missing during
TTS pauses. ffmpeg cannot see any of that. These two checks catch most of it in
seconds, and both are meant to be *read*, not just run:

- `contact_sheet()` → one PNG of the whole video on a grid. Open it. Every frame
  that ships is on it.
- `narration_match()` → transcribe the finished file and diff it against the
  script you meant to say. Catches a truncated VO, a mis-muxed audio track, or a
  beat whose narration never made it in. ≥0.93 is normal for Korean TTS
  (proper nouns always mis-transcribe); a sudden drop means something is wrong.
- `narration_drift()` → transcribe a REFERENCE audio and the OUTPUT and diff
  those two. For stages that regenerate the voice (generative lip-sync, voice
  conversion), where the words themselves can change — a wrong date reads as a
  clean render. Comparing two transcripts cancels the ASR's own mistakes.
"""
from __future__ import annotations

import difflib
import math
import re
import subprocess

from . import config


# ── pre-synthesis VO script lint ────────────────────────────────────────────
#
# narration_match is an AFTER-the-fact net with holes: whisper normalises what it
# hears, so a line read with the WRONG number word transcribes back to the digits
# you wrote and the check passes. "7번" read as "일곱 번" (native Korean numeral
# instead of Sino-Korean) transcribed as "7번" and looked perfect — a listener
# caught it. Reading the script BEFORE synthesis is the only systematic defence.

# Counters that pull the NATIVE reading out of a digit ("7번" → "일곱 번").
_COUNTERS = "번|년|위|명|개|회|주|층|살|권|장|마리|대|잔|칸|줄"
_COUNTER_RX = re.compile(rf"(\d+)\s*({_COUNTERS})")
_DIGIT_RX = re.compile(r"\d")
# Day-of-month forms that stay ambiguous even spelled out: 십일일(11th) is heard
# as 12일, 이십일일(21st) as 20일. Spelling doesn't fix these — drop the day.
_AMBIGUOUS_DAY_RX = re.compile(r"(삼?십일일|이십일일)")


def lint_vo(text: str, *, allow_digits: bool = False) -> list[dict]:
    """Check a VO line BEFORE synthesis. Returns [{kind, match, note}] — warnings
    only, never raises: a year passed straight through to a card, or a deliberate
    digit, is legitimate.

    Catches the three ways Korean edge-tts mis-reads a script, each with its own
    prescription:
      - `digits_in_vo` — any Arabic numeral: big figures get chopped
        ("13조" → "1, 3조"). Write numbers in Hangul in the spoken line.
      - `native_counter` — digits + a counter read as a native numeral
        ("7번" → "일곱 번"). Write the Sino-Korean form ("칠 번").
      - `ambiguous_day` — 십일일 / 이십일일 / 삼십일일 are mis-HEARD however you
        spell them. Take the day out of the VO and leave it on the card only.
    """
    out: list[dict] = []
    t = text or ""
    for m in _COUNTER_RX.finditer(t):
        out.append({
            "kind": "native_counter", "match": m.group(0),
            "note": f"'{m.group(0)}' is read with the NATIVE numeral "
                    f"(e.g. 7번 → '일곱 번'). Write the Sino-Korean form in the "
                    f"spoken line ('칠 번') and keep the digits on the card.",
        })
    if not allow_digits and _DIGIT_RX.search(t):
        out.append({
            "kind": "digits_in_vo", "match": _DIGIT_RX.search(t).group(0),
            "note": "Arabic numerals in a Korean VO get mis-chunked "
                    "('13조' → '1, 3조'). Spell the number out in Hangul in the "
                    "spoken line; keep the digits on the card.",
        })
    for m in _AMBIGUOUS_DAY_RX.finditer(t):
        out.append({
            "kind": "ambiguous_day", "match": m.group(0),
            "note": f"'{m.group(0)}' is mis-HEARD (11일 sounds like 12일, "
                    f"21일 like 20일) no matter how it is spelled — remove the day "
                    f"from the VO and show it on the card only.",
        })
    return out


def _duration(src: str) -> float:
    return float(subprocess.check_output(
        [config.FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(src)]).strip())


def contact_sheet(src: str, out_png: str, *, every: float = 8.0,
                  cols: int | None = None, rows: int | None = None,
                  width: int = 300) -> str:
    """Tile one frame every `every` seconds into a single grid PNG.

    Leave `cols`/`rows` unset and the grid is sized to the video so no frame is
    dropped (a cut-off sheet reads as "I checked it" when you didn't). Give one
    to fix that dimension; give both to force an exact grid (may truncate)."""
    tiles = max(1, math.ceil(_duration(src) / every))
    if cols and rows:
        pass                                   # explicit grid — caller's choice
    elif cols:
        rows = math.ceil(tiles / cols)
    elif rows:
        cols = math.ceil(tiles / rows)
    else:                                      # auto: near-square, covers all tiles
        cols = math.ceil(math.sqrt(tiles))
        rows = math.ceil(tiles / cols)

    # Cap the sampled frames at exactly one grid BEFORE tiling. Without this,
    # more frames than fit produced several tile outputs and which one landed in
    # the file depended on ffmpeg's -frames:v/image2 behaviour — on some builds
    # the LAST group won, so the sheet silently began mid-video (a cold open
    # missing from the sheet reads as "I checked it"). One group in → one sheet
    # out, always anchored at t=0.
    cap = cols * rows
    if tiles > cap:
        print(f"[qc] contact_sheet: {tiles} frames at every={every}s do not fit "
              f"{cols}x{rows}={cap}; showing the FIRST {cap} "
              f"(~{cap * every:.0f}s of {_duration(src):.0f}s). "
              f"Raise the grid or `every` to cover the whole video.")
    subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", "-i", src,
                    "-vf", (f"fps=1/{every},scale={width}:-1,"
                            f"trim=end_frame={cap},tile={cols}x{rows}"),
                    "-frames:v", "1", out_png], check=True)
    return out_png


_KO_DIGIT = {"영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4,
             "오": 5, "육": 6, "륙": 6, "칠": 7, "팔": 8, "구": 9}
_KO_SMALL = {"십": 10, "백": 100, "천": 1000}       # positional, within a group
_KO_BIG = {"만": 10_000, "억": 10**8, "조": 10**12}  # magnitude words, kept as text

# NATIVE Korean numerals. A VO script is forced to use these — edge-tts reads the
# Sino-Korean form wrongly before a counter ("여덟 번째", "아홉 시") — while whisper
# writes them back as digits, so they must fold too. Only folded when a counter
# follows, so ordinary words ("네 가지 이유" is fine, but "네" alone isn't a number)
# don't get mangled.
_KO_NATIVE = {
    "한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4,
    "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10,
    "열한": 11, "열두": 12, "열세": 13, "열네": 14, "스물": 20, "스무": 20,
    "서른": 30, "마흔": 40, "쉰": 50, "예순": 60, "일흔": 70, "여든": 80, "아흔": 90,
    "첫": 1,
}
_NATIVE_COUNTERS = ("번째", "번", "시", "개", "배", "명", "마리", "가지", "살", "쌍",
                    "권", "잔", "칸", "줄", "군데", "곳", "달")
_NATIVE_RX = re.compile(
    r"\b(" + "|".join(sorted(_KO_NATIVE, key=len, reverse=True)) + r")\s*"
    r"(" + "|".join(sorted(_NATIVE_COUNTERS, key=len, reverse=True)) + r")")

# Words that, when they follow a number, prove it IS a number — so even a single
# digit syllable gets folded ("육 분" → 6분, "칠 점 칠" → 7.7). "점" is in here
# because a decimal point is exactly this case.
_SINO_RUN = "[영공일이삼사오육륙칠팔구십백천만억조]"
_DECIMAL_RX = re.compile(rf"({_SINO_RUN}+)\s*점\s*({_SINO_RUN}+)")


def _sino_value(run: str) -> int:
    """Sino-Korean number word → int ("사십오" → 45, "천칠백" → 1700)."""
    total = cur = 0
    for ch in run:
        if ch in _KO_DIGIT:
            cur = _KO_DIGIT[ch]
        elif ch in _KO_SMALL:
            total += (cur or 1) * _KO_SMALL[ch]
            cur = 0
        elif ch in _KO_BIG:
            total = (total + cur) * _KO_BIG[ch]
            cur = 0
    return total + cur


def _sino_digits(run: str) -> str:
    """Fraction digits are spoken one by one ("영사" → "04")."""
    if all(ch in _KO_DIGIT for ch in run):
        return "".join(str(_KO_DIGIT[ch]) for ch in run)
    return str(_sino_value(run))          # "십오" after a point → 15 (rare)


_FOLD_FOLLOWERS = ("점", "분", "초", "시간", "시", "번째", "번", "개", "명", "마리",
                   "퍼센트", "프로", "원", "달러", "엔", "위안", "년", "월", "일",
                   "위", "층", "배", "쌍", "도", "회", "주", "권", "미터", "킬로",
                   "그램", "리터", "톤", "평", "억", "조", "만")


def _fold_numerals(s: str) -> str:
    """Fold Korean number words into digits so a spoken-form script compares
    cleanly with a digit-form transcript.

    Voice scripts SHOULD spell big numbers out in Hangul — edge-tts mis-chunks
    Arabic figures ("13조 1700억" → "일 ,삼조…") — but whisper transcribes them
    back as digits, which dragged narration_match below its 0.93 line on a
    perfectly good episode. Each run of number words is converted within its
    magnitude group ("천칠백" → 1700), while the magnitude word itself stays as
    text ("조"/"억"/"만") because the transcript keeps it too.
    """
    # Unit symbols: a VO script spells them out ("퍼센트") and whisper writes the
    # symbol ("%"), which the alphanumeric filter then drops entirely — pure
    # notation noise in the ratio.
    for sym, word in (("%", "퍼센트"), ("℃", "도"), ("°C", "도"), ("°", "도")):
        s = s.replace(sym, word)
    # Native numerals next (they are words, not syllable arithmetic), only when a
    # counter follows: "여덟 번째" → "8번째", "아홉 시" → "9시".
    s = _NATIVE_RX.sub(lambda m: f"{_KO_NATIVE[m.group(1)]}{m.group(2)}", s)
    # Then decimals, before the arithmetic fold: the FRACTION is read as a digit
    # sequence, not a number ("팔 점 영사" is 8.04, not 8.4), so it can't go
    # through the same path as the integer part.
    s = _DECIMAL_RX.sub(
        lambda m: f"{_sino_value(m.group(1))}.{_sino_digits(m.group(2))}", s)

    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        if s[i] not in _KO_DIGIT and s[i] not in _KO_SMALL:
            out.append(s[i])
            i += 1
            continue
        # Scan the run of number words, but only FOLD it when it really reads as
        # a number: it must contain a positional/magnitude word (십·백·천·만·억·조)
        # or be a multi-digit sequence. A bare digit syllable inside an ordinary
        # word ("네이버", "이유") must pass through untouched.
        j = i
        total, cur, saw_unit, digits = 0, 0, False, 0
        while j < n and (s[j] in _KO_DIGIT or s[j] in _KO_SMALL):
            if s[j] in _KO_DIGIT:
                cur = _KO_DIGIT[s[j]]
                digits += 1
            else:
                total += (cur or 1) * _KO_SMALL[s[j]]
                cur = 0
                saw_unit = True
            j += 1
        follows_big = j < n and s[j] in _KO_BIG
        # A SINGLE digit syllable is only a number when something numeric follows
        # ("육 분" → 6분, "칠 점 칠" → 7.7); bare 이/사/구 inside ordinary words
        # ("이유", "사과", "구글") must survive untouched.
        tail = s[j:j + 6].lstrip()
        follows_counter = any(tail.startswith(c) for c in _FOLD_FOLLOWERS)
        if saw_unit or follows_big or follows_counter or digits > 1:
            out.append(str(total + cur))
            i = j
            if follows_big:                 # magnitude word stays as text
                out.append(s[i])
                i += 1
        else:                                # not a number — copy verbatim
            out.append(s[i:j])
            i = j
    folded = "".join(out)
    # Decimal point: "칠 점 칠" reads as 7.7 — whisper writes "7.7". Do this after
    # the digit fold so both sides are already numeric.
    folded = re.sub(r"(\d)\s*점\s*(\d)", r"\1.\2", folded)
    return folded


def narration_drift(reference: str, output: str, language: str = "ko",
                    *, fold_numerals: bool = True, min_ratio: float = 0.97) -> dict:
    """Compare what the REFERENCE audio says with what the OUTPUT says.

    Use this whenever a stage REGENERATES the voice instead of copying it — a
    generative lip-sync (LTX and friends), a voice conversion, a re-dub. Those
    stages take your TTS as a style reference and synthesize new speech, so the
    words can come out different: a real episode had "7월 14일" become
    "10월 14일" (a wrong date — publishable misinformation), "핵 역량" become
    "핵 영향", "합동 공습" become "협동 공습" — in 3 of 7 clips.

    `narration_match(output, script)` cannot catch this: a mismatch there is
    indistinguishable from whisper mis-hearing the audio. Transcribing BOTH with
    the same model cancels the ASR's own errors — whatever differs is drift the
    generator introduced.

    Returns {ratio, ok, drift: [{reference, output}], reference_transcript,
    output_transcript}. `ok` is ratio >= min_ratio AND no drift spans. Run it per
    clip BEFORE assembly: re-rendering one clip is far cheaper than redoing the
    episode.
    """
    from .steps.transcribe import transcribe

    def say(path: str) -> str:
        return " ".join(w.text for w in transcribe(path, language))

    ref_text, out_text = say(reference), say(output)

    def toks(s: str) -> list[str]:
        if fold_numerals:
            s = _fold_numerals(s)
        return ["".join(c for c in t.lower() if c.isalnum()) for t in s.split() if t.strip()]

    a, b = toks(ref_text), toks(out_text)
    sm = difflib.SequenceMatcher(None, a, b)
    drift = [{"reference": " ".join(a[i1:i2]), "output": " ".join(b[j1:j2])}
             for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
    ratio = sm.ratio()
    return {"ratio": ratio, "ok": ratio >= min_ratio and not drift, "drift": drift,
            "reference_transcript": ref_text, "output_transcript": out_text}


def narration_match(video: str, script: str, language: str = "ko",
                    *, fold_numerals: bool = True) -> dict:
    """Transcribe `video` and compare to the script it should be reading.

    Returns {ratio, words, transcript}. Compare on alphanumerics only so spacing
    and punctuation don't drown the signal. With `fold_numerals` (default) the
    Korean spelled-out numbers a voice script should use are folded toward
    digits before comparing, so following that rule doesn't cost ~0.10 of ratio
    against whisper's digit transcript. ≥0.93 is normal for Korean TTS."""
    from .steps.transcribe import transcribe

    words = transcribe(video, language)
    heard = " ".join(w.text for w in words)

    def norm(s: str) -> str:
        if fold_numerals:
            s = _fold_numerals(s)
        # Case-fold Latin: a script says "ETF", whisper writes "etf" (and the
        # Hangul spelling "이티에프" is a separate, unfoldable mismatch — see the
        # docstring note).
        return "".join(c for c in s.lower() if c.isalnum())

    ratio = difflib.SequenceMatcher(None, norm(script), norm(heard)).ratio()
    return {"ratio": ratio, "words": len(words), "transcript": heard}
