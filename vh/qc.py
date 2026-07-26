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
"""
from __future__ import annotations

import difflib
import math
import subprocess

from . import config


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
        if saw_unit or follows_big or digits > 1:
            out.append(str(total + cur))
            i = j
            if follows_big:                 # magnitude word stays as text
                out.append(s[i])
                i += 1
        else:                                # not a number — copy verbatim
            out.append(s[i:j])
            i = j
    return "".join(out)


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
        return "".join(c for c in s if c.isalnum())

    ratio = difflib.SequenceMatcher(None, norm(script), norm(heard)).ratio()
    return {"ratio": ratio, "words": len(words), "transcript": heard}
