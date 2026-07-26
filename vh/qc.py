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
    subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", "-i", src,
                    "-vf", f"fps=1/{every},scale={width}:-1,tile={cols}x{rows}",
                    "-frames:v", "1", out_png], check=True)
    return out_png


def narration_match(video: str, script: str, language: str = "ko") -> dict:
    """Transcribe `video` and compare to the script it should be reading.

    Returns {ratio, words, transcript}. Compare on alphanumerics only so spacing
    and punctuation don't drown the signal."""
    from .steps.transcribe import transcribe

    words = transcribe(video, language)
    heard = " ".join(w.text for w in words)
    norm = lambda s: "".join(c for c in s if c.isalnum())          # noqa: E731
    ratio = difflib.SequenceMatcher(None, norm(script), norm(heard)).ratio()
    return {"ratio": ratio, "words": len(words), "transcript": heard}
