"""Caption rendering.

Builds an ASS subtitle file and burns it in with ffmpeg. Two looks:
  - "word": CapCut/SubMagic-style — one line at a time, the currently spoken
    word popped in an accent colour.
  - "line": clean lower-third line captions.
`vad`-grouped words come from steps.transcribe.
"""
from __future__ import annotations

from pathlib import Path

from .. import config
from ..probe import run
from .transcribe import Word


def _ass_ts(sec: float) -> str:
    sec = max(sec, 0.0)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs == 100:
        cs = 0
        s += 1
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


# ASS colours are &HAABBGGRR (alpha,blue,green,red), AA=00 opaque.
_WHITE = "&H00FFFFFF"
_ACCENT = "&H0000E5FF"   # amber/yellow highlight
_BLACK = "&H00000000"


def _group_lines(words: list[Word], max_words: int) -> list[list[Word]]:
    """Group words into caption lines, breaking at sentence-ending punctuation so
    a period never sits mid-line with the next sentence's opening words."""
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words or w.text.strip().endswith((".", "?", "!", "…")):
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    return lines


def _escape(t: str) -> str:
    return t.replace("{", "(").replace("}", ")").replace("\n", " ")


def build_ass(
    words: list[Word],
    width: int,
    height: int,
    style: str = "word",
    max_words: int = 4,
    font: str | None = None,
    chapters: list | None = None,
    title_hold: float = 3.0,
) -> str:
    font = font or config.CAPTION_FONT
    fontsize = int(height * (0.055 if style == "line" else 0.075))
    margin_v = int(height * (0.08 if style == "line" else 0.16))
    outline = max(2, int(fontsize * 0.08))

    title_fs = int(height * 0.062)
    title_mv = int(height * 0.10)   # from top (Alignment 8 = top-center)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Base,{font},{fontsize},{_WHITE},{_WHITE},{_BLACK},&H64000000,-1,0,0,0,100,100,0,0,1,{outline},2,2,60,60,{margin_v},1
Style: Title,{font},{title_fs},{_WHITE},{_WHITE},{_BLACK},&HA0000000,-1,0,0,0,100,100,0,0,3,4,1,8,80,80,{title_mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []

    # Title cards at each chapter boundary (fade in/out, held title_hold sec).
    for ch in (chapters or []):
        start = max(0.0, float(ch.start))
        end = start + title_hold
        text = _escape(str(ch.title))
        events.append(
            f"Dialogue: 1,{_ass_ts(start)},{_ass_ts(end)},Title,,0,0,0,,"
            f"{{\\fad(350,350)}}{text}"
        )
    events += _caption_events(words, max_words, "Base", mode=style)
    return header + "\n".join(events) + "\n"


_HOLD_MAX = 1.6   # cap on how long a caption lingers into a pause (seconds)


def _caption_events(words: list[Word], max_words: int, style_name: str,
                    mode: str = "word", hold_through_pauses: bool = True) -> list[str]:
    """Caption Dialogue lines (word-pop or line), reusable across layouts.

    A global `cursor` forces every event to start at/after the previous event's
    end, so overlapping/backwards Whisper word timestamps can never make two
    caption events show at once (which libass would stack into multiple lines
    that overflow the caption zone).

    hold_through_pauses (default on): extend the last caption of each line to the
    NEXT line's start (capped at _HOLD_MAX) so a caption stays on screen through
    inter-sentence silence instead of blinking off. Without it, edge-tts's ~1s
    sentence gaps left the frame subtitle-less ~20% of the time."""
    events: list[str] = []
    cursor = 0.0
    lines = [ln for ln in _group_lines(words, max_words) if ln]
    for li, line in enumerate(lines):
        next_line_start = lines[li + 1][0].start if li + 1 < len(lines) else None
        hold_to = (min(next_line_start, line[-1].end + _HOLD_MAX)
                   if (hold_through_pauses and next_line_start is not None) else None)
        if mode == "line":
            start = max(line[0].start, cursor)
            end = max(line[-1].end, start + 0.3)
            if hold_to is not None:
                end = max(end, hold_to)
            cursor = end
            text = _escape(" ".join(w.text for w in line))
            events.append(f"Dialogue: 0,{_ass_ts(start)},{_ass_ts(end)},{style_name},,0,0,0,,{text}")
        else:
            for i, w in enumerate(line):
                start = max(w.start, cursor)
                if i + 1 < len(line):
                    nxt = line[i + 1].start
                else:
                    nxt = hold_to if hold_to is not None else w.end   # linger through pause
                end = max(nxt, start + 0.12)   # strictly after `start`, never overlaps
                cursor = end
                parts = []
                for j, ww in enumerate(line):
                    tok = _escape(ww.text)
                    if j == i:
                        parts.append(f"{{\\c{_ACCENT}\\fscx112\\fscy112}}{tok}{{\\c{_WHITE}\\fscx100\\fscy100}}")
                    else:
                        parts.append(tok)
                events.append(f"Dialogue: 0,{_ass_ts(start)},{_ass_ts(end)},{style_name},,0,0,0,,{' '.join(parts)}")
    return events


def build_boxed_ass(
    words: list[Word],
    canvas_w: int,
    canvas_h: int,
    video_top: int,
    video_bottom: int,
    chapters: list | None = None,
    video_title: str | None = None,
    style: str = "word",
    max_words: int = 4,
    font: str | None = None,
    title_end: float | None = None,
    hold_through_pauses: bool = True,
) -> str:
    """ASS for the boxed vertical layout: header in the top band (persistent
    section title, or a fixed video title), captions in the bottom band.

    title_end: when to end the fixed `video_title` header (and the last
    chapter). Defaults to `words[-1].end + 3.0` — a 3s tail so the header
    doesn't vanish the instant narration stops. Pass an explicit value to retire
    the header earlier (e.g. before an end card)."""
    font = font or config.CAPTION_FONT
    top_band = max(1, video_top)
    bottom_band = max(1, canvas_h - video_bottom)

    head_fs = int(min(top_band * 0.42, canvas_w * 0.060))
    head_mv = max(16, (top_band - head_fs) // 2)          # from top (align 8)
    head_outline = max(2, int(head_fs * 0.07))
    cap_fs = int(canvas_w * 0.055)
    cap_mv = max(16, (bottom_band - cap_fs) // 2)          # from bottom (align 2)
    cap_outline = max(2, int(cap_fs * 0.08))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {canvas_w}
PlayResY: {canvas_h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{cap_fs},{_WHITE},{_WHITE},{_BLACK},&H64000000,-1,0,0,0,100,100,0,0,1,{cap_outline},1,2,60,60,{cap_mv},1
Style: Head,{font},{head_fs},{_ACCENT},{_WHITE},{_BLACK},&H00000000,-1,0,0,0,100,100,0,0,1,{head_outline},0,8,60,60,{head_mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    end_all = title_end if title_end is not None else ((words[-1].end + 3.0) if words else 3600.0)
    chs = sorted(chapters or [], key=lambda c: float(c.start))

    if chs:   # persistent current-section header
        for i, ch in enumerate(chs):
            s = max(0.0, float(ch.start))
            e = float(chs[i + 1].start) if i + 1 < len(chs) else end_all
            events.append(
                f"Dialogue: 0,{_ass_ts(s)},{_ass_ts(e)},Head,,0,0,0,,"
                f"{{\\fad(300,200)}}{_escape(ch.title)}"
            )
    elif video_title:   # fixed title for the whole clip
        events.append(
            f"Dialogue: 0,{_ass_ts(0.0)},{_ass_ts(end_all)},Head,,0,0,0,,{_escape(video_title)}"
        )

    events += _caption_events(words, max_words, "Cap", mode=style,
                              hold_through_pauses=hold_through_pauses)
    return header + "\n".join(events) + "\n"


def _fg_escape(p: str) -> str:
    """Escape a path for use inside a filtergraph option value."""
    return str(p).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def burn(src: str, dst: str, ass_path: str, fontsdir: str | None = None) -> str:
    """Burn an ASS file into the video (re-encode video, copy audio)."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    safe = _fg_escape(ass_path)
    fdir = fontsdir or config.CAPTION_FONTSDIR
    sub = f"subtitles='{safe}'"
    if fdir:
        sub += f":fontsdir='{_fg_escape(fdir)}'"
    from .. import remote
    remote.ffmpeg_run([
        config.FFMPEG, "-y", "-i", str(src),
        "-vf", sub,
        *config.encode_args(),
        str(dst),
    ], reads=[str(src), str(ass_path)], write=str(dst))
    return dst


def write_ass(content: str, path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")
    return path
