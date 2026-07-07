"""Interstitial title cards.

Instead of overlaying the section title on top of the video, splice a full-frame
title card in at each chapter boundary. `card_ass` designs one card; `render_card`
rasterises it; `build_with_interstitials` cuts the source at chapter boundaries,
interleaves cards, and re-times the word captions onto the lengthened timeline
so they still line up.
"""
from __future__ import annotations

from pathlib import Path

from .. import config
from ..probe import run, probe
from .caption import _ass_ts, _escape, _fg_escape, build_ass, write_ass, burn
from .transcribe import Word

_WHITE = "&H00FFFFFF"
_ACCENT = "&H0000E5FF"
_BLACK = "&H00000000"


def card_ass(text: str, index: int, total: int, w: int, h: int, dur: float,
             font: str | None = None) -> str:
    font = font or config.CAPTION_FONT
    ttl_fs = int(h * 0.075)
    num_fs = int(h * 0.030)
    ttl_out = max(2, int(ttl_fs * 0.05))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Num,{font},{num_fs},{_ACCENT},{_WHITE},{_BLACK},{_BLACK},-1,0,0,0,100,100,6,0,1,0,0,5,0,0,0,1
Style: Ttl,{font},{ttl_fs},{_WHITE},{_WHITE},{_BLACK},{_BLACK},-1,0,0,0,100,100,0,0,1,{ttl_out},0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    end = _ass_ts(dur)
    num = f"CHAPTER {index:02d}  ·  {total:02d}"
    # accent divider line drawn with an ASS drawing command
    line_w = int(w * 0.16)
    events = [
        f"Dialogue: 0,{_ass_ts(0)},{end},Num,,0,0,0,,{{\\fad(350,300)\\pos({w//2},{int(h*0.40)})}}{_escape(num)}",
        f"Dialogue: 0,{_ass_ts(0)},{end},Ttl,,0,0,0,,{{\\fad(350,300)\\pos({w//2},{int(h*0.52)})}}{_escape(text)}",
        f"Dialogue: 0,{_ass_ts(0)},{end},Num,,0,0,0,,{{\\fad(350,300)\\pos({w//2},{int(h*0.60)})\\p1}}m 0 0 l {line_w} 0 {line_w} 3 0 3{{\\p0}}",
    ]
    return header + "\n".join(events) + "\n"


def render_card(dst: str, text: str, index: int, total: int, w: int, h: int,
                dur: float = 1.8, fps: float = 60.0, bg_color: str = "0x0B0B14",
                fontsdir: str | None = None) -> str:
    """Render one title-card clip (solid bg + burned title + silent audio)."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    ass_path = str(Path(dst).with_suffix(".ass"))
    write_ass(card_ass(text, index, total, w, h, dur), ass_path)
    fdir = fontsdir or config.CAPTION_FONTSDIR
    sub = f"subtitles='{_fg_escape(ass_path)}'" + (f":fontsdir='{_fg_escape(fdir)}'" if fdir else "")
    from .. import remote
    remote.ffmpeg_run([
        config.FFMPEG, "-y",
        "-f", "lavfi", "-i", f"color=c={bg_color}:s={w}x{h}:r={fps}:d={dur}",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-t", f"{dur}", "-vf", sub,
        *config.video_args(), "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(dst),
    ], reads=[ass_path], write=str(dst))
    return dst


def _chapter_index(t: float, bounds: list[float]) -> int:
    idx = 0
    for i in range(len(bounds) - 1):
        if t >= bounds[i]:
            idx = i
    return idx


def _snap_to_pause(t: float, words: list[Word], window: float = 3.5) -> float:
    """Move a chapter boundary to a nearby speech pause so a title card never
    cuts a word mid-utterance. Picks the widest inter-word gap whose midpoint is
    within `window` of t; falls back to the nearest word end if speech is dense."""
    best = None  # (gap_size, midpoint)
    for i in range(len(words) - 1):
        g0, g1 = words[i].end, words[i + 1].start
        gap = g1 - g0
        if gap <= 0:
            continue
        mid = (g0 + g1) / 2.0
        if abs(mid - t) <= window:
            if best is None or gap > best[0] or (gap == best[0] and abs(mid - t) < abs(best[1] - t)):
                best = (gap, mid)
    if best is not None:
        return best[1]
    ends = [w.end for w in words]
    return min(ends, key=lambda x: abs(x - t)) if ends else t


def build_with_interstitials(
    src: str,
    dst: str,
    chapters: list,
    words: list[Word],
    card_dur: float = 1.8,
    style: str = "word",
    max_words: int = 5,
    font: str | None = None,
    bg_color: str = "0x0B0B14",
    workdir: str | None = None,
    caption_words: list | None = None,
    bounds: list | None = None,
) -> str:
    """Splice a full-frame title card in at each chapter boundary and re-time the
    word captions onto the lengthened timeline. One ffmpeg pass does the trims,
    the interleaved concat, and the caption burn.

    `caption_words` (already in the final, card-lengthened timeline) overrides the
    captions — used to burn a translated track (e.g. English dub subtitles).
    `bounds` (len == chapters+1, in src time) overrides the snapped chapter cuts —
    pass exact cut points when the caller already knows them (e.g. a speech-paced
    re-cut where each card lands on a known clip boundary)."""
    from .chapters import Chapter

    info = probe(src)
    W, H, fps, dur = info.width, info.height, info.fps or 30.0, info.duration
    work = Path(workdir or (Path(dst).parent / "_inter"))
    work.mkdir(parents=True, exist_ok=True)

    chs = sorted(chapters, key=lambda c: float(c.start))
    if not chs or float(chs[0].start) > 0.01:
        chs = [Chapter(0.0, chs[0].title if chs else "시작")] + chs
    n = len(chs)
    if bounds is not None:
        bounds = list(bounds)
    else:
        # Snap each interior boundary to a nearby speech pause so a card never cuts
        # a word. First boundary stays at 0; keep boundaries strictly increasing.
        bounds = [0.0]
        for c in chs[1:]:
            snapped = _snap_to_pause(float(c.start), words)
            bounds.append(max(snapped, bounds[-1] + 0.05))
        bounds.append(dur)

    # A) render one card per chapter
    cards = []
    for i, c in enumerate(chs):
        cp = str(work / f"card_{i:02d}.mp4")
        render_card(cp, c.title, i + 1, n, W, H, dur=card_dur, fps=fps, bg_color=bg_color)
        cards.append(cp)

    # B) captions built PER CHAPTER so no caption line straddles a card gap.
    #    Within a chapter every word shifts by the same (chapter_index+1)*card_dur.
    from .caption import _caption_events
    base = build_ass([], W, H, style=style, max_words=max_words, font=font)  # header only
    evs: list[str] = []
    if caption_words is not None:
        evs = _caption_events(list(caption_words), max_words, "Base", mode=style)
    else:
        for i in range(n):
            sh = (i + 1) * card_dur
            lo, hi = bounds[i], bounds[i + 1]
            # clamp each word inside its chapter so no caption bleeds onto a card
            wi = [Word(max(lo, w.start) + sh, min(hi - 0.01, w.end) + sh, w.text)
                  for w in words if lo <= w.start < hi]
            wi = [w for w in wi if w.end > w.start]
            evs += _caption_events(wi, max_words, "Base", mode=style)
    ass_path = str(work / "captions.ass")
    write_ass(base.rstrip() + "\n" + "\n".join(evs) + "\n", ass_path)

    # C) one ffmpeg pass: split -> trim segments -> interleave with cards -> burn
    inputs = ["-i", str(src)]
    for cp in cards:
        inputs += ["-i", cp]

    vsplit = "".join(f"[vin{i}]" for i in range(n))
    asplit = "".join(f"[ain{i}]" for i in range(n))
    parts = [f"[0:v]split={n}{vsplit}", f"[0:a]asplit={n}{asplit}"]
    for i in range(n):
        s, e = bounds[i], bounds[i + 1]
        parts.append(f"[vin{i}]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS,"
                     f"fps={fps},scale={W}:{H},setsar=1,format=yuv420p[v{i}]")
        parts.append(f"[ain{i}]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS,"
                     f"aformat=sample_rates=48000:channel_layouts=stereo[a{i}]")
    for i in range(n):
        k = i + 1
        parts.append(f"[{k}:v]fps={fps},scale={W}:{H},setsar=1,format=yuv420p[cv{i}]")
        parts.append(f"[{k}:a]aformat=sample_rates=48000:channel_layouts=stereo[ca{i}]")
    order = "".join(f"[cv{i}][ca{i}][v{i}][a{i}]" for i in range(n))
    parts.append(f"{order}concat=n={2*n}:v=1:a=1[ccv][cca]")
    fdir = config.CAPTION_FONTSDIR
    subf = f"[ccv]subtitles='{_fg_escape(ass_path)}'" + (f":fontsdir='{_fg_escape(fdir)}'" if fdir else "") + "[outv]"
    parts.append(subf)
    filt = ";".join(parts)

    from .. import remote
    remote.ffmpeg_run([
        config.FFMPEG, "-y", *inputs,
        "-filter_complex", filt,
        "-map", "[outv]", "-map", "[cca]",
        *config.encode_args(),
        str(dst),
    ], reads=[str(src), *cards, ass_path], write=str(dst))
    return dst
