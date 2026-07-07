"""Boxed vertical composition.

Turns a landscape (16:9) source into a vertical (9:16) frame WITHOUT blurred
bars as the main event: the video sits in a centered band, the freed top space
becomes a header zone (section title / video title) and the bottom space a
caption zone. One ffmpeg pass — background + video overlay + burned ASS — so
only a single NVENC encode.

Returns (dst, video_top, video_bottom) so the caller can size the ASS bands.
"""
from __future__ import annotations

from pathlib import Path

from .. import config
from ..probe import run, probe
from .caption import _fg_escape
from .transcribe import Word


def video_band(src: str, canvas_w: int, canvas_h: int) -> tuple[int, int, int]:
    """Full-width video band centered on the canvas -> (band_h, top, bottom)."""
    info = probe(src)
    band_h = round(canvas_w * info.height / info.width)
    band_h -= band_h % 2
    top = max(0, (canvas_h - band_h) // 2)
    top -= top % 2
    return band_h, top, top + band_h


def compose_boxed(
    src: str,
    dst: str,
    canvas_w: int,
    canvas_h: int,
    ass_path: str,
    top: int,
    bg: str = "solid",
    bg_color: str = "0x0B0B14",
    fontsdir: str | None = None,
    zoom: float = 1.0,
    focus_x: float = 0.5,
    focus_y: float = 0.5,
    pan_x: tuple | None = None,
    duration: float | None = None,
) -> str:
    """`zoom` > 1 punches into the source (crop `1/zoom` of it, centered on
    focus_x/focus_y in 0..1) so screencast text reads larger in the video band.
    `pan_x=(f0,f1)` with `duration` slides the horizontal focus from f0 to f1 —
    holding f0 for the first third, panning across the middle third, holding f1 —
    to follow the on-screen pointer (e.g. left page early, right page late)."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    ass = _fg_escape(ass_path)
    fdir = fontsdir or config.CAPTION_FONTSDIR
    sub = f"subtitles='{ass}'" + (f":fontsdir='{_fg_escape(fdir)}'" if fdir else "")
    z = ""
    if zoom and zoom > 1.001:
        yoff = f"(ih-ih/{zoom:.4f})*{focus_y:.3f}"
        if pan_x is not None and duration:
            f0, f1 = pan_x
            ts, te = 0.33 * duration, 0.66 * duration
            xoff = (f"(iw-iw/{zoom:.4f})*({f0:.3f}+({f1:.3f}-{f0:.3f})"
                    f"*clip((t-{ts:.2f})/{max(te - ts, 0.1):.2f}\\,0\\,1))")
        else:
            xoff = f"(iw-iw/{zoom:.4f})*{focus_x:.3f}"
        z = f"crop=iw/{zoom:.4f}:ih/{zoom:.4f}:{xoff}:{yoff},"

    if bg == "blur":
        vf = (
            f"[0:v]{z}scale={canvas_w}:-2[v];"
            f"[0:v]scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,"
            f"crop={canvas_w}:{canvas_h},boxblur=40:2,eq=brightness=-0.30[bg];"
            f"[bg][v]overlay=0:{top}[base];"
            f"[base]{sub}[out]"
        )
    else:  # solid color bands via pad
        vf = (
            f"[0:v]{z}scale={canvas_w}:-2,"
            f"pad={canvas_w}:{canvas_h}:0:{top}:color={bg_color}[base];"
            f"[base]{sub}[out]"
        )

    from .. import remote
    remote.ffmpeg_run([
        config.FFMPEG, "-y", "-i", str(src),
        "-filter_complex", vf, "-map", "[out]", "-map", "0:a?",
        *config.encode_args(),
        str(dst),
    ], reads=[str(src), str(ass_path)], write=str(dst))
    return dst


def compose_summary(
    src: str,
    dst: str,
    segments: list,
    words: list,
    header: str,
    canvas_w: int = 1080,
    canvas_h: int = 1920,
    style: str = "word",
    max_words: int = 4,
    bg_color: str = "0x0B0B14",
    workdir: str | None = None,
    caption_words: list | None = None,
) -> tuple:
    """Summary Short: cut several highlight windows from `src`, concat them, then
    box into a vertical frame (header top, video band, caption bottom). Each
    window is snapped to speech pauses (no abrupt cut); captions are re-timed to
    the concatenated timeline. One ffmpeg pass, remote-offloaded.

    `caption_words` (already in concatenated-montage time) overrides the captions
    — used to burn a translated track (e.g. English) instead of the source
    transcript."""
    from .. import remote
    from .caption import build_boxed_ass, write_ass

    info = probe(src)
    SW, SH, fps = info.width, info.height, info.fps or 30.0

    # snap each window to SENTENCE boundaries so a clip never starts or ends
    # mid-thought. A sentence break = a word ending in .?!… OR a >=0.45s pause;
    # start snaps to a sentence start, end to a sentence end. Falls back to the
    # nearest word boundary if no sentence break is near.
    sent_starts, sent_ends = [], []
    for i, w in enumerate(words):
        gap = (words[i + 1].start - w.end) if i + 1 < len(words) else 99.0
        if w.text.strip().endswith((".", "?", "!", "…")) or gap >= 0.45:
            sent_ends.append(w.end)
            if i + 1 < len(words):
                sent_starts.append(words[i + 1].start)
    if words:
        sent_starts.append(words[0].start)
    all_starts = [w.start for w in words]
    all_ends = [w.end for w in words]

    def _snap(cands, t, tol, fallback):
        if cands:
            c = min(cands, key=lambda x: abs(x - t))
            if abs(c - t) <= tol:
                return c
        c = min(fallback, key=lambda x: abs(x - t))
        return c if abs(c - t) <= 1.5 else t

    segs = []
    for s, e in segments:
        ss = _snap(sent_starts, float(s), 5.0, all_starts)
        ee = _snap(sent_ends, float(e), 5.0, all_ends)
        if ee - ss > 0.5:
            segs.append((ss, ee))
    # sort by source time: keeps concat order == single-decode order (avoids the
    # filtergraph buffering that otherwise appends stray footage) and reads as a
    # natural chronological summary.
    segs.sort(key=lambda z: z[0])
    K = len(segs)

    band_h = round(canvas_w * SH / SW)
    band_h -= band_h % 2
    top = (canvas_h - band_h) // 2
    top -= top % 2
    bottom = top + band_h

    # captions: either a supplied translated track (already montage-time) or the
    # source transcript re-timed onto the concatenated timeline.
    if caption_words is not None:
        clipwords = list(caption_words)
    else:
        clipwords, offset = [], 0.0
        for ss, ee in segs:
            for w in words:
                if ss <= w.start < ee:
                    clipwords.append(Word(offset + (max(ss, w.start) - ss),
                                          offset + (min(ee, w.end) - ss), w.text))
            offset += (ee - ss)

    work = Path(workdir or (Path(dst).parent / "_sum"))
    work.mkdir(parents=True, exist_ok=True)
    ass_path = str(work / "summary.ass")
    write_ass(build_boxed_ass(clipwords, canvas_w, canvas_h, top, bottom,
                              video_title=header, style=style, max_words=max_words),
              ass_path)

    vsplit = "".join(f"[vin{k}]" for k in range(K))
    asplit = "".join(f"[ain{k}]" for k in range(K))
    parts = [f"[0:v]split={K}{vsplit}", f"[0:a]asplit={K}{asplit}"]
    for k, (ss, ee) in enumerate(segs):
        parts.append(f"[vin{k}]trim=start={ss:.3f}:end={ee:.3f},setpts=PTS-STARTPTS,"
                     f"fps={fps},scale={SW}:{SH},setsar=1,format=yuv420p[v{k}]")
        parts.append(f"[ain{k}]atrim=start={ss:.3f}:end={ee:.3f},asetpts=PTS-STARTPTS,"
                     f"aformat=sample_rates=48000:channel_layouts=stereo[a{k}]")
    order = "".join(f"[v{k}][a{k}]" for k in range(K))
    parts.append(f"{order}concat=n={K}:v=1:a=1[cv][ca]")
    parts.append(f"[cv]scale={canvas_w}:-2,pad={canvas_w}:{canvas_h}:0:{top}:color={bg_color}[base]")
    fdir = config.CAPTION_FONTSDIR
    parts.append(f"[base]subtitles='{_fg_escape(ass_path)}'"
                 + (f":fontsdir='{_fg_escape(fdir)}'" if fdir else "") + "[outv]")
    filt = ";".join(parts)

    remote.ffmpeg_run([
        config.FFMPEG, "-y", "-i", str(src),
        "-filter_complex", filt, "-map", "[outv]", "-map", "[ca]",
        *config.encode_args(), str(dst),
    ], reads=[str(src), ass_path], write=str(dst))
    return dst, segs
