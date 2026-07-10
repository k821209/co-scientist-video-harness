"""News / text -> vertical short primitives.

Two reusable pieces the screencast pipeline doesn't cover:

- edge_tts_speak(): free Microsoft-Edge neural TTS -> wav. Needed for languages
  Kokoro (vh.steps.dub) can't do — notably KOREAN, plus many others. edge-tts
  does NOT emit word boundaries for Korean, so caption word-timings come from
  transcribe(wav, prompt=script) — the script primes Whisper so it aligns to the
  exact words instead of mis-hearing its own TTS (numbers, homophones).

- ken_burns() / montage(): slow-zoom a still image, and concat several into a
  fast-cut montage (news images shouldn't sit still >~3 s). zoompan needs a
  -frames:v cap or a single looped image multiplies frames into a huge clip.

The full news-short LOOK (image band on top, headline/eyebrow/source at bottom,
per-image "AI 생성 이미지" ribbon vs real-photo credit, word-pop captions) is
assembled by the /news-short skill from these + vh.steps.caption; see the repo.
Prereq: pip install edge-tts. Requires network (edge-tts calls MS's service).
"""
from __future__ import annotations

import asyncio
import difflib
import subprocess

from .. import config
from .transcribe import Word


def align_to_script(words: list, script: str) -> list:
    """Correct caption WORDS to the known script while keeping the transcript's
    timings. Transcribing our own TTS mis-hears numbers/homophones ("정유 4사" ->
    "정유사사"); a long `initial_prompt` fixes those but can truncate long audio.
    This instead transcribes cleanly (complete) then aligns tokens to the script
    (difflib), so every script word is captioned with a sensible time — robust
    across lengths. Use: align_to_script(transcribe(wav, language='ko'), script)."""
    S = script.split()
    wt = [w.text for w in words]
    out: list[Word] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wt, S, autojunk=False).get_opcodes():
        if tag == "equal":
            for wi, sj in zip(range(i1, i2), range(j1, j2)):
                out.append(Word(words[wi].start, words[wi].end, S[sj]))
            continue
        sblk = S[j1:j2]
        if not sblk:
            continue                                       # deletion -> drop
        if i2 > i1:
            t0, t1 = words[i1].start, words[i2 - 1].end
        else:                                              # insertion -> after prev
            t0 = out[-1].end if out else 0.0
            t1 = t0 + 0.4 * len(sblk)
        step = max(0.08, (t1 - t0) / len(sblk))
        for k, s in enumerate(sblk):
            out.append(Word(t0 + k * step, t0 + (k + 1) * step, s))
    return out


async def _stream(text: str, voice: str, out_mp3: str):
    import edge_tts
    c = edge_tts.Communicate(text, voice)
    with open(out_mp3, "wb") as f:
        async for ch in c.stream():
            if ch["type"] == "audio":
                f.write(ch["data"])


def edge_tts_speak(text: str, out_wav: str, voice: str = "ko-KR-SunHiNeural") -> str:
    """Synthesize `text` to `out_wav` (48 kHz) with a free edge-tts neural voice.
    Korean voices: ko-KR-SunHiNeural (F), ko-KR-InJoonNeural (M). Pair with
    transcribe(out_wav, prompt=text) for accurate caption word-timings."""
    mp3 = out_wav.rsplit(".", 1)[0] + ".mp3"
    asyncio.run(_stream(text, voice, mp3))
    subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", "-i", mp3,
                    "-ar", "48000", out_wav], check=True)
    return out_wav


def ken_burns(image: str, dur: float, out_mp4: str, w: int = 1080, h: int = 1056,
              zoom: float = 0.12, drift: bool = False, fps: int = 30) -> str:
    """Slow zoom-in (Ken Burns) on a still image -> `dur`-second clip at w x h."""
    nf = max(1, int(round(dur * fps)))
    rate = round(zoom / nf, 6)
    xy = (f"x='(iw-iw/zoom)*(0.25+0.5*on/{nf})':y='ih/2-(ih/zoom/2)'" if drift
          else "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'")
    vf = (f"scale={2 * w}:{2 * h}:force_original_aspect_ratio=increase,crop={2 * w}:{2 * h},"
          f"zoompan=z='min(zoom+{rate},{1 + zoom:.3f})':d={nf}:{xy}:s={w}x{h}:fps={fps},"
          f"setsar=1,format=yuv420p")
    subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", "-loop", "1", "-i", image,
                    "-vf", vf, "-frames:v", str(nf), *config.encode_args(), "-an", out_mp4],
                   check=True)
    return out_mp4


def montage(items: list[tuple], out_mp4: str, workdir: str, w: int = 1080, h: int = 1056,
            fps: int = 30) -> str:
    """items = [(image, dur), ...]. Ken-Burns each, alternating drift, then concat
    into one video band (no audio). Reuse an image across items for more cuts than
    unique stills — keep repeats >=5 apart."""
    from pathlib import Path
    work = Path(workdir).expanduser(); work.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, (img, dur) in enumerate(items):
        cp = str(work / f"kb{i}.mp4")
        ken_burns(img, dur, cp, w, h, drift=bool(i % 2), fps=fps)
        clips.append(cp)
    lst = str(work / "kb.txt")
    with open(lst, "w") as f:
        for cp in clips:
            f.write(f"file '{Path(cp).name}'\n")
    subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", lst, "-c", "copy", out_mp4], check=True)
    return out_mp4


# ── full news-short assembly (VO → captions → band → 9:16 → mux) ──────────────

def _dur(path: str) -> float:
    r = subprocess.run([config.FFPROBE, "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True)
    return float(r.stdout.strip() or 0.0)


def _ts(t: float) -> str:
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def _overlay_layer(ass: str, *, font: str, accent: str, eyebrow: str, source: str,
                   card: str, card_sub: str | None = None, ribbon: str | None = None,
                   badge: str | None = None, disclosure: str | None = None,
                   credits: list | None = None, starts: list | None = None,
                   off: float, total: float, card_in: float) -> str:
    """Inject the shared news-short overlay (styles + Dialogue events) into a
    boxed ASS — used by BOTH build_short and build_clip_short so they stay
    symmetric. Top-right provenance = per-shot merged `credits` when any are
    given, else the global `ribbon` (a claim like "AI 생성 이미지"); pass neither
    to assert nothing. `disclosure` = bottom conflict-of-interest footnote;
    `badge` = an info chip shown during the 2nd shot; `accent` = ASS BGR colour."""
    F = font
    src_mv = 92 if disclosure else 56          # lift the source line above a disclosure
    styles = [
        f"Style: Eyebrow,{F},34,{accent},&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,4,0,1,3,0,8,60,60,96,1",
        f"Style: Bar,{F},34,{accent},&H00FFFFFF,{accent},&H00000000,0,0,0,0,100,100,0,0,1,0,0,8,0,0,0,1",
        f"Style: Src,{F},28,&H00C8C8C8,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,0,2,50,50,{src_mv},1",
        f"Style: Disc,{F},25,&H009090A8,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,0,2,50,50,52,1",
        f"Style: Cred,{F},28,&H00FFFFFF,&H00FFFFFF,&H00000000,&H78000000,-1,0,0,0,100,100,0,0,3,8,0,9,24,24,452,1",
        f"Style: Badge,{F},40,{accent},&H00FFFFFF,&H00202020,&H64000000,-1,0,0,0,100,100,2,0,1,4,3,5,0,0,0,1",
        f"Style: Card,{F},104,{accent},&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,12,0,1,0,0,5,0,0,0,1",
        f"Style: CardSub,{F},36,&H00C8C8C8,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,6,0,1,0,0,5,0,0,0,1",
    ]
    ass = ass.replace("\n\n[Events]", "\n" + "\n".join(styles) + "\n\n[Events]")
    ev = [
        f"Dialogue: 0,{_ts(0.0)},{_ts(off)},Bar,,0,0,0,,{{\\an7\\pos(480,150)\\p1}}m 0 0 l 120 0 l 120 5 l 0 5{{\\p0}}",
        f"Dialogue: 0,{_ts(0.0)},{_ts(off)},Eyebrow,,0,0,0,,{eyebrow}",
        f"Dialogue: 0,{_ts(0.0)},{_ts(total)},Src,,0,0,0,,{source}",
    ]
    # provenance top-right: per-shot credits (merge adjacent equal), else ribbon
    if credits and any(credits) and starts:
        i = 0
        while i < len(credits):
            j = i
            while j + 1 < len(credits) and credits[j + 1] == credits[i]:
                j += 1
            if credits[i]:
                seg_start = 0.0 if i == 0 else starts[i]
                seg_end = starts[j + 1] if j + 1 < len(starts) else total
                ev.append(f"Dialogue: 0,{_ts(seg_start)},{_ts(min(seg_end, off))},Cred,,0,0,0,,{credits[i]}")
            i = j + 1
    elif ribbon:
        ev.append(f"Dialogue: 0,{_ts(0.0)},{_ts(off)},Cred,,0,0,0,,{ribbon}")
    if disclosure:
        ev.append(f"Dialogue: 0,{_ts(0.0)},{_ts(total)},Disc,,0,0,0,,{disclosure}")
    if badge and starts and len(starts) > 2:
        bs = starts[1]
        be = min(starts[2], bs + 3.2)
        ev.append(f"Dialogue: 0,{_ts(bs)},{_ts(be)},Badge,,0,0,0,,{{\\an5\\pos(540,1360)\\fad(250,250)}}{badge}")
    ev.append(f"Dialogue: 0,{_ts(card_in)},{_ts(total)},Card,,0,0,0,,{{\\an5\\pos(540,930)\\fad(350,0)}}{card}")
    if card_sub:
        ev.append(f"Dialogue: 0,{_ts(card_in + 0.15)},{_ts(total)},CardSub,,0,0,0,,{{\\an5\\pos(540,1020)\\fad(400,0)}}{card_sub}")
    return ass.rstrip("\n") + "\n" + "\n".join(ev) + "\n"


def _clip_pad(span: float) -> float:
    """Last-frame hold so a clip fills its WHOLE sentence span — no fixed cap: a
    short clip under a long sentence must still cover it (a 4 s cap silently left
    the band short → dropped tail)."""
    return max(0.0, span)


def _check_output_duration(got: float, total: float, tol: float = 0.25) -> None:
    """Fail loud when the render isn't VO+tail — i.e. the end card / tail was
    silently dropped. The tolerance is deliberately tight; don't loosen it."""
    if abs(got - total) > tol:
        raise RuntimeError(f"output {got:.2f}s != expected {total:.2f}s — the end "
                           f"card / tail was dropped (a clip couldn't fill its span?)")


def _workdir(workdir: str | None, prefix: str):
    """Resolve a caller's workdir to an ABSOLUTE, ~-expanded path (or a fresh
    tempdir). expanduser() so "~/wd" goes to $HOME (not a literal '~' dir under
    cwd); resolve() so ffmpeg concat lists / -i paths don't depend on cwd."""
    import pathlib
    import tempfile
    p = pathlib.Path(workdir) if workdir else pathlib.Path(tempfile.mkdtemp(prefix=prefix))
    return p.expanduser().resolve()


def _norm(s: str) -> str:
    import re
    return re.sub(r"\s+", "", s)


def _find_anchor(words: list, anchor: str, pos: int) -> int:
    """First word index ≥ pos whose text (possibly spanning consecutive tokens)
    starts with `anchor`, whitespace-insensitive. Handles Korean multi-word
    anchors that align/edge-tts split across tokens (e.g. "이 속도를" →
    ["이","속도를,"]). Raises with the nearby tokens on failure."""
    target = _norm(anchor)
    if not target:
        raise ValueError("empty anchor")
    for k in range(pos, len(words)):
        acc = ""
        for m in range(k, min(k + 8, len(words))):
            acc += _norm(words[m].text)
            if acc.startswith(target):
                return k
            if len(acc) >= len(target):      # accumulated past target without a prefix match
                break
    nearby = " ".join(w.text for w in words[pos:pos + 10])
    raise ValueError(
        f"shot anchor {anchor!r} not found after word {pos}. "
        f"Next tokens were: {nearby!r} — pick an anchor that matches the "
        f"aligned tokens (whitespace is ignored; a multi-token span is fine)."
    )


def build_short(
    script: str,
    shots: list,           # [(anchor, image[, credit]), ...] — one per sentence/clause
    out: str,
    *,
    headline: str,
    eyebrow: str,
    source: str,
    ribbon: str | None = None,         # global provenance ribbon, e.g. "AI 생성 이미지".
                                       # DEFAULT None — never asserts; the /news-short
                                       # guardrail requires setting it for AI images.
    card: str = "AIVO",
    card_sub: str | None = None,
    disclosure: str | None = None,     # conflict-of-interest footnote (optional)
    badge: str | None = None,          # info chip shown during the 2nd shot
    accent: str = "&H0000E5FF",        # eyebrow/bar/card colour (ASS BGR)
    voice: str = "ko-KR-SunHiNeural",
    workdir: str | None = None,
    tail: float = 2.8,                 # silent end-card seconds after the VO
    max_cut: float = 3.6,              # long sentence → split into ≤ this many s
    canvas_w: int = 1080,
    canvas_h: int = 1920,
    band_h: int = 1056,
    font: str = "Noto Sans CJK KR",
    max_repeat: int = 2,
) -> dict:
    """Assemble a full vertical news Short from a script + per-sentence image map.

    The pipeline the three reference build scripts shared verbatim: edge-tts VO →
    align_to_script captions → map each shot's anchor to its sentence start →
    Ken-Burns montage band → boxed ASS (headline + captions) + overlays (eyebrow,
    accent bar, source·date, AI ribbon, optional disclosure, end card) → compose
    the 9:16 frame (pad + band fade + burned subtitles + final fade) → mux the VO
    (video length preserved so the silent end card survives).

    shots: [(anchor, image_path)] — anchor is a phrase near the sentence start;
    matching is whitespace-insensitive and spans tokens. Style margins are tuned
    for the default 1080×1920 canvas. Returns {final, duration, vo, words}.
    """
    import json
    import pathlib
    import shutil
    from .caption import build_boxed_ass, write_ass, _fg_escape
    from .transcribe import transcribe
    from . import dub

    top = (canvas_h - band_h) // 2
    bottom = top + band_h
    wd = _workdir(workdir, "vh_short_")
    wd.mkdir(parents=True, exist_ok=True)

    # 1. voiceover
    vo = str(wd / "vo.wav")
    edge_tts_speak(script, vo, voice=voice)
    d_vo = _dur(vo)
    total = d_vo + tail

    # 2. captions (timing from Whisper, words from the script)
    words = align_to_script(transcribe(vo, language="ko"), script)
    if not words:
        raise ValueError("no aligned words — VO transcription failed")

    # 3. image band — one image per sentence, cut on the sentence boundary.
    # shots may be (anchor, image) or (anchor, image, credit) for per-photo source.
    norm = [(s[0], s[1], s[2] if len(s) > 2 else None) for s in shots]
    anchored, pos = [], 0
    for anchor, img, _cred in norm:
        j = _find_anchor(words, anchor, pos)
        anchored.append((words[j].start, img))
        pos = j + 1
    anchored[0] = (0.0, anchored[0][1])         # first shot always starts at 0
    starts = [t for t, _ in anchored] + [total]
    imgs = [img for _, img in anchored]
    credits = [c for _, _, c in norm]
    items = []
    for i, img in enumerate(imgs):
        span = starts[i + 1] - starts[i]
        n = max(1, round(span / max_cut))       # long sentence → split, same image
        for _ in range(n):
            items.append((img, round(span / n, 3), i))
    bad = [i for i in range(len(items) - 1)
           if items[i][0] == items[i + 1][0] and items[i][2] != items[i + 1][2]]
    if bad:
        raise ValueError(f"same image on adjacent sentences at cut(s) {bad}")
    from collections import Counter
    worst = Counter(imgs).most_common(1)[0]
    if worst[1] > max_repeat:
        raise ValueError(f"image {worst[0]!r} reused across {worst[1]} sentences "
                         f"(> max_repeat={max_repeat}) — add more stills")
    band = str(wd / "band.mp4")
    if (wd / "kb").exists():
        shutil.rmtree(wd / "kb")
    montage([(p, d) for p, d, _ in items], band, str(wd / "kb"), w=canvas_w, h=band_h)

    # 4. ASS: boxed headline + captions + shared overlay layer
    ass = build_boxed_ass(words, canvas_w, canvas_h, top, bottom, video_title=headline,
                          style="word", max_words=4, title_end=d_vo + 0.30,
                          hold_through_pauses=True)
    ass = _overlay_layer(ass, font=font, accent=accent, eyebrow=eyebrow, source=source,
                         card=card, card_sub=card_sub, ribbon=ribbon, badge=badge,
                         disclosure=disclosure, credits=credits, starts=starts,
                         off=d_vo + 0.30, total=total, card_in=d_vo + 0.55)
    ass_path = write_ass(ass, str(wd / "short.ass"))

    # 5. compose 9:16 (pad + band fade-out at VO end + burned subs + final fade)
    composed = str(wd / "composed.mp4")
    vf = (f"pad={canvas_w}:{canvas_h}:0:{top}:color=0x0B0B14,"
          f"fade=t=out:st={d_vo + 0.10:.2f}:d=1.10:color=0x0B0B14,"
          f"subtitles='{_fg_escape(ass_path)}':fontsdir='{_fg_escape(config.CAPTION_FONTSDIR)}',"
          f"fade=t=out:st={total - 0.45:.2f}:d=0.45")
    subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", "-i", band,
                    "-vf", vf, *config.video_args(), "-an", composed], check=True)

    # 6. mux VO (mux_audio preserves the full video length → end card survives)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    dub.mux_audio(composed, vo, out)
    got = _dur(out)
    _check_output_duration(got, total)
    return {"final": out, "duration": got, "vo": d_vo, "words": len(words),
            "sentences": len(shots), "cuts": len(items)}


# ── clip-quotation shorts (short YouTube clips + our VO) ──────────────────────

def fetch_clip(url: str, start, dur: float, dst: str, *,
               keep_audio: bool = False, max_height: int = 1080) -> str:
    """Partial-download a few seconds of a YouTube video as a VIDEO-ONLY clip
    (no original audio, so a news/critique quotation doesn't reuse the copyright
    soundtrack — our VO is added later). Wraps yt-dlp with the right flags:
    --download-sections, --force-keyframes-at-cuts, -f bv, and --ffmpeg-location
    (ffmpeg is often not on PATH). `start` = seconds or "MM:SS"; `dur` seconds.

    Source responsibly: safe-tier channels (official label / member official
    channels), on-screen credit, news/critique purpose (Content-ID risk remains).
    """
    import os
    import pathlib

    def _sec(t) -> float:
        if isinstance(t, str) and ":" in t:
            parts = [float(p) for p in t.split(":")]
            return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
        return float(t)

    s = _sec(start)
    e = s + float(dur)
    fmt = (f"b[height<=?{max_height}]" if keep_audio
           else f"bv[height<=?{max_height}]")   # bv = video only → no soundtrack
    ytdlp = os.environ.get("VH_YTDLP", "yt-dlp")
    cmd = [ytdlp, "--download-sections", f"*{s:.3f}-{e:.3f}",
           "--force-keyframes-at-cuts", "-f", fmt, "-o", str(dst), url]
    ffdir = os.path.dirname(config.FFMPEG)
    if ffdir:                                    # help yt-dlp find ffmpeg
        cmd[1:1] = ["--ffmpeg-location", ffdir]
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    return str(dst)


def build_clip_short(
    script: str,
    shots: list,           # [(anchor, clip_path, is_vlog, credit), ...]
    out: str,
    *,
    headline: str,
    eyebrow: str,
    source: str,
    badge: str | None = None,          # brief person/info chip (shown on 2nd shot)
    disclosure: str | None = None,     # conflict-of-interest footnote (optional)
    accent: str = "&H00F5A0FF",        # eyebrow/bar/card colour (ASS BGR)
    card: str = "AIVO",
    card_sub: str | None = None,
    voice: str = "ko-KR-SunHiNeural",
    workdir: str | None = None,
    tail: float = 2.8,
    vlog_crop: float = 0.82,           # keep top 82% of vlog clips (drop burned-in subs)
    canvas_w: int = 1080,
    canvas_h: int = 1920,
    band_h: int = 1056,
    font: str = "Noto Sans CJK KR",
) -> dict:
    """Assemble a clip-quotation Short: several short source clips (already
    downloaded, e.g. via fetch_clip) reframed to the 9:16 band and trimmed to
    each sentence's span, over our VO, with a per-clip source credit on screen.

    shots: [(anchor, clip_path, is_vlog, credit)] — one per sentence. `is_vlog`
    True crops the bottom `1-vlog_crop` (burned-in captions) before reframing.
    `credit` is the on-screen source (adjacent equal credits are merged).
    Style margins tuned for 1080×1920. Returns {final, duration, vo, words}.
    """
    import json
    import pathlib
    from .caption import build_boxed_ass, write_ass, _fg_escape
    from .transcribe import transcribe
    from . import dub

    top = (canvas_h - band_h) // 2
    bottom = top + band_h
    # absolute + ~-expanded: ffmpeg resolves each concat `file` entry relative to
    # the concat file's dir, so a relative workdir would double-prefix.
    wd = _workdir(workdir, "vh_clip_")
    (wd / "bandclips").mkdir(parents=True, exist_ok=True)

    # 1. VO + 2. captions
    vo = str(wd / "vo.wav")
    edge_tts_speak(script, vo, voice=voice)
    d_vo = _dur(vo)
    total = d_vo + tail
    words = align_to_script(transcribe(vo, language="ko"), script)
    if not words:
        raise ValueError("no aligned words — VO transcription failed")

    # 3. sentence spans via robust anchors
    anchored, pos = [], 0
    for anchor, clip, _v, _c in shots:
        j = _find_anchor(words, anchor, pos)
        anchored.append((words[j].start, clip))
        pos = j + 1
    anchored[0] = (0.0, anchored[0][1])
    starts = [t for t, _ in anchored] + [total]
    spans = [round(starts[i + 1] - starts[i], 3) for i in range(len(shots))]

    # 4. reframe each clip to the band, trimmed to its span, then concat
    va = config.video_args()
    lines = []
    for i, (anchor, clip, is_vlog, cred) in enumerate(shots):
        dst = str(wd / "bandclips" / f"{i:02d}.mp4")
        pre = f"crop=iw:ih*{vlog_crop}:0:0," if is_vlog else ""
        # hold the last frame for the WHOLE span (a short clip must still fill a
        # long sentence) — a fixed cap silently left the band short → lost tail.
        vf = (f"{pre}scale={canvas_w}:{band_h}:force_original_aspect_ratio=increase,"
              f"crop={canvas_w}:{band_h},fps=30,"
              f"tpad=stop_mode=clone:stop_duration={_clip_pad(spans[i]):.3f},setsar=1,format=yuv420p")
        subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", "-i", str(clip),
                        "-vf", vf, "-t", f"{spans[i]:.3f}", *va, "-an", dst], check=True)
        lines.append(f"file '{dst}'")
    concat_list = str(wd / "concat.txt")
    pathlib.Path(concat_list).write_text("\n".join(lines) + "\n")
    band = str(wd / "band.mp4")
    subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", concat_list, "-c", "copy", band], check=True)

    # 5. ASS: headline + captions + shared overlay layer (per-clip credits)
    ass = build_boxed_ass(words, canvas_w, canvas_h, top, bottom, video_title=headline,
                          style="word", max_words=4, title_end=d_vo + 0.30,
                          hold_through_pauses=True)
    ass = _overlay_layer(ass, font=font, accent=accent, eyebrow=eyebrow, source=source,
                         card=card, card_sub=card_sub, badge=badge, disclosure=disclosure,
                         credits=[s[3] for s in shots], starts=starts,
                         off=d_vo + 0.30, total=total, card_in=d_vo + 0.55)
    ass_path = write_ass(ass, str(wd / "short.ass"))

    # 6. compose 9:16 + 7. mux
    composed = str(wd / "composed.mp4")
    vf = (f"pad={canvas_w}:{canvas_h}:0:{top}:color=0x0B0B14,"
          f"fade=t=out:st={d_vo + 0.10:.2f}:d=1.10:color=0x0B0B14,"
          f"subtitles='{_fg_escape(ass_path)}':fontsdir='{_fg_escape(config.CAPTION_FONTSDIR)}',"
          f"fade=t=out:st={total - 0.45:.2f}:d=0.45")
    subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", "-i", band,
                    "-vf", vf, *va, "-an", composed], check=True)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    dub.mux_audio(composed, vo, out)
    got = _dur(out)
    _check_output_duration(got, total)
    return {"final": out, "duration": got, "vo": d_vo, "words": len(words),
            "clips": len(shots)}
