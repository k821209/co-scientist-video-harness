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
    work = Path(workdir); work.mkdir(parents=True, exist_ok=True)
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
    shots: list,           # [(anchor_text, image_path), ...] — one per sentence/clause
    out: str,
    *,
    headline: str,
    eyebrow: str,
    source: str,
    ribbon: str = "AI 생성 이미지",
    card: str = "AIVO",
    card_sub: str | None = None,
    disclosure: str | None = None,     # conflict-of-interest footnote (optional)
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
    wd = pathlib.Path(workdir) if workdir else pathlib.Path(__import__("tempfile").mkdtemp(prefix="vh_short_"))
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

    # 3. image band — one image per sentence, cut on the sentence boundary
    anchored, pos = [], 0
    for anchor, img in shots:
        j = _find_anchor(words, anchor, pos)
        anchored.append((words[j].start, img))
        pos = j + 1
    anchored[0] = (0.0, anchored[0][1])         # first shot always starts at 0
    starts = [t for t, _ in anchored] + [total]
    imgs = [img for _, img in anchored]
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

    # 4. ASS: boxed headline + captions, then overlay styles/events
    ass = build_boxed_ass(words, canvas_w, canvas_h, top, bottom, video_title=headline,
                          style="word", max_words=4, title_end=d_vo + 0.30,
                          hold_through_pauses=True)
    F = font
    styles = [
        f"Style: Eyebrow,{F},34,&H0000E5FF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,4,0,1,3,0,8,60,60,96,1",
        f"Style: Bar,{F},34,&H0000E5FF,&H00FFFFFF,&H0000E5FF,&H00000000,0,0,0,0,100,100,0,0,1,0,0,8,0,0,0,1",
        f"Style: Src,{F},29,&H00C8C8C8,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,0,2,50,50,92,1",
        f"Style: Disc,{F},25,&H009090A8,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,0,2,50,50,52,1",
        f"Style: Ribbon,{F},28,&H00FFFFFF,&H00FFFFFF,&H00000000,&H78000000,-1,0,0,0,100,100,0,0,3,8,0,9,24,24,452,1",
        f"Style: Card,{F},104,&H0000E5FF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,12,0,1,0,0,5,0,0,0,1",
        f"Style: CardSub,{F},34,&H00C8C8C8,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,6,0,1,0,0,5,0,0,0,1",
    ]
    ass = ass.replace("\n\n[Events]", "\n" + "\n".join(styles) + "\n\n[Events]")

    off = d_vo + 0.30
    card_in = d_vo + 0.55
    events = [
        # accent bar: libass anchors a \p1 drawing's bbox by \an7 top-left, so
        # draw from (0,0) with non-negative coords (NOT \an5 — that mis-places drawings).
        f"Dialogue: 0,{_ts(0.0)},{_ts(off)},Bar,,0,0,0,,{{\\an7\\pos(480,150)\\p1}}m 0 0 l 120 0 l 120 5 l 0 5{{\\p0}}",
        f"Dialogue: 0,{_ts(0.0)},{_ts(off)},Eyebrow,,0,0,0,,{eyebrow}",
        f"Dialogue: 0,{_ts(0.0)},{_ts(off)},Ribbon,,0,0,0,,{ribbon}",
        f"Dialogue: 0,{_ts(0.0)},{_ts(total)},Src,,0,0,0,,{source}",
    ]
    if disclosure:
        events.append(f"Dialogue: 0,{_ts(0.0)},{_ts(total)},Disc,,0,0,0,,{disclosure}")
    events.append(f"Dialogue: 0,{_ts(card_in)},{_ts(total)},Card,,0,0,0,,{{\\an5\\pos(540,930)\\fad(350,0)}}{card}")
    if card_sub:
        events.append(f"Dialogue: 0,{_ts(card_in + 0.15)},{_ts(total)},CardSub,,0,0,0,,{{\\an5\\pos(540,1020)\\fad(400,0)}}{card_sub}")
    ass_path = write_ass(ass.rstrip("\n") + "\n" + "\n".join(events) + "\n", str(wd / "short.ass"))

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
    return {"final": out, "duration": got, "vo": d_vo, "words": len(words),
            "sentences": len(shots), "cuts": len(items)}
