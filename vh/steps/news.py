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
import subprocess

from .. import config


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
