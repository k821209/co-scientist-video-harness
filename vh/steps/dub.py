"""AI dub into another language (default English) via Kokoro TTS on the render host.

Translation is Claude-in-the-loop (like chapters): Claude translates each montage
segment to the target language, then this module:
  tts_segments()   — per-segment TTS audio + word timestamps (remote Kokoro)
  assemble_dub()   — fit each segment into its time slot (pad / atempo), concat
  caption_words()  — word timestamps offset onto the montage timeline (synced subs)
  mux_audio()      — replace the video's audio with the dub track

Then compose.compose_summary(..., caption_words=caption_words(...)) burns the
translated captions, and mux_audio() swaps in the dub. Prereq: kokoro + soundfile
installed in the render host's VH_RENDER_PYTHON env (pip install kokoro soundfile;
espeak-ng for g2p).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .. import config
from ..probe import run
from .transcribe import Word

DUB_VOICE = os.environ.get("VH_DUB_VOICE", "am_adam")   # Kokoro voice
DUB_LANG = os.environ.get("VH_DUB_LANG", "a")           # 'a' = American English

_KOKORO_WORKER = r'''
import sys, json
import numpy as np, soundfile as sf
from kokoro import KPipeline
inp = json.load(open(sys.argv[1])); outdir = sys.argv[2]; voice = sys.argv[3]; lang = sys.argv[4]
p = KPipeline(lang_code=lang)
meta = []
for seg in inp:
    i, text = seg["i"], seg["text"]
    audios, words, off = [], [], 0.0
    for r in p(text, voice=voice):
        a = r.audio
        a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a, dtype=np.float32)
        for t in (getattr(r, "tokens", None) or []):
            st = getattr(t, "start_ts", None)
            if st is not None and (t.text or "").strip():
                en = getattr(t, "end_ts", None) or (st + 0.25)
                words.append({"text": t.text, "start": off + float(st), "end": off + float(en)})
        audios.append(a); off += len(a) / 24000.0
    audio = np.concatenate(audios) if audios else np.zeros(1, np.float32)
    sf.write(f"{outdir}/seg{i}.wav", audio, 24000)
    meta.append({"i": i, "dur": len(audio) / 24000.0, "words": words})
json.dump(meta, open(f"{outdir}/meta.json", "w"), ensure_ascii=False)
print("TTS_DONE")
'''


def tts_segments(texts: list[str], workdir: str, voice: str | None = None) -> list[dict]:
    """Generate per-segment TTS on the render host. Returns meta with local wav
    paths + word timestamps. Requires a configured render host with kokoro."""
    from .. import remote
    if not remote.enabled():
        raise RuntimeError("dub needs a render host (VH_RENDER_*) with kokoro installed")
    voice = voice or DUB_VOICE
    work = Path(workdir); work.mkdir(parents=True, exist_ok=True)
    rdir = f"{config.RENDER_TMP.rstrip('/')}/vhdub_{os.getpid()}"
    remote.sh(f"mkdir -p {rdir}")
    remote.push_text(json.dumps([{"i": i, "text": t} for i, t in enumerate(texts)],
                                ensure_ascii=False), f"{rdir}/in.json")
    remote.push_text(_KOKORO_WORKER, f"{rdir}/worker.py")
    remote.sh(f"{config.RENDER_PYTHON} {rdir}/worker.py {rdir}/in.json {rdir} "
              f"{voice} {DUB_LANG}")
    remote.pull(f"{rdir}/meta.json", str(work / "meta.json"))
    meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
    for m in meta:
        wp = str(work / f"seg{m['i']}.wav")
        remote.pull(f"{rdir}/seg{m['i']}.wav", wp)
        m["wav"] = wp
    remote.cleanup(rdir)
    return sorted(meta, key=lambda m: m["i"])


def assemble_dub(meta: list[dict], slots: list[tuple], out_wav: str,
                 max_atempo: float = 1.4) -> str:
    """Fit each segment's TTS into its (start,end) slot — pad if short, atempo if
    long — and concat into one track matching the montage timeline."""
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    inputs, filt, labels = [], [], []
    for k, (m, (s, e)) in enumerate(zip(meta, slots)):
        dur = e - s
        inputs += ["-i", m["wav"]]
        if m["dur"] > dur:
            tempo = min(m["dur"] / dur, max_atempo)
            filt.append(f"[{k}]atempo={tempo:.4f},apad=whole_dur={dur:.3f},atrim=0:{dur:.3f}[a{k}]")
        else:
            filt.append(f"[{k}]apad=whole_dur={dur:.3f},atrim=0:{dur:.3f}[a{k}]")
        labels.append(f"[a{k}]")
    filt.append(f"{''.join(labels)}concat=n={len(meta)}:v=0:a=1[out]")
    run([config.FFMPEG, "-y", *inputs, "-filter_complex", ";".join(filt),
         "-map", "[out]", "-ar", "48000", str(out_wav)])
    return out_wav


def caption_words(meta: list[dict], slot_starts: list[float]) -> list[Word]:
    """Word timestamps offset onto the montage timeline (for synced subtitles)."""
    out = []
    for m, start in zip(meta, slot_starts):
        for w in m["words"]:
            txt = w["text"].strip()
            if txt:
                out.append(Word(start + w["start"], start + w["end"], txt))
    return out


def _media_duration(path: str) -> float:
    """Duration (s) from ffprobe format — works for audio-only files too."""
    p = run([config.FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)])
    # run() uses text=True, so stdout is already a str — do NOT .decode() it.
    # Only ValueError is caught: a type mistake must surface, not silently → 0.0
    # (that's what made the audio-longer branch dead, re-truncating video).
    try:
        return float((p.stdout or "").strip() or 0.0)
    except ValueError:
        return 0.0


def mux_audio(video: str, audio: str, dst: str) -> str:
    """Replace a video's audio with `audio`.

    Never silently drops footage: the old `-shortest` cut the video to the audio
    length, so a video tail past the narration (e.g. a silent end card) vanished
    with no warning. Now the output length = max(video, audio):
      - audio shorter → pad it with silence, keep the full video (copy, no
        re-encode) so the end card survives;
      - audio longer  → hold the last video frame so no narration is cut.
    """
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    vdur, adur = _media_duration(video), _media_duration(audio)
    if adur <= vdur + 0.05:
        # video is the longer/equal stream — pad audio to it, keep video as-is.
        run([config.FFMPEG, "-y", "-i", str(video), "-i", str(audio),
             "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
             "-b:a", "192k", "-af", "apad", "-shortest", str(dst)])
    else:
        # audio outlasts video — freeze the last frame so speech isn't clipped.
        pad = adur - vdur
        run([config.FFMPEG, "-y", "-i", str(video), "-i", str(audio),
             "-map", "0:v", "-map", "1:a",
             "-vf", f"tpad=stop_mode=clone:stop_duration={pad:.3f}",
             *config.video_args(),          # configured encoder (NVENC / VH_VENC)
             "-c:a", "aac", "-b:a", "192k", "-shortest", str(dst)])
    return dst
