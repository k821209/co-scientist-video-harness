"""Transcription -> word-level segments + SRT.

Two backends (config.TRANSCRIBE_BACKEND):
  "gpu" -> transformers Whisper on the GB10 GPU, shelled out to the deepspeed
           env (CTranslate2 has no CUDA aarch64 wheel). ~2.4x realtime.
  "cpu" -> in-process faster-whisper int8. Portable, ~0.3x realtime here.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

from .. import config


@dataclass
class Word:
    start: float
    end: float
    text: str


_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        _MODEL = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE,
            cpu_threads=config.WHISPER_CPU_THREADS,
        )
    return _MODEL


def _extract_wav(src: str, wav: str) -> str:
    """16 kHz mono wav for ASR."""
    from ..probe import run
    run([config.FFMPEG, "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(wav)])
    return wav


def _transcribe_gpu(src: str, language: str | None) -> list[Word]:
    worker = str(Path(__file__).with_name("gpu_asr.py"))
    with tempfile.TemporaryDirectory() as td:
        wav = _extract_wav(src, f"{td}/audio.wav")
        out_json = f"{td}/words.json"
        proc = subprocess.run(
            [config.GPU_PYTHON, worker, wav, out_json,
             language or "auto", config.WHISPER_MODEL],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if proc.returncode != 0 or not Path(out_json).exists():
            tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
            raise RuntimeError(f"GPU ASR failed ({proc.returncode}):\n{tail}")
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    return [Word(**d) for d in data]


def _transcribe_cpu(src: str, language: str | None) -> list[Word]:
    segments, _info = _model().transcribe(
        str(src), language=language, word_timestamps=True, vad_filter=True,
    )
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            txt = w.word.strip()
            if txt:
                words.append(Word(start=w.start, end=w.end, text=txt))
    return words


# Self-contained faster-whisper worker shipped to the render host (needs only
# faster-whisper installed there; CUDA if available, else CPU int8).
_REMOTE_ASR_WORKER = r'''
import sys, json
wav, out, lang, model = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
lang = None if lang in ("", "auto", "none") else lang
from faster_whisper import WhisperModel
try:
    m = WhisperModel(model, device="cuda", compute_type="float16")
except Exception:
    m = WhisperModel(model, device="cpu", compute_type="int8")
segs, info = m.transcribe(wav, language=lang, word_timestamps=True, vad_filter=True)
words = [{"start": float(w.start), "end": float(w.end), "text": w.word.strip()}
         for s in segs for w in (s.words or []) if w.word.strip()]
json.dump(words, open(out, "w"), ensure_ascii=False)
'''


def _transcribe_remote(src: str, language: str | None) -> list[Word]:
    import os
    from .. import remote
    tag = "vh_" + "".join(c for c in Path(src).stem if c.isalnum())[:24] + f"_{os.getpid()}"
    rtmp = config.RENDER_TMP.rstrip("/")
    rwav, rjson, rwork = f"{rtmp}/{tag}.wav", f"{rtmp}/{tag}.json", f"{rtmp}/{tag}_asr.py"
    with tempfile.TemporaryDirectory() as td:
        wav = _extract_wav(src, f"{td}/audio.wav")           # local ffmpeg (light)
        remote.push(wav, rwav)
        remote.push_text(_REMOTE_ASR_WORKER, rwork)
        remote.sh(f"{config.RENDER_PYTHON} {rwork} {rwav} {rjson} "
                  f"{language or 'auto'} {config.WHISPER_MODEL}")
        local_json = f"{td}/words.json"
        remote.pull(rjson, local_json)
        data = json.loads(Path(local_json).read_text(encoding="utf-8"))
    remote.cleanup(rwav, rjson, rwork)
    return [Word(**d) for d in data]


def transcribe(src: str, language: str | None = None) -> list[Word]:
    """Return a flat list of timed words. Backend per config.TRANSCRIBE_BACKEND
    ('auto' -> remote if a render host is configured, else local GPU/CPU)."""
    from .. import remote
    backend = config.TRANSCRIBE_BACKEND
    if backend == "auto":
        backend = "remote" if remote.enabled() else "gpu"

    if backend == "remote":
        try:
            return _transcribe_remote(src, language)
        except Exception as e:
            print(f"[transcribe] remote host failed ({e}); falling back to local")
            backend = "gpu"
    if backend == "gpu":
        try:
            return _transcribe_gpu(src, language)
        except Exception as e:
            print(f"[transcribe] GPU backend failed ({e}); falling back to CPU")
    return _transcribe_cpu(src, language)


def _ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(words: list[Word], path: str, max_words: int = 8) -> str:
    """Group words into readable SRT cues (soft subtitles for YouTube)."""
    lines, idx, i = [], 1, 0
    while i < len(words):
        chunk = words[i:i + max_words]
        start, end = chunk[0].start, chunk[-1].end
        text = " ".join(w.text for w in chunk)
        lines.append(f"{idx}\n{_ts(start)} --> {_ts(end)}\n{text}\n")
        idx += 1
        i += max_words
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path


def dump_words(words: list[Word], path: str) -> str:
    Path(path).write_text(
        json.dumps([asdict(w) for w in words], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
