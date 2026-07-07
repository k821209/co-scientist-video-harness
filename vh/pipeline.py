"""Orchestrator: raw recording -> captioned, framed, publish-ready mp4.

Stage order (timing-preserving so word timestamps stay valid):
  clean (silence trim) -> reframe (aspect) -> transcribe -> caption burn
Each stage writes an intermediate into <workdir> so runs are resumable/inspectable.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import get_preset
from .probe import probe
from .steps import clean as _clean
from .steps import reframe as _reframe
from .steps import transcribe as _tx
from .steps import caption as _cap


@dataclass
class Result:
    final: str
    srt: str
    ass: str
    words_json: str
    n_words: int
    duration_out: float


def run_pipeline(
    src: str,
    preset_name: str,
    outdir: str,
    language: str | None = None,
    keep_intermediates: bool = True,
) -> Result:
    preset = get_preset(preset_name)
    src = str(Path(src).expanduser().resolve())
    stem = Path(src).stem
    work = Path(outdir) / stem
    work.mkdir(parents=True, exist_ok=True)

    info = probe(src)
    print(f"[probe] {info.width}x{info.height} {info.duration:.1f}s "
          f"audio={'yes' if info.has_audio else 'NO'} fps={info.fps:.1f}")
    if not info.has_audio:
        print("[warn] no audio track — skipping silence-trim & captions")

    # 1) clean
    cur = src
    if preset.clean and info.has_audio:
        cleaned = str(work / f"{stem}.1_clean.mp4")
        print(f"[clean] auto-editor threshold={preset.silence_threshold} "
              f"margin={preset.margin}")
        cur = _clean.clean(cur, cleaned, preset)

    # 2) reframe (reframe.py inspects source aspect and logs its own decision)
    if preset.reframe_mode != "none" or (info.width, info.height) != (preset.target_w, preset.target_h):
        framed = str(work / f"{stem}.2_frame.mp4")
        cur = _reframe.reframe(cur, framed, preset)

    # 3) transcribe + 4) caption
    words: list = []
    srt = ass = ""
    words_json = ""
    if info.has_audio and preset.caption_style != "none":
        print(f"[transcribe] whisper model={_tx.config.WHISPER_MODEL} "
              f"backend={_tx.config.TRANSCRIBE_BACKEND}")
        words = _tx.transcribe(cur, language=language)
        print(f"[transcribe] {len(words)} words")
        srt = _tx.write_srt(words, str(work / f"{stem}.srt"), max_words=preset.max_words_per_line + 4)
        words_json = _tx.dump_words(words, str(work / f"{stem}.words.json"))

        cinfo = probe(cur)
        ass_content = _cap.build_ass(
            words, cinfo.width, cinfo.height,
            style=preset.caption_style, max_words=preset.max_words_per_line,
        )
        ass = _cap.write_ass(ass_content, str(work / f"{stem}.ass"))
        final = str(work / f"{stem}.final.mp4")
        print(f"[caption] burning {preset.caption_style} captions")
        _cap.burn(cur, final, ass)
    else:
        final = str(work / f"{stem}.final.mp4")
        shutil.copy(cur, final)

    out_info = probe(final)
    if not keep_intermediates:
        for p in work.glob(f"{stem}.[12]_*.mp4"):
            p.unlink(missing_ok=True)

    print(f"[done] {final}  ({out_info.duration:.1f}s, "
          f"trimmed {info.duration - out_info.duration:.1f}s)")
    return Result(
        final=final, srt=srt, ass=ass, words_json=words_json,
        n_words=len(words), duration_out=out_info.duration,
    )
