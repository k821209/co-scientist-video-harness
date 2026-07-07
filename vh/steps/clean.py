"""Silence / dead-air removal — native ffmpeg (no external binary).

auto-editor 29.x ships an x86_64-only compiled core and won't run on aarch64
(GB10), so we detect silence with ffmpeg's `silencedetect` and drop the silent
spans ourselves with select/aselect. Timing-preserving: word timestamps taken
from the cleaned output stay valid.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .. import config
from ..probe import run, probe


def _threshold_ratio(spec: str) -> float:
    """'3%' -> 0.03, '0.04' -> 0.04."""
    spec = spec.strip()
    if spec.endswith("%"):
        return float(spec[:-1]) / 100.0
    return float(spec)


def _margin_sec(spec: str) -> float:
    """'0.25sec' -> 0.25, '250ms' -> 0.25, '0.3' -> 0.3."""
    spec = spec.strip().lower()
    if spec.endswith("sec"):
        return float(spec[:-3])
    if spec.endswith("ms"):
        return float(spec[:-2]) / 1000.0
    if spec.endswith("s"):
        return float(spec[:-1])
    return float(spec)


_SIL_START = re.compile(r"silence_start:\s*(-?\d+\.?\d*)")
_SIL_END = re.compile(r"silence_end:\s*(-?\d+\.?\d*)")


def _detect_silences(src: str, noise: float, min_dur: float) -> list[tuple[float, float]]:
    proc = subprocess.run(
        [config.FFMPEG, "-hide_banner", "-i", str(src),
         "-af", f"silencedetect=noise={noise}:d={min_dur}", "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    log = proc.stderr
    starts = [float(m) for m in _SIL_START.findall(log)]
    ends = [float(m) for m in _SIL_END.findall(log)]
    spans: list[tuple[float, float]] = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else None
        spans.append((s, e if e is not None else s))  # open span closed by caller
    return spans


def _keep_intervals(duration: float, silences: list[tuple[float, float]],
                    margin: float, min_keep: float = 0.05) -> list[tuple[float, float]]:
    """Complement of silences within [0, duration], padded by `margin`."""
    # shrink each silence by margin on both sides (=> keep a little around speech)
    padded = []
    for s, e in silences:
        s2, e2 = s + margin, e - margin
        if e2 > s2:
            padded.append((s2, e2))
    keep, cursor = [], 0.0
    for s, e in padded:
        if s > cursor:
            keep.append((cursor, min(s, duration)))
        cursor = max(cursor, e)
    if cursor < duration:
        keep.append((cursor, duration))
    return [(a, b) for a, b in keep if b - a >= min_keep]


def clean(src: str, dst: str, preset) -> str:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    info = probe(src)
    noise = _threshold_ratio(preset.silence_threshold)
    margin = _margin_sec(preset.margin)
    min_sil = getattr(preset, "min_silence", 0.35)

    silences = _detect_silences(src, noise, min_sil)
    keep = _keep_intervals(info.duration, silences, margin)

    # nothing to cut (or detector found everything silent) -> passthrough copy
    if not silences or not keep or len(keep) == 1 and keep[0] == (0.0, info.duration):
        run([config.FFMPEG, "-y", "-i", str(src), "-c", "copy", str(dst)])
        return dst

    sel = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in keep)
    vf = f"select='{sel}',setpts=N/FRAME_RATE/TB"
    af = f"aselect='{sel}',asetpts=N/SR/TB"
    from .. import remote
    remote.ffmpeg_run([
        config.FFMPEG, "-y", "-i", str(src),
        "-vf", vf, "-af", af,
        *config.encode_args(),
        str(dst),
    ], reads=[str(src)], write=str(dst))
    return dst
