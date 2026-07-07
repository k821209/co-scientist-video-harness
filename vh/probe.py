"""ffprobe helpers + a thin ffmpeg runner."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from . import config


def run(cmd: list[str], quiet: bool = True) -> subprocess.CompletedProcess:
    """Run a command, raising with captured stderr on failure."""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"command failed ({proc.returncode}):\n  {' '.join(cmd)}\n{tail}")
    if not quiet and proc.stderr:
        print(proc.stderr.strip().splitlines()[-1])
    return proc


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    has_audio: bool
    fps: float


def probe(path: str) -> MediaInfo:
    proc = run([
        config.FFPROBE, "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    data = json.loads(proc.stdout)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        raise RuntimeError(f"no video stream in {path}")
    num, den = (v.get("avg_frame_rate", "0/1").split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) else 0.0
    return MediaInfo(
        duration=float(data["format"].get("duration", 0.0)),
        width=int(v["width"]),
        height=int(v["height"]),
        has_audio=a is not None,
        fps=fps,
    )
