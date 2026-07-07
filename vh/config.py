"""Central config: binary paths + per-source-type presets.

Binaries resolve from PATH by default (portable). If your ffmpeg/python live in
an isolated env that isn't on PATH, point VH_BIN at that env's bin dir (e.g.
`export VH_BIN=/path/to/conda/envs/myenv/bin`) — every tool is then taken from
there — or override each one individually (VH_FFMPEG, VH_FFPROBE, ...). Setting
VH_BIN also avoids depending on `conda activate`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

# VH_BIN pins an isolated env's bin dir; unset -> resolve each tool from PATH.
VHARNESS_BIN = os.environ.get("VH_BIN", "").rstrip("/")

def _bin(name: str, env: str) -> str:
    val = os.environ.get(env)
    if val:
        return val
    return f"{VHARNESS_BIN}/{name}" if VHARNESS_BIN else name

FFMPEG = _bin("ffmpeg", "VH_FFMPEG")
FFPROBE = _bin("ffprobe", "VH_FFPROBE")
PYTHON = _bin("python", "VH_PYTHON") if (VHARNESS_BIN or os.environ.get("VH_PYTHON")) else sys.executable
AUTO_EDITOR = _bin("auto-editor", "VH_AUTO_EDITOR")

# Whisper: CPU int8 by default (GB10 / sm_121 has no CTranslate2 CUDA wheel yet).
WHISPER_MODEL = os.environ.get("VH_WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("VH_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("VH_WHISPER_COMPUTE", "int8")
# GB10 Grace CPU has ~20 cores — let CTranslate2 use them (default is 4).
WHISPER_CPU_THREADS = int(os.environ.get("VH_WHISPER_THREADS", str(min(16, (os.cpu_count() or 4)))))

# Transcription backend:
#   "auto" -> remote if a render host is configured, else gpu.
#   "remote" -> offload to VH_RENDER_HOST (faster-whisper CUDA on x86_64 ~59x RT).
#   "gpu" -> transformers Whisper on the local GPU via GPU_PYTHON (steps/gpu_asr.py).
#   "cpu" -> in-process faster-whisper int8 (portable but slow here).
TRANSCRIBE_BACKEND = os.environ.get("VH_ASR_BACKEND", "auto")
GPU_PYTHON = os.environ.get("VH_GPU_PYTHON") or sys.executable or "python3"

# Remote render host — OPT-IN, user-provided, never hardcoded. Heavy stages
# (transcription; later encoding) offload here so the local GPU is left alone.
# Unset -> everything runs locally.
#   export VH_RENDER_HOST="user@host"   VH_RENDER_PORT="7777"
#   export VH_RENDER_PYTHON="/path/to/env/bin/python"   # must have faster-whisper+CUDA
RENDER_HOST = os.environ.get("VH_RENDER_HOST")            # "" / unset -> local only
RENDER_PORT = os.environ.get("VH_RENDER_PORT")            # optional ssh port
RENDER_PYTHON = os.environ.get("VH_RENDER_PYTHON", "python3")
RENDER_TMP = os.environ.get("VH_RENDER_TMP", "/tmp")
# Remote encoding: ffmpeg on the host + its font dir (defaults assume the host
# has ffmpeg on PATH and the same Noto path; override if not).
RENDER_FFMPEG = os.environ.get("VH_RENDER_FFMPEG", "ffmpeg")
# mirror the caption font dir by default (defined above, same env fallback)
RENDER_FONTSDIR = os.environ.get(
    "VH_RENDER_FONTSDIR",
    os.environ.get("VH_CAPTION_FONTSDIR", "/usr/share/fonts/opentype/noto"),
)
# Persistent input cache on the host: an uploaded file (keyed by size+mtime+name)
# is reused across ffmpeg calls / re-renders instead of re-uploading. Empty ->
# default {RENDER_TMP}/vh_cache. remote.clear_cache() wipes it.
RENDER_CACHE = os.environ.get("VH_RENDER_CACHE", "")


def render_host_enabled() -> bool:
    return bool(RENDER_HOST)

# Caption font. Noto Sans CJK KR renders Korean/CJK (Arial => tofu boxes).
# fontsdir is handed to libass explicitly because conda ffmpeg's fontconfig
# doesn't always scan the system font dirs.
CAPTION_FONT = os.environ.get("VH_CAPTION_FONT", "Noto Sans CJK KR")
CAPTION_FONTSDIR = os.environ.get(
    "VH_CAPTION_FONTSDIR", "/usr/share/fonts/opentype/noto"
)

# Video encoder. GB10 has NVENC — use it (CPU x264 is far too slow for 1080p60,
# 30-min sources). Set VH_VENC=libx264 to force software.
VIDEO_ENCODER = os.environ.get("VH_VENC", "h264_nvenc")
AUDIO_ARGS = ["-c:a", "aac", "-b:a", "192k"]


def video_args() -> list[str]:
    if VIDEO_ENCODER == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23",
                "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p"]


def encode_args() -> list[str]:
    """Video + audio + faststart flags for a final-ish mp4."""
    return video_args() + AUDIO_ARGS + ["-movflags", "+faststart"]


@dataclass
class Preset:
    """Tuning knobs per source type."""
    name: str
    # auto-editor silence trimming
    clean: bool = True
    silence_threshold: str = "4%"     # audio level below this = "silent"
    margin: str = "0.2sec"            # keep this much padding around kept audio
    # captioning
    caption_style: str = "word"       # "word" (CapCut-style pop) | "line" | "none"
    max_words_per_line: int = 4
    # output framing
    aspect: str = "16:9"              # "16:9" long-form | "9:16" shorts
    reframe_mode: str = "pad"         # "pad" (blur bars) | "crop" (center) | "none"
    target_w: int = 1920
    target_h: int = 1080
    allow_upscale: bool = False       # if src aspect matches target, upscale to target anyway?
    # boxed layout: landscape -> vertical with header/caption zones (see compose.py)
    layout: str = "fill"              # "fill" (reframe+caption) | "boxed" (3-zone)
    bg: str = "solid"                 # boxed background: "solid" | "blur"
    bg_color: str = "0x0B0B14"        # solid band colour
    video_title: str | None = None    # fixed header when there are no chapters


PRESETS: dict[str, Preset] = {
    # Screen recordings / lectures: aggressive silence cut, readable line captions.
    "screencast": Preset(
        name="screencast",
        silence_threshold="3%",
        margin="0.25sec",
        caption_style="line",
        max_words_per_line=6,
        aspect="16:9", reframe_mode="none",
        target_w=1920, target_h=1080,
    ),
    # Talking head (webcam/camera): gentle cut, punchy word captions.
    "talkinghead": Preset(
        name="talkinghead",
        silence_threshold="4%",
        margin="0.3sec",
        caption_style="word",
        max_words_per_line=4,
        aspect="16:9", reframe_mode="none",
        target_w=1920, target_h=1080,
    ),
    # Vertical Shorts: word captions, 9:16 blur-pad reframe.
    "shorts": Preset(
        name="shorts",
        silence_threshold="4%",
        margin="0.15sec",
        caption_style="word",
        max_words_per_line=3,
        aspect="9:16", reframe_mode="pad",
        target_w=1080, target_h=1920,
    ),
    # Landscape -> vertical Shorts with 3-zone layout: header on top, video in
    # the middle band, captions on the bottom. Best for reframing 16:9 talks.
    "shorts_boxed": Preset(
        name="shorts_boxed",
        clean=False,                  # keep timeline simple; enable if needed
        caption_style="word",
        max_words_per_line=4,
        aspect="9:16",
        target_w=1080, target_h=1920,
        layout="boxed", bg="solid",
    ),
    # Slides / narration-over-static: don't over-cut, line captions.
    "slides": Preset(
        name="slides",
        silence_threshold="2%",
        margin="0.4sec",
        caption_style="line",
        max_words_per_line=7,
        aspect="16:9", reframe_mode="none",
        target_w=1920, target_h=1080,
    ),
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise SystemExit(
            f"unknown preset '{name}'. choices: {', '.join(PRESETS)}"
        )
    return PRESETS[name]
