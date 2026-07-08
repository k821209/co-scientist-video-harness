"""Regression guard for mux_audio length handling.

This exact bug shipped once: _media_duration returned 0.0 (a str/bytes mistake),
so the audio-longer freeze-frame branch went dead and video was silently
truncated — and it stayed green if you only checked the audio-SHORTER case.
So we assert BOTH directions. Skipped when ffmpeg/ffprobe aren't on PATH.
"""
import os
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe not available",
)


def _encoders() -> str:
    try:
        return subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                              capture_output=True, text=True).stdout
    except Exception:
        return ""


# config.video_args() only emits h264_nvenc or libx264 — the freeze branch
# re-encodes, so it needs one of them present.
_H264 = next((e for e in ("h264_nvenc", "libx264") if e in _encoders()), None)


def _mk_video(path, secs):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    f"color=c=blue:s=320x240:d={secs}", "-pix_fmt", "yuv420p", path],
                   check=True, capture_output=True)


def _mk_audio(path, secs):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={secs}", path],
                   check=True, capture_output=True)


@pytest.fixture
def media(tmp_path):
    if _H264:
        os.environ["VH_VENC"] = _H264
    v5, v2 = str(tmp_path / "v5.mp4"), str(tmp_path / "v2.mp4")
    a5, a2 = str(tmp_path / "a5.wav"), str(tmp_path / "a2.wav")
    _mk_video(v5, 5); _mk_video(v2, 2); _mk_audio(a5, 5); _mk_audio(a2, 2)
    return {"v5": v5, "v2": v2, "a5": a5, "a2": a2, "dir": str(tmp_path)}


def test_media_duration_nonzero(media):
    from vh.steps.dub import _media_duration
    assert abs(_media_duration(media["v5"]) - 5.0) < 0.2   # 0.0 here = the regression
    assert abs(_media_duration(media["a2"]) - 2.0) < 0.2


def test_mux_preserves_video_tail(media):
    """Audio shorter than video (e.g. a silent end card) → keep the full video."""
    from vh.steps.dub import mux_audio, _media_duration
    out = media["dir"] + "/A.mp4"
    mux_audio(media["v5"], media["a2"], out)
    assert abs(_media_duration(out) - 5.0) < 0.2


@pytest.mark.skipif(_H264 is None, reason="no h264 encoder (nvenc/libx264) to re-encode")
def test_mux_does_not_clip_audio(media):
    """Audio longer than video → freeze last frame, don't cut narration.
    This is the branch that was dead when _media_duration returned 0.0."""
    from vh.steps.dub import mux_audio, _media_duration
    out = media["dir"] + "/B.mp4"
    mux_audio(media["v2"], media["a5"], out)
    assert abs(_media_duration(out) - 5.0) < 0.2
