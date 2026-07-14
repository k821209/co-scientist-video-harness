"""Fast color-mode smoke test for build_rank_race (HANDOFF TODO #2).

Renders a tiny weighted-Voronoi race and asserts the summary: output exists,
step/entry counts, per-step leaders (incl. a lead change), and frame/duration
math. The host codec (config.video_args → nvenc/libx264) is monkeypatched to a
native encoder so the test runs anywhere ffmpeg is present, decoupled from the
render box's H.264 build.
"""
import shutil

import pytest

from vh import config
from vh.steps import build_rank_race          # re-export surface (steps/__init__)
from vh.steps.rank_race import build_rank_race as build_rank_race_direct

pytestmark = pytest.mark.skipif(
    shutil.which(config.FFMPEG) is None and shutil.which("ffmpeg") is None,
    reason="needs ffmpeg",
)


def test_reexport_matches_module():
    assert build_rank_race is build_rank_race_direct


def test_color_mode_summary_and_leaders(tmp_path, monkeypatch):
    # Native, universally-present encoder → no dependency on the host's H.264.
    monkeypatch.setattr(config, "video_args",
                        lambda: ["-c:v", "mpeg4", "-q:v", "5", "-pix_fmt", "yuv420p"])
    out = tmp_path / "race.mp4"
    res = build_rank_race(
        entries=[("a", "Alpha", (200, 40, 60)), ("b", "Beta", (40, 120, 220))],
        series={"a": [3, 1], "b": [1, 3]},        # leader flips Alpha -> Beta
        labels=["T1", "T2"],
        out=str(out),
        fill="color", headline="테스트 레이스",
        hold_s=0.3, morph_s=0.2, fps=10,
        w=320, h=560, grid=(24, 42),
    )
    assert out.exists() and out.stat().st_size > 0
    assert res["fill"] == "color"
    assert res["steps"] == 2 and res["entries"] == 2
    assert res["leaders"] == ["Alpha", "Beta"]    # per-step #1 tracked incl. the flip
    assert res["follow_leader"] is False
    # frames = HOLD(0.3*10=3) + MORPH(0.2*10=2) + HOLD(3) = 8
    assert res["frames"] == 8
    assert res["duration"] == round(8 / 10, 2)


def test_clip_mode_requires_clips(tmp_path):
    with pytest.raises(ValueError, match="clip"):
        build_rank_race(
            entries=[("a", "Alpha", (200, 40, 60)), ("b", "Beta", (40, 120, 220))],
            series={"a": [1, 2], "b": [2, 1]}, labels=["T1", "T2"],
            out=str(tmp_path / "x.mp4"), fill="clip",   # no clips= → error
        )
