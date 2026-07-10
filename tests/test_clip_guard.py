"""Lock the build_clip_short tail fix (9c75854).

The fix is inside the function body (an ffmpeg command + a runtime duration
guard), so it can't be reached by signature inspection and a full render needs
ffmpeg + network TTS. Source inspection is the pragmatic ffmpeg-free lock — a
mutation reverting either half turns one of these red (verified against the
reporter's mutation A/B).
"""
import inspect
from vh.steps import news


def test_clip_fills_full_span_not_a_fixed_cap():
    src = inspect.getsource(news.build_clip_short)
    assert "stop_duration=4," not in src          # the removed hardcoded 4s cap
    assert "stop_duration={spans[i]" in src       # per-sentence-span hold


def test_both_shorts_guard_output_length():
    # both must fail loud if the end card / tail is silently dropped
    for fn in (news.build_short, news.build_clip_short):
        src = inspect.getsource(fn)
        assert "abs(got - total)" in src
        assert "RuntimeError" in src
