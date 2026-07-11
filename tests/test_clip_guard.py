"""Behavior locks for the clip-span tail fix (9c75854), via two named seams so
we test what the code DOES, not how it's spelled (feedback dae63ba45df2 →
f9e8f3f47cb5: source-string tests missed a loosened threshold and false-flagged
an equivalent refactor)."""
import inspect

import pytest

from vh.steps import news


def test_clip_pad_fills_full_span_no_fixed_cap():
    # a short clip under a long sentence must hold for the WHOLE span
    assert news._clip_pad(20.0) == pytest.approx(20.0)   # NOT capped at 4
    assert news._clip_pad(2.0) == pytest.approx(2.0)


def test_output_guard_raises_on_dropped_tail():
    with pytest.raises(RuntimeError):
        news._check_output_duration(20.57, 23.37)        # the real bug's numbers
    news._check_output_duration(63.50, 63.53)            # within tol → no raise


def test_output_guard_threshold_stays_tight():
    # loosening the tolerance (e.g. to 100) would let a dropped tail through —
    # this 3s gap must always raise
    with pytest.raises(RuntimeError):
        news._check_output_duration(10.0, 13.0)


def test_both_shorts_use_the_seams():
    # a revert to an inline stop_duration=4 / no guard drops these calls
    assert "_clip_pad(" in inspect.getsource(news.build_clip_short)
    for fn in (news.build_short, news.build_clip_short):
        assert "_check_output_duration(" in inspect.getsource(fn)


def test_clip_short_fit_and_fill_options():
    import inspect
    p = inspect.signature(news.build_clip_short).parameters
    assert p["fill"].default == "freeze"                  # backward-compatible default
    src = inspect.getsource(news.build_clip_short)
    assert "force_original_aspect_ratio=decrease" in src and "pad=" in src   # fit=letterbox
    assert "-stream_loop" in src                          # fill="loop"
    assert "warnings.warn" in src                         # short-clip warning


def test_norm_clip_shots_unifies_arity():
    # 4- and 5-element shots both normalize to a 5-tuple that unpacks the same
    # way in every loop (the regression: one loop unpacked 4, another 5)
    four = news._norm_clip_shots([("a", "c", False, "Reuters")])
    five = news._norm_clip_shots([("a", "c", False, "Reuters", {"fit": True})])
    assert four[0] == ("a", "c", False, "Reuters", {})
    assert five[0][4] == {"fit": True}
    for anchor, clip, is_vlog, cred, opts in four + five:   # must not raise
        assert isinstance(opts, dict)
