"""workdir resolution for build_short / build_clip_short (no ffmpeg).

Guards two shipped regressions: a relative workdir broke ffmpeg concat, and
'~/…' silently created a literal '~' dir under cwd instead of $HOME.
"""
import os
import pathlib
from vh.steps.news import _workdir


def test_relative_becomes_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wd = _workdir("wd_rel", "p_")
    assert wd.is_absolute()
    assert wd == (tmp_path / "wd_rel").resolve()


def test_tilde_expands_to_home():
    wd = _workdir("~/vh_test_xyz", "p_")
    assert wd == pathlib.Path.home() / "vh_test_xyz"
    assert "~" not in str(wd)


def test_default_is_absolute_tempdir():
    wd = _workdir(None, "p_")
    assert wd.is_absolute() and wd.is_dir()
