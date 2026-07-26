"""contact_sheet grid auto-sizing (feedback 8587d866794a item 3)."""
from vh import qc


def _tile_arg(monkeypatch, dur, **kw):
    captured = {}
    monkeypatch.setattr(qc, "_duration", lambda src: dur)
    def fake_run(args, **k):
        vf = args[args.index("-vf") + 1]
        captured["tile"] = [p for p in vf.split(",") if p.startswith("tile=")][0]
        return None
    monkeypatch.setattr(qc.subprocess, "run", fake_run)
    qc.contact_sheet("in.mp4", "out.png", **kw)
    c, r = captured["tile"].removeprefix("tile=").split("x")
    return int(c), int(r)


def test_auto_grid_covers_all_tiles(monkeypatch):
    # 98s / every 8s = 13 tiles → grid must hold >= 13, near-square
    cols, rows = _tile_arg(monkeypatch, 98.0, every=8.0)
    assert cols * rows >= 13 and abs(cols - rows) <= 1


def test_partial_grid_fills_missing_dim(monkeypatch):
    cols, rows = _tile_arg(monkeypatch, 98.0, every=8.0, cols=7)
    assert cols == 7 and cols * rows >= 13


def test_explicit_grid_is_respected(monkeypatch):
    cols, rows = _tile_arg(monkeypatch, 98.0, every=8.0, cols=7, rows=2)
    assert (cols, rows) == (7, 2)
