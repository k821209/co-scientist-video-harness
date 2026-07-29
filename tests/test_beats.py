"""Beat coercion: episode files already carry plain 7-tuples — keep them working."""
from vh.steps.beats import Beat


def test_tuple_is_accepted():
    b = Beat.coerce(("b1", "gfx", "본문 나레이션", "g_lineup", 0.0, None, None))
    assert (b.id, b.kind, b.visual, b.at) == ("b1", "gfx", "g_lineup", 0.0)
    assert b.caption is None and b.credit is None


def test_b0_with_caption_becomes_eyecatch():
    """The cold open's caption is the punch line, rendered big and centred."""
    b = Beat.coerce(("b0", "clip", "훅", "keynote", 11.0, "폴드7의 후속이 아니다", "영상 · 공식"))
    assert b.eyecatch and b.at == 11.0


def test_later_beat_caption_is_not_eyecatch():
    b = Beat.coerce(("b3", "clip", "본문", "film", 4.0, "메인 7.6형", "영상 · 공식"))
    assert not b.eyecatch


def test_beat_passes_through():
    b = Beat("b2", "photo", "본문", "img/x.jpg", caption="캡션")
    assert Beat.coerce(b) is b


# ── segment reuse fingerprint (feedback 8587d866794a) ──────────────────────

def test_seg_fingerprint_stable_and_sensitive(tmp_path):
    from vh.steps.beats import Beat, _seg_fingerprint
    card = tmp_path / "g_open.png"; card.write_bytes(b"CARDv1")
    ov = tmp_path / "ov_b1.png"; ov.write_bytes(b"OVERLAYv1")
    b = Beat.coerce(("b1", "gfx", "본문", "g_open", 0.0, None, None))
    kw = dict(gfx=tmp_path, clips={}, precrop={}, fps=25, canvas=(1080, 1920),
              encode_args=["-c:v", "mpeg4"])
    fp1 = _seg_fingerprint(b, 7.3, ov, **kw)
    assert fp1 == _seg_fingerprint(b, 7.3, ov, **kw)          # stable
    assert fp1 != _seg_fingerprint(b, 9.0, ov, **kw)          # VO length changed
    ov.write_bytes(b"OVERLAYv2-caption-edit")
    assert fp1 != _seg_fingerprint(b, 7.3, ov, **kw)          # overlay changed
    card.write_bytes(b"CARDv2-relaid-out-and-longer")           # size+mtime change
    assert fp1 != _seg_fingerprint(b, 7.3, tmp_path / "ov_b1.png", **kw)  # source card changed


def test_seg_fingerprint_clip_tracks_precrop(tmp_path):
    from vh.steps.beats import Beat, _seg_fingerprint
    clip = tmp_path / "keynote.mp4"; clip.write_bytes(b"CLIPBYTES")
    ov = tmp_path / "ov_b0.png"; ov.write_bytes(b"OV")
    b = Beat.coerce(("b0", "clip", "훅", "keynote", 11.0, "cap", "cred"))
    base = dict(gfx=tmp_path, clips={"keynote": str(clip)}, fps=25,
                canvas=(1080, 1920), encode_args=["-c:v", "mpeg4"])
    fp1 = _seg_fingerprint(b, 7.3, ov, precrop={}, **base)
    fp2 = _seg_fingerprint(b, 7.3, ov, precrop={"keynote": "crop=iw:ih*0.9:0:0,"}, **base)
    assert fp1 != fp2                                          # precrop is part of identity


def test_final_encode_rejects_bad_value(tmp_path):
    """final_encode is validated up-front (before any ffmpeg work)."""
    import pytest
    from vh.steps.beats import build_beat_short
    with pytest.raises(ValueError, match="final_encode"):
        build_beat_short([("b1", "gfx", "x", "g", 0.0, None, None)],
                         str(tmp_path / "o.mp4"), workdir=str(tmp_path / "wd"),
                         final_encode="nope")


def test_seg_fingerprint_uses_content_hash_for_cards(tmp_path):
    """A card-render script rewrites every png, so mtime must NOT invalidate a
    segment whose card content is identical (feedback 7b0e9a266d7c item 2)."""
    import os
    import time
    from vh.steps.beats import Beat, _seg_fingerprint
    card = tmp_path / "g_open.png"; card.write_bytes(b"IDENTICAL-CARD-BYTES")
    ov = tmp_path / "ov.png"; ov.write_bytes(b"OV")
    b = Beat.coerce(("b1", "gfx", "본문", "g_open", 0.0, None, None))
    kw = dict(gfx=tmp_path, clips={}, precrop={}, fps=25, canvas=(1080, 1920),
              encode_args=["-c:v", "mpeg4"])
    fp1 = _seg_fingerprint(b, 7.3, ov, **kw)
    # same bytes, new mtime (script redrew the card unchanged)
    os.utime(card, (time.time() + 60, time.time() + 60))
    assert _seg_fingerprint(b, 7.3, ov, **kw) == fp1          # cache survives
    card.write_bytes(b"REALLY-DIFFERENT-CARD")                 # real edit
    assert _seg_fingerprint(b, 7.3, ov, **kw) != fp1


def test_seg_fingerprint_tracks_vo_content(tmp_path):
    """Re-recorded narration of the SAME length still invalidates the segment."""
    from vh.steps.beats import Beat, _seg_fingerprint
    card = tmp_path / "g.png"; card.write_bytes(b"C")
    ov = tmp_path / "ov.png"; ov.write_bytes(b"OV")
    vo = tmp_path / "b1.mp3"; vo.write_bytes(b"OLD-AUDIO")
    b = Beat.coerce(("b1", "gfx", "본문", "g", 0.0, None, None))
    kw = dict(gfx=tmp_path, clips={}, precrop={}, fps=25, canvas=(1080, 1920),
              encode_args=["-c:v", "mpeg4"])
    fp1 = _seg_fingerprint(b, 7.3, ov, vo_path=vo, **kw)
    vo.write_bytes(b"NEW-AUDIO")                              # same byte-length
    assert _seg_fingerprint(b, 7.3, ov, vo_path=vo, **kw) != fp1


# ── ref_video: quoted clip inside a card window (feedback 49f3ca81469d) ─────

def test_ref_video_without_a_window_raises_with_guidance(tmp_path):
    import pytest
    from vh.steps.beats import build_beat_short
    (tmp_path / "wd").mkdir()
    with pytest.raises(ValueError, match="no window"):
        build_beat_short([("b0", "gfx", "훅", "g_hook", 0.0, None, None)],
                         str(tmp_path / "o.mp4"), workdir=str(tmp_path / "wd"),
                         gfx_dir=str(tmp_path), ref_video={"g_hook": "clip.mp4"})


def test_seg_fingerprint_tracks_ref_video_and_in_point(tmp_path):
    from vh.steps.beats import Beat, _seg_fingerprint
    card = tmp_path / "g_hook.png"; card.write_bytes(b"CARD")
    ov = tmp_path / "ov.png"; ov.write_bytes(b"OV")
    clip = tmp_path / "ref.mp4"; clip.write_bytes(b"REFCLIP")
    b = Beat.coerce(("b0", "gfx", "훅", "g_hook", 0.0, None, None))
    kw = dict(gfx=tmp_path, clips={}, precrop={}, fps=25, canvas=(1080, 1920),
              encode_args=["-c:v", "mpeg4"], window=[60, 320, 960, 540])
    fp_at3 = _seg_fingerprint(b, 6.5, ov, ref=(str(clip), 3.0), **kw)
    assert fp_at3 != _seg_fingerprint(b, 6.5, ov, ref=(str(clip), 9.0), **kw)  # in-point
    assert fp_at3 != _seg_fingerprint(b, 6.5, ov, ref=None, **kw)              # window on/off
    kw2 = {**kw, "window": [60, 320, 720, 405]}
    assert fp_at3 != _seg_fingerprint(b, 6.5, ov, ref=(str(clip), 3.0), **kw2)  # window size


# ── long ref_video warning + dry_run + credit skip (2026-07-29 batch) ───────

def test_long_reference_video_warns(tmp_path, capsys, monkeypatch):
    """A source much longer than the beat renders only its head — a generated
    diagram's payoff never reaches the screen (feedback 2ad1accde731)."""
    from vh.steps import beats as B
    monkeypatch.setattr(B, "_dur", lambda p: 34.8 if "two" in str(p) else 12.7)
    monkeypatch.setattr(B, "_run", lambda args: None)
    monkeypatch.setattr(B.news, "edge_tts_speak", lambda *a, **k: None)
    (tmp_path / "wd" / "vo").mkdir(parents=True)
    (tmp_path / "wd" / "vo" / "b0.mp3").write_bytes(b"x")
    (tmp_path / "g.png").write_bytes(b"png")
    (tmp_path / "g.windows.json").write_text('{"g": [60, 320, 960, 540]}')
    (tmp_path / "two.mp4").write_bytes(b"clip")
    try:
        B.build_beat_short([("b0", "gfx", "본문", "g", 0.0, None, None)],
                           str(tmp_path / "o.mp4"), workdir=str(tmp_path / "wd"),
                           gfx_dir=str(tmp_path), lint_script=False,
                           ref_video={"g": (str(tmp_path / "two.mp4"), 0.0)})
    except Exception:
        pass                                        # later stages are stubbed
    out = capsys.readouterr().out
    assert "never" in out and "34.8s" in out and "diagram" in out


def test_dry_run_returns_lengths_without_rendering(tmp_path, monkeypatch):
    from vh.steps import beats as B
    monkeypatch.setattr(B, "_dur", lambda p: 6.0 if "b0" in str(p) else 9.5)
    monkeypatch.setattr(B.news, "edge_tts_speak", lambda *a, **k: None)
    monkeypatch.setattr(B, "_run", lambda args: (_ for _ in ()).throw(
        AssertionError("dry_run must not render")))
    (tmp_path / "wd" / "vo").mkdir(parents=True)
    for bid in ("b0", "b1"):
        (tmp_path / "wd" / "vo" / f"{bid}.mp3").write_bytes(b"x")
    r = B.build_beat_short(
        [("b0", "gfx", "가", "g_a", 0.0, None, None),
         ("b1", "gfx", "나", "g_b", 0.0, None, None)],
        str(tmp_path / "o.mp4"), workdir=str(tmp_path / "wd"),
        gfx_dir=str(tmp_path), dry_run=True, lint_script=False)
    assert r["dry_run"] and r["total"] == 16.5          # (6.0+.5) + (9.5+.5)
    assert [b["seg"] for b in r["beats"]] == [6.5, 10.0]


def test_credit_skipped_when_card_draws_its_own_source(tmp_path, capsys, monkeypatch):
    """A screen-pinned credit drifts into the card's own source line as the card
    zooms, so the card's line wins (feedback bf8839249a77)."""
    from vh.steps import beats as B
    monkeypatch.setattr(B, "_dur", lambda p: 3.0)
    monkeypatch.setattr(B, "_run", lambda args: None)
    monkeypatch.setattr(B.news, "edge_tts_speak", lambda *a, **k: None)
    captured = {}
    monkeypatch.setattr(B, "_overlay",
                        lambda beat, path, *a, **k: captured.setdefault("credit", beat.credit) or path)
    (tmp_path / "wd" / "vo").mkdir(parents=True)
    (tmp_path / "wd" / "vo" / "b0.mp3").write_bytes(b"x")
    (tmp_path / "g_a.png").write_bytes(b"png")
    (tmp_path / "g_a.windows.json").write_text('{"_reserved_bottom": [[46, 1824, 428, 28]]}')
    try:
        B.build_beat_short([("b0", "gfx", "본문", "g_a", 0.0, None, "사진 · Wikimedia")],
                           str(tmp_path / "o.mp4"), workdir=str(tmp_path / "wd"),
                           gfx_dir=str(tmp_path), lint_script=False)
    except Exception:
        pass
    assert captured["credit"] is None                     # overlay got no credit
    assert "draws its own source line" in capsys.readouterr().out
