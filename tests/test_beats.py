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
