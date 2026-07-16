"""Weighted-Voronoi rank race (9:16 / 16:9).

A "ranking race" video: the frame is partitioned into polygon cells whose AREA
is proportional to each entry's value, and the partition MORPHS over time as the
values change. Each cell is filled either with the entry's brand colour
(`fill="color"`) or with a looping video clip cover-fit to the cell
(`fill="clip"`, the K-pop / idol-trend look). Optionally the background music
FOLLOWS THE LEADER: whoever is #1 at a given moment, their track plays, and the
BGM cross-fades to the new leader's track when the top changes.

No external deps beyond numpy + Pillow (the Voronoi solver is a pure-numpy pixel
power-diagram; no scipy). Encoding uses the shared `config` binaries.

    from vh.steps.rank_race import build_rank_race

    build_rank_race(
        entries=[("exaone","EXAONE",(205,30,95)), ("clova","HyperCLOVA X",(3,199,90)), ...],
        series={"exaone":[0.20,0.24,0.29], "clova":[0.30,0.12,0.02], ...},  # per label
        labels=["1월","2월","3월"],
        out="race.mp4",
        headline="국내 AI 모델 관심도 레이스",
        fill="color",                       # or "clip"
        # clips={"exaone":"exaone.mp4", ...},        # required if fill="clip"
        # audio={"exaone":"exaone.wav", ...}, follow_leader=True,   # leader BGM
    )

fill="clip"/audio note: source clips responsibly (official channels, on-screen
credit, news/critique purpose — Content-ID risk remains). Use `news.fetch_clip`
(video-only for cells, keep_audio=True for the leader track).
"""
from __future__ import annotations

import math
import pathlib
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import config

_FONTDIR = config.CAPTION_FONTSDIR


def _font(bold: bool, size: int):
    name = "NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc"
    return ImageFont.truetype(f"{_FONTDIR}/{name}", size, index=0)


# ── weighted Voronoi (pixel power-diagram) solver ────────────────────────────
def _solver_grid(gw: int, gh: int):
    ys, xs = np.mgrid[0:gh, 0:gw]
    return xs.astype(np.float64), ys.astype(np.float64)


def _assign(xs, ys, pts, w):
    """Pixel -> argmin_i (|p - site_i|^2 - w_i).  Returns label map (gh, gw)."""
    d2 = (xs[..., None] - pts[:, 0]) ** 2 + (ys[..., None] - pts[:, 1]) ** 2 - w[None, None, :]
    return np.argmin(d2, axis=2)


def _solve(targets, pts, w, xs, ys, total, iters=70, lr=0.9):
    """Adapt weights (w) + Lloyd-relax sites (pts) so cell areas match `targets`."""
    n = len(targets)
    for _ in range(iters):
        lab = _assign(xs, ys, pts, w)
        area = np.bincount(lab.ravel(), minlength=n) / total
        for k in range(n):
            m = lab == k
            if m.any():
                pts[k] = [xs[m].mean(), ys[m].mean()]
        w += lr * (targets - area) * (total / n)
        w -= w.mean()
    return pts.copy(), w.copy()


def _keyframes(shares, gw, gh):
    """Solve every timestep, warm-starting from the previous (keeps cells stable)."""
    xs, ys = _solver_grid(gw, gh)
    total = gw * gh
    n = shares.shape[1]
    i = np.arange(n) + 0.5
    r = np.sqrt(i / n); th = i * np.pi * (3 - np.sqrt(5))
    pts = np.stack([gw / 2 + r * np.cos(th) * gw * 0.42,
                    gh / 2 + r * np.sin(th) * gh * 0.42], axis=1)
    w = np.zeros(n)
    kf = []
    for j in range(shares.shape[0]):
        pts, w = _solve(shares[j], pts, w, xs, ys, total)
        kf.append((pts.copy(), w.copy()))
    return kf, xs, ys


# ── cell fill ────────────────────────────────────────────────────────────────
def _cell_geom(lab_grid, n, cw, ch):
    """Per-cell bbox + cover-fit params (video is centred & scaled to cover the
    cell's bbox, so the subject stays visible wherever the cell sits)."""
    lab = np.asarray(Image.fromarray(lab_grid.astype(np.uint8)).resize((cw, ch), Image.NEAREST))
    geoms = []
    for k in range(n):
        m = lab == k
        if not m.any():
            geoms.append(None); continue
        yk, xk = np.where(m)
        y0, y1, x0, x1 = int(yk.min()), int(yk.max()) + 1, int(xk.min()), int(xk.max()) + 1
        bh, bw = y1 - y0, x1 - x0
        s = max(bw / cw, bh / ch)
        nw, nh = max(bw, round(cw * s)), max(bh, round(ch * s))
        ox, oy = (nw - bw) // 2, (nh - bh) // 2
        geoms.append((k, y0, y1, x0, x1, bh, bw, nw, nh, ox, oy, m[y0:y1, x0:x1]))
    return lab, geoms


def _boundaries(lab):
    b = np.zeros(lab.shape, bool)
    b[:, 1:] |= lab[:, 1:] != lab[:, :-1]; b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[1:, :] |= lab[1:, :] != lab[:-1, :]; b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    return b


def _compose_color(lab_grid, colors, w, h):
    img = colors[lab_grid].astype(np.uint8)
    img[_boundaries(lab_grid)] = (16, 16, 20)
    return np.asarray(Image.fromarray(img).resize((w, h), Image.NEAREST))


def _compose_clip(lab_small, geoms, frames, fidx, w, h):
    ch, cw = lab_small.shape
    out = np.zeros((ch, cw, 3), np.uint8)
    for g in geoms:
        if g is None:
            continue
        k, y0, y1, x0, x1, bh, bw, nw, nh, ox, oy, mm = g
        vk = frames[k][fidx % len(frames[k])]
        tile = np.asarray(Image.fromarray(vk).resize((nw, nh), Image.BILINEAR))[oy:oy + bh, ox:ox + bw]
        sub = out[y0:y1, x0:x1]; sub[mm] = tile[mm]
    out[_boundaries(lab_small)] = (252, 252, 255)
    return np.asarray(Image.fromarray(out).resize((w, h), Image.BILINEAR))


# ── overlay (header, date chip, per-cell labels, NOW PLAYING) ────────────────
def _overlay_layer(lab_grid, shares, entries, date, accent, headline, eyebrow, note,
                   w, h, band_top, leader_k=None, leader_song=None, gw=270, gh=480,
                   raw=None, val_fmt=None):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    n = len(entries)
    order = np.argsort(-shares); rank = {int(k): p + 1 for p, k in enumerate(order)}
    sx, sy = w / gw, h / gh
    ys, xs = np.mgrid[0:gh, 0:gw]
    for k in range(n):
        m = lab_grid == k
        if not m.any():
            continue
        xk, yk = xs[m], ys[m]
        cx = float(np.clip(xk.mean() * sx, 96, w - 96))
        cy = float(np.clip(yk.mean() * sy, band_top + 120, h - 150))
        wpx = (xk.max() - xk.min()) * sx
        frac = shares[k]; name = entries[k][1]; lc = entries[k][2]
        fs = int(np.clip(34 + math.sqrt(frac) * 150, 26, 108))
        fn = _font(True, fs); maxw = max(wpx * 0.9, 80)
        while fs > 24 and d.textlength(name, font=fn) > maxw:
            fs -= 3; fn = _font(True, fs)
        ft, fp = _font(True, int(fs * 0.58)), _font(True, int(fs * 0.72))

        def cc(t, fo, yy, fill):
            l, tt, rr, bb = d.textbbox((0, 0), t, font=fo)
            x = cx - (rr - l) / 2 - l; y = yy - (bb - tt) / 2 - tt
            for ox2, oy2 in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)):
                d.text((x + ox2, y + oy2), t, font=fo, fill=(0, 0, 0))
            d.text((x, y), t, font=fo, fill=fill)
        cc(f"{rank[k]}위", ft, cy - fs * 0.74, (255, 255, 255))
        cc(name, fn, cy, lc)
        vtxt = val_fmt(raw[k]) if (raw is not None and val_fmt) else f"{frac * 100:.0f}%"
        cc(vtxt, fp, cy + fs * 0.76, (255, 255, 255))

    # header
    d.rectangle([0, 0, w, band_top], fill=(16, 14, 26))
    d.rectangle([0, band_top - 6, w, band_top], fill=accent)
    f1, f2 = _font(True, 60), _font(False, 30)
    for txt, fo, yy, fl in [(headline, f1, 42, (255, 255, 255)),
                            (" · ".join(x for x in (eyebrow, note) if x), f2, 132, (150, 200, 255))]:
        if not txt:
            continue
        l, t, rr, b = d.textbbox((0, 0), txt, font=fo)
        d.text((w / 2 - (rr - l) / 2 - l, yy), txt, font=fo, fill=fl)
    d.rectangle([0, 0, w - 1, h - 1], outline=accent, width=10)
    # date chip
    fo = _font(True, 44); l, t, rr, b = d.textbbox((0, 0), date, font=fo); tw = rr - l
    d.rounded_rectangle([w / 2 - tw / 2 - 30, band_top + 16, w / 2 + tw / 2 + 30, band_top + 86],
                        radius=16, fill=accent)
    d.text((w / 2 - tw / 2 - l, band_top + 51 - (b - t) / 2 - t), date, font=fo, fill=(18, 18, 22))
    # NOW PLAYING (leader BGM)
    if leader_k is not None and leader_song:
        npx = f"♪ NOW PLAYING — {entries[leader_k][1]} '{leader_song}'"
        lc = entries[leader_k][2]
        fo = _font(True, 34); l, t, rr, b = d.textbbox((0, 0), npx, font=fo); tw = rr - l
        d.rounded_rectangle([w / 2 - tw / 2 - 28, h - 92, w / 2 + tw / 2 + 28, h - 28], radius=14, fill=(18, 16, 28))
        d.text((w / 2 - tw / 2 - l, h - 60 - (b - t) / 2 - t), npx, font=fo, fill=lc)
    return np.asarray(im)


def _composite(video_rgb, layer_rgba):
    a = layer_rgba[..., 3:4].astype(np.float32) / 255.0
    return (video_rgb.astype(np.float32) * (1 - a) + layer_rgba[..., :3].astype(np.float32) * a).astype(np.uint8)


def _decode_clip(path, cw, ch, nf, fps, dst_dir):
    """Cover-fit a clip to (cw, ch) portrait, keep `nf` frames (looped)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not list(dst_dir.glob("*.png")):
        subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", "-i", str(path),
                        "-vf", f"scale={cw}:{ch}:force_original_aspect_ratio=increase,crop={cw}:{ch},fps={fps}",
                        "-frames:v", str(nf), str(dst_dir / "%03d.png")], check=True)
    fs = sorted(dst_dir.glob("*.png"))[:nf]
    return np.stack([np.asarray(Image.open(f).convert("RGB")) for f in fs])


def _card_frame(spec: dict, w: int, h: int, accent) -> np.ndarray:
    """A full-frame intro/outro title card (centred). spec keys:
    eyebrow, title, subtitle, lines[list], (seconds handled by caller)."""
    img = Image.new("RGB", (w, h), (14, 17, 25))
    d = ImageDraw.Draw(img)
    d.rectangle([5, 5, w - 6, h - 6], outline=accent, width=8)
    cx = w / 2

    def cc(t, fo, y, fill):
        if t:
            d.text((cx, y), t, font=fo, fill=fill, anchor="mm")

    cc(spec.get("eyebrow"), _font(True, 36), h * 0.28, accent)
    title = spec.get("title", "")
    fs = 96; fn = _font(True, fs)
    while fs > 40 and d.textlength(title, font=fn) > w * 0.82:
        fs -= 4; fn = _font(True, fs)
    cc(title, fn, h * 0.40, (255, 255, 255))
    d.line([cx - 130, h * 0.40 + fs * 0.72, cx + 130, h * 0.40 + fs * 0.72], fill=accent, width=6)
    cc(spec.get("subtitle"), _font(False, 42), h * 0.50, (182, 192, 208))
    y = h * 0.60
    for ln in spec.get("lines", []):
        cc(ln, _font(True, 46), y, (232, 237, 246)); y += h * 0.072
    return np.asarray(img)


# ── public API ───────────────────────────────────────────────────────────────
def build_rank_race(entries, series, labels, out, *, fill="color",
                    clips=None, audio=None, songs=None, follow_leader=False, crossfade=0.35,
                    headline="", eyebrow="", note="", accent=(255, 205, 60),
                    intro=None, outro=None,
                    hold_s=3.0, morph_s=0.6, w=1080, h=1920, fps=30,
                    grid=(270, 480), comp=(540, 960), clip_frames=60, workdir=None,
                    values=None, val_fmt=None):
    """Render a weighted-Voronoi rank race.

    entries : [(key, display_name, (r,g,b)), ...]        # order = cell index
    series  : {key: [v0, v1, ...]}                        # one value per label (auto-normalised per step)
    labels  : [str, ...]                                  # step / date chip captions
    out     : output mp4 path
    fill    : "color" (brand colour + label) | "clip" (cover-fit video per cell)
    clips   : {key: video_path}                           # required for fill="clip"
    audio   : {key: audio_path}                           # tracks for follow_leader BGM
    songs   : {key: song_title}                           # NOW PLAYING caption (defaults to display name)
    follow_leader : play the current #1's audio track, cross-fading on lead change
    intro / outro : optional {"eyebrow","title","subtitle","lines":[...],"seconds":float}
        title cards prepended / appended. With follow_leader the intro plays the
        first leader's track and the outro plays the winner's.
    hold_s / morph_s : seconds each step holds / morphs to the next
    values / val_fmt : optional absolute values per cell shown in the label
        INSTEAD of the share %. values={key: [v0, v1, ...]} (one per label,
        same shape as series); val_fmt(v)->str formats it (e.g. view counts:
        lambda v: f"{v/1e8:.1f}억"). Interpolated during morphs. Omit both to
        keep the default "NN%" label.
    Returns a dict summary.
    """
    work = pathlib.Path(workdir) if workdir else pathlib.Path(out).with_suffix("")
    work = pathlib.Path(str(work) + "_work"); work.mkdir(parents=True, exist_ok=True)
    keys = [e[0] for e in entries]
    n, gw, gh, cw, ch = len(entries), grid[0], grid[1], comp[0], comp[1]
    colors = np.array([e[2] for e in entries], dtype=np.uint8)

    shares = np.array([[float(series[k][j]) for k in keys] for j in range(len(labels))], float)
    shares /= shares.sum(axis=1, keepdims=True)
    K = shares.shape[0]
    # optional raw values per cell/step (e.g. absolute view counts) for label display
    raw_arr = None
    if values:
        raw_arr = np.array([[float(values[k][j]) for k in keys] for j in range(len(labels))], float)

    kf, xs, ys = _keyframes(shares, gw, gh)
    leader_kf = [int(np.argmax(shares[j])) for j in range(K)]

    frames = None
    if fill == "clip":
        if not clips:
            raise ValueError("fill='clip' requires clips={key: video_path}")
        frames = [_decode_clip(clips[k], cw, ch, clip_frames, fps, work / f"clip_{k}") for k in keys]

    song_name = {e[0]: (songs.get(e[0]) if songs else None) or e[1] for e in entries}  # NOW PLAYING label

    HOLD, MORPH = max(1, round(hold_s * fps)), max(1, round(morph_s * fps))
    band_top = 250

    silent = work / "silent.mp4"
    proc = subprocess.Popen([config.FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                             *config.video_args(), str(silent)], stdin=subprocess.PIPE)

    def render(lab_grid, sh, date, lk, raw=None):
        lead_song = song_name.get(keys[lk]) if follow_leader else None
        lead = lk if follow_leader else None
        lay = _overlay_layer(lab_grid, sh, entries, date, accent, headline, eyebrow, note,
                             w, h, band_top, lead, lead_song, gw, gh, raw, val_fmt)
        if fill == "clip":
            ls, geoms = _cell_geom(lab_grid, n, cw, ch)
            return ls, geoms, lay
        return None, None, lay

    def _emit_card(spec, default_s):
        secs = float(spec.get("seconds", default_s))
        nfr = max(1, round(secs * fps))
        buf = np.ascontiguousarray(_card_frame(spec, w, h, accent), dtype=np.uint8).tobytes()
        for _ in range(nfr):
            proc.stdin.write(buf)
        return nfr

    n_intro = _emit_card(intro, 2.5) if intro else 0

    blocks = []
    fi = 0
    for j in range(K):
        pA, wA = kf[j]
        labA = _assign(xs, ys, pA, wA)
        ls, geoms, lay = render(labA, shares[j], labels[j], leader_kf[j],
                                raw_arr[j] if raw_arr is not None else None)
        for _ in range(HOLD):
            vid = _compose_clip(ls, geoms, frames, fi, w, h) if fill == "clip" else _compose_color(labA, colors, w, h)
            proc.stdin.write(np.ascontiguousarray(_composite(vid, lay), dtype=np.uint8).tobytes()); fi += 1
        nblk = HOLD
        if j < K - 1:
            pB, wB = kf[j + 1]
            for f in range(MORPH):
                e = ((f + 1) / MORPH); e = e * e * (3 - 2 * e)
                p = pA * (1 - e) + pB * e; wv = wA * (1 - e) + wB * e
                sh = shares[j] * (1 - e) + shares[j + 1] * e
                rw = (raw_arr[j] * (1 - e) + raw_arr[j + 1] * e) if raw_arr is not None else None
                dt = labels[j] if e < 0.5 else labels[j + 1]
                lab = _assign(xs, ys, p, wv)
                ls, geoms, lay = render(lab, sh, dt, leader_kf[j], rw)
                vid = _compose_clip(ls, geoms, frames, fi, w, h) if fill == "clip" else _compose_color(lab, colors, w, h)
                proc.stdin.write(np.ascontiguousarray(_composite(vid, lay), dtype=np.uint8).tobytes()); fi += 1
            nblk += MORPH
        blocks.append((leader_kf[j], nblk))
    n_outro = _emit_card(outro, 3.5) if outro else 0
    proc.stdin.close(); proc.wait()
    if intro and blocks:
        blocks[0] = (blocks[0][0], blocks[0][1] + n_intro)   # intro plays first leader's track
    if outro and blocks:
        blocks[-1] = (blocks[-1][0], blocks[-1][1] + n_outro)  # outro plays winner's track
    fi += n_intro + n_outro

    # ── leader-following BGM (or single track / silent) ──────────────────────
    final = out
    have_audio = follow_leader and audio and all(audio.get(k) for k in keys)
    if have_audio:
        spans = []
        for lk, nb in blocks:
            if spans and spans[-1][0] == lk:
                spans[-1] = (lk, spans[-1][1] + nb)
            else:
                spans.append((lk, nb))
        segs = []
        for si, (lk, nb) in enumerate(spans):
            dur = nb / fps; seg = work / f"seg_{si}.m4a"
            fout = max(0.0, dur - crossfade)
            # -stream_loop -1: loop the track so a leader span longer than the
            # source audio still fills (else -t truncates and -shortest cuts video).
            subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", "-stream_loop", "-1",
                            "-i", str(audio[keys[lk]]),
                            "-t", f"{dur:.3f}", "-af",
                            f"afade=t=in:st=0:d={crossfade},afade=t=out:st={fout:.3f}:d={crossfade}",
                            "-ar", "48000", "-ac", "2", str(seg)], check=True)
            segs.append(seg)
        lst = work / "bgm.txt"; lst.write_text("".join(f"file '{s}'\n" for s in segs))
        bgm = work / "bgm_full.m4a"
        subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(lst), "-c", "copy", str(bgm)], check=True)
        subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", "-i", str(silent), "-i", str(bgm),
                        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", str(final)], check=True)
    else:
        subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", "-i", str(silent),
                        "-c", "copy", str(final)], check=True)

    return {"final": str(final), "frames": fi, "duration": round(fi / fps, 2),
            "entries": n, "steps": K, "fill": fill,
            "leaders": [entries[l][1] for l in leader_kf],
            "follow_leader": bool(have_audio)}
