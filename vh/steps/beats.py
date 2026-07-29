"""Beat-driven vertical Short assembler — self-drawn cards + quoted clips + photos.

Why this exists
---------------
`news.build_short` takes *only* stills and `news.build_clip_short` takes *only*
video clips, but every real episode mixes them: a quoted broadcast clip for the
cold open, a self-drawn card for the numbers, a CC photo for the person, another
clip for the payoff. Producing that meant hand-writing an `assemble.py` per
episode — ~120 lines of ffmpeg copied around, where each copy re-derived the same
handful of decisions (and re-introduced the same handful of bugs).

The shape that survived ~20 episodes and is encoded here:

* **Audio-led timing.** Each beat's VO is synthesized separately and *its* length
  sets the segment length (VO + a small pad). Never one long VO cut to fit
  pictures — the pictures follow the voice.
* **Blur-pad, never side-crop.** A 16:9 source in a 9:16 frame is width-fitted
  over a blurred, darkened copy of itself. Cropping to fill throws away ~44% of
  a wide shot, usually including whoever is talking.
* **Per-source pre-crop.** Broadcast sources carry burned-in captions and station
  banners; crop them per source (`precrop=`) before reframing. Ratios differ per
  source, so they are an argument, not a constant.
* **Credit on every quoted frame.** A quoted clip without an on-screen source
  line is not publishable; `credit` renders bottom-left for the beat's duration.
  It is pinned to the SCREEN, while a gfx card drifts under the Ken Burns zoom —
  so a card that prints its own source line (`cardkit.Card.source`) would collide
  with it mid-beat. When the card's sidecar says it does, this module skips its
  own credit overlay and says so; put the source on the card OR in `credit`, not
  both.
* **Cold-open eyecatch.** `eyecatch=True` on the first beat renders a big centred
  punch line over the shot — the 1-second hook that decides retention.

What it does NOT do: pick your in-points. Sample the source with a contact sheet
(`vh.qc.contact_sheet`) and *look* before choosing — a guessed in-point lands on
a panel reaction or a piano cutaway often enough to matter.

    from vh.steps.beats import build_beat_short
    BEATS = [
        ("b0", "clip", "…VO…", "keynote", 11.0, "caption", "credit"),
        ("b1", "gfx",  "…VO…", "g_lineup", 0.0, None, None),
    ]
    build_beat_short(BEATS, "out.mp4", workdir="wd",
                     clips={"keynote": "clips/keynote.mp4"}, gfx_dir=".",
                     outro="outro.png", bgm="bgm.wav")
"""
from __future__ import annotations

import pathlib
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, replace

from .. import config
from ..qc import lint_vo
from . import news

FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_INDEX = 1


@dataclass
class Beat:
    """One narration unit and the picture that carries it."""
    id: str
    kind: str                 # "gfx" (card png) | "clip" (video) | "photo" (still)
    text: str                 # VO script for this beat
    visual: str               # gfx: card name | clip: key in `clips` | photo: image path
    at: float = 0.0           # clip in-point (seconds)
    caption: str | None = None
    credit: str | None = None
    eyecatch: bool = False    # big centred punch text (cold open)

    @classmethod
    def coerce(cls, b) -> "Beat":
        """Accept a Beat or the 7-tuple episode files already use."""
        if isinstance(b, cls):
            return b
        bid, kind, text, visual, at, caption, credit = b
        return cls(bid, kind, text, visual, at or 0.0, caption, credit,
                   eyecatch=(bid == "b0" and bool(caption)))


def _dur(path) -> float:
    return float(subprocess.check_output(
        [config.FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)]).strip())


def _run(args):
    subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", *args], check=True)


_HASH_MAX_BYTES = 8 * 1024 * 1024      # hash anything up to 8MB (cards, photos, VO)


def _file_sig(path, *, prefer_hash: bool = True) -> str:
    """A file's identity for caching.

    Small files (cards, stills, VO — under _HASH_MAX_BYTES) are identified by
    CONTENT HASH: a card-render script usually redraws every card, so mtime
    would invalidate all segments when one card actually changed. Large files
    (video clips) fall back to size+mtime — hashing them would cost more than
    the re-encode it saves."""
    try:
        st = os.stat(path)
    except OSError:
        return f"{path}:missing"
    if prefer_hash and st.st_size <= _HASH_MAX_BYTES:
        try:
            h = hashlib.sha1(pathlib.Path(path).read_bytes()).hexdigest()
            return f"{pathlib.Path(path).name}:sha1:{h}"
        except OSError:
            pass
    return f"{pathlib.Path(path).name}:{st.st_size}:{int(st.st_mtime)}"


def _seg_fingerprint(b: "Beat", sd: float, ov_path, *, gfx, clips, precrop,
                     fps: int, canvas, encode_args, vo_path=None,
                     ref=None, window=None) -> str:
    """Everything that determines a segment's pixels — so an unchanged beat can
    be skipped on a rebuild (reuse_segments). Covers: the beat's timing (sd, from
    its VO length), the source file (card png / photo by content hash, clip by
    size+mtime), the clip in-point + precrop, the overlay image content
    (caption/credit/accent — hashed), the VO audio content, and the encode
    geometry/params. Hashing the VO means a re-recorded line of the same length
    still invalidates the segment (belt-and-braces with the VO cache key)."""
    parts = [b.kind, str(b.visual), f"at={b.at}", f"sd={sd}", f"fps={fps}",
             f"{canvas[0]}x{canvas[1]}", "enc=" + "|".join(map(str, encode_args))]
    if b.kind == "gfx":
        parts.append("src=" + _file_sig(pathlib.Path(gfx) / f"{b.visual}.png"))
    elif b.kind == "photo":
        parts.append("src=" + _file_sig(b.visual))
    else:  # clip — big file, stat-based
        parts.append("src=" + _file_sig(clips.get(b.visual, b.visual), prefer_hash=False))
        parts.append("precrop=" + precrop.get(b.visual, ""))
    if vo_path is not None:
        parts.append("vo=" + _file_sig(vo_path))
    if ref is not None:                 # reference video inside a card window
        rsrc, rat = (ref if isinstance(ref, (tuple, list)) else (ref, 0.0))
        parts.append("ref=" + _file_sig(rsrc, prefer_hash=False) + f"@{float(rat or 0.0)}")
        parts.append("win=" + ",".join(map(str, window or [])))
    try:
        parts.append("ov=" + hashlib.sha1(pathlib.Path(ov_path).read_bytes()).hexdigest())
    except OSError:
        pass
    return hashlib.sha1("\x1e".join(parts).encode("utf-8")).hexdigest()


def _overlay(beat: Beat, path, w: int, h: int, accent, font_bold, font_regular):
    """Transparent PNG: source credit (bottom-left) + caption (lower band)."""
    from PIL import Image, ImageDraw, ImageFont

    fb = lambda s: ImageFont.truetype(font_bold, s, index=FONT_INDEX)      # noqa: E731
    fr = lambda s: ImageFont.truetype(font_regular, s, index=FONT_INDEX)   # noqa: E731
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def centred(cx, y, t, font, fill):
        l, tt, r, b = d.textbbox((0, 0), t, font=font)
        d.text((cx - (r - l) / 2 - l, y), t, font=font, fill=fill)

    if beat.credit:
        f = fr(26)
        l, t, r, b = d.textbbox((0, 0), beat.credit, font=f)
        d.rounded_rectangle([28, h - 96, 28 + (r - l) + 36, h - 46], radius=8,
                            fill=(0, 0, 0, 180))
        d.text((46, h - 85), beat.credit, font=f, fill=(226, 224, 236))

    if beat.caption and beat.eyecatch:
        d.rectangle([0, int(h * 0.63), w, int(h * 0.762)], fill=(0, 0, 0, 125))
        centred(w // 2, int(h * 0.655), beat.caption, fb(78), (246, 242, 250))
        d.rounded_rectangle([w // 2 - 90, int(h * 0.735), w // 2 + 90,
                             int(h * 0.735) + 12], radius=6, fill=accent)
    elif beat.caption:
        f = fb(44)
        l, t, r, b = d.textbbox((0, 0), beat.caption, font=f)
        cw = r - l
        y = int(h * 0.764)
        d.rounded_rectangle([(w - cw) // 2 - 30, y - 14, (w + cw) // 2 + 30, y + 72],
                            radius=12, fill=(0, 0, 0, 190))
        centred(w // 2, y, beat.caption, f, (246, 242, 250))
        d.rectangle([(w - cw) // 2 - 30, y - 14, (w - cw) // 2 - 24, y + 72], fill=accent)

    im.save(path)
    return path


def build_beat_short(
    beats,
    out: str,
    *,
    workdir: str,
    clips: dict | None = None,
    gfx_dir: str | None = None,
    voice: str = "ko-KR-SunHiNeural",
    rate: str = "+5%",
    outro: str | None = None,
    outro_dur: float = 3.0,
    bgm: str | None = None,
    bgm_volume: float = 0.16,
    loudnorm: float | None = -15.0,
    precrop: dict | None = None,
    accent=(186, 148, 255, 255),
    canvas=(1080, 1920),
    fps: int = 25,
    pad_gfx: float = 0.5,
    pad_clip: float = 0.45,
    font_bold: str = FONT_BOLD,
    font_regular: str = FONT_REGULAR,
    reuse_vo: bool = True,
    reuse_segments: bool = True,
    final_encode: str = "reencode",
    ref_video: dict | None = None,
    lint_script: bool = True,
    dry_run: bool = False,
) -> dict:
    """Assemble a beat-driven Short. Returns {final, duration, vo, segments}.

    beats     Beat objects or 7-tuples (id, kind, text, visual, at, caption, credit)
    clips     {name: path} for kind="clip"; `visual` is the key
    gfx_dir   directory holding "<visual>.png" for kind="gfx" (default: workdir)
    precrop   {clip_name: "crop=iw:ih*0.90:0:0,"} — trailing comma, applied first
    bgm       looped under the VO with sidechain ducking; None = voice only
    loudnorm  target LUFS for the final mix (None to skip)
    reuse_vo  keep already-synthesized vo/<id>.mp3 (re-runs stay fast)
    reuse_segments  skip re-encoding a segment whose inputs are unchanged
              (source file, in-point, VO length, overlay, encode params — all
              fingerprinted in workdir/.seg_cache.json). A one-card edit then
              re-encodes one segment, not all N. Delete the workdir to force a
              full rebuild.
    final_encode  "reencode" (default) runs the final libx264 pass for a small
              publish-ready file; "copy" stream-copies instead (~60x faster,
              ~40% larger) for intermediate preview builds — re-encode once
              before publishing. (concat is always stream-copied.)
    ref_video  {card_name: clip_path | (clip_path, in_point)} — play a quoted
              clip INSIDE that card's window (drawn with `cardkit.Card.window`,
              coordinates read from the card's `.windows.json` sidecar). The
              window runs for the whole beat, so a quote needs no filler line
              and no length matching; the clip is fitted whole and never
              precropped (see the gfx branch). Audio is dropped — only the VO is
              heard — so keep the on-screen `credit`.
    lint_script  run qc.lint_vo on each beat's text before synthesis and print
              warnings (digits / native-counter readings / ambiguous day forms).
              Warnings only; set False to silence.
    dry_run   synthesize the VO and return the LENGTHS only — no segments, no
              concat, no encode: {dry_run, total, beats:[{bid, kind, vo, seg,
              lint}], outro, lint}. A beat's length is however long edge-tts takes
              to read it, which is hard to guess, so use this to fit the 3-minute
              ceiling by editing the SCRIPT (zero re-encodes) before building
              once. The per-beat `seg` values are also what a generated diagram's
              framerate should be matched to.

    Segment length is VO length + pad, so a clip must have at least that much
    material left after its in-point — asserted per beat rather than silently
    freezing on the last frame.
    """
    if final_encode not in ("reencode", "copy"):
        raise ValueError(f"final_encode must be 'reencode' or 'copy', got {final_encode!r}")
    W, H = canvas
    accent = tuple(accent)
    if len(accent) == 3:            # accept an RGB card-colour; PIL overlay needs RGBA
        accent = (*accent, 255)
    wd = pathlib.Path(workdir)
    (wd / "vo").mkdir(parents=True, exist_ok=True)
    gfx = pathlib.Path(gfx_dir) if gfx_dir else wd
    clips = clips or {}
    precrop = precrop or {}
    ref_video = ref_video or {}
    beats = [Beat.coerce(b) for b in beats]

    # Window rectangles come from each card's sidecar (written by Card.save), so
    # the renderer and the assembler never share coordinates in code.
    windows: dict = {}
    card_draws_source: set = set()
    for name in {*ref_video, *(b.visual for b in beats if b.kind == "gfx")}:
        sc = gfx / f"{name}.windows.json"
        if not sc.exists():
            continue
        try:
            data = json.loads(sc.read_text())
        except Exception:
            continue
        if data.pop("_reserved_bottom", None):
            # The card drew its own bottom source line (cardkit Card.source).
            card_draws_source.add(name)
        windows.update(data)
    missing = [n for n in ref_video if n not in windows]
    if missing:
        raise ValueError(
            f"ref_video names with no window: {missing}. Draw the window with "
            f"cardkit Card.window('<name>', x, y, w, h) and save the card — that "
            f"writes <card>.windows.json next to the PNG in {gfx}.")

    # Per-segment reuse cache: {beat_id: input-fingerprint} from the last build.
    cache_path = wd / ".seg_cache.json"
    prev_cache: dict = {}
    if reuse_segments and cache_path.exists():
        try:
            prev_cache = json.loads(cache_path.read_text())
        except Exception:
            prev_cache = {}
    new_cache: dict = {}
    reused = 0

    blur = (f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
            f"boxblur=22:2,eq=brightness=-0.18")

    # 1. voice-over per beat (audio leads)
    #
    # The cache key MUST include the script — keying on the file's existence
    # alone meant a re-written line kept the OLD audio: the card showed the new
    # wording while the voice read the old sentence, silently, exit 0.
    vo_cache_path = wd / "vo" / ".vo_cache.json"
    prev_vo: dict = {}
    if vo_cache_path.exists():
        try:
            prev_vo = json.loads(vo_cache_path.read_text())
        except Exception:
            prev_vo = {}
    new_vo: dict = {}
    if lint_script:
        # Read the script BEFORE spending a synthesis on it — whisper normalises
        # what it hears, so some mis-readings can never show up in a transcript.
        for b in beats:
            for w in lint_vo(b.text or ""):
                print(f"[vo-lint] {b.id} {w['kind']}: {w['match']} — {w['note']}")
    for b in beats:
        mp3 = wd / "vo" / f"{b.id}.mp3"
        key = hashlib.sha1(
            "\x1e".join([b.text or "", voice, rate]).encode("utf-8")).hexdigest()
        if reuse_vo and mp3.exists() and prev_vo.get(b.id) == key:
            new_vo[b.id] = key
            continue
        news.edge_tts_speak(b.text, str(mp3.with_suffix(".wav")), voice=voice, rate=rate)
        new_vo[b.id] = key
    vo_cache_path.write_text(json.dumps(new_vo))

    if dry_run:
        # Length-only pass: the VO is what decides every beat's length, and a
        # Short has a hard 3-minute ceiling — so answer "how long will this be?"
        # without rendering a single frame. Trimming the script from here costs
        # zero re-encodes (a full build, even final_encode="copy", still renders
        # every segment).
        beat_rows, total = [], 0.0
        for b in beats:
            vd = _dur(wd / "vo" / f"{b.id}.mp3")
            sd = round(vd + (pad_gfx if b.kind == "gfx" else pad_clip), 3)
            beat_rows.append({"bid": b.id, "kind": b.kind, "vo": round(vd, 3),
                              "seg": sd,
                              "lint": lint_vo(b.text or "") if lint_script else []})
            total += sd
        if outro:
            total += outro_dur
        return {"dry_run": True, "total": round(total, 2), "beats": beat_rows,
                "outro": outro_dur if outro else 0.0,
                "lint": [w for r in beat_rows for w in r["lint"]]}

    # 2. one video segment per beat, sized by its VO
    segs = []
    for b in beats:
        vo = wd / "vo" / f"{b.id}.mp3"
        vd = _dur(vo)
        sd = round(vd + (pad_gfx if b.kind == "gfx" else pad_clip), 3)
        nf = int(sd * fps)
        ob = b
        if b.kind == "gfx" and b.credit and b.visual in card_draws_source:
            # The card already prints its own source line, and our credit overlay
            # is pinned to the SCREEN while the card drifts under the Ken Burns
            # zoom — so the two collide part-way through the beat (visible only in
            # some frames). Let the card's own line stand.
            print(f"[beats] {b.id}: card {b.visual!r} draws its own source line, "
                  f"so the beat's credit is skipped (a fixed overlay would drift "
                  f"into it as the card zooms). Remove `credit` for this beat, or "
                  f"drop Card.source() from the card.")
            ob = replace(b, credit=None)
        ov = _overlay(ob, wd / f"ov_{b.id}.png", W, H, accent, font_bold, font_regular)
        seg = wd / f"seg_{b.id}.mp4"

        fp = _seg_fingerprint(b, sd, ov, gfx=gfx, clips=clips, precrop=precrop,
                              fps=fps, canvas=(W, H), encode_args=config.encode_args(),
                              vo_path=vo, ref=ref_video.get(b.visual),
                              window=windows.get(b.visual))
        if reuse_segments and seg.exists() and prev_cache.get(b.id) == fp:
            new_cache[b.id] = fp
            reused += 1
            segs.append((b.id, sd))
            continue

        if b.kind == "gfx":
            win = windows.get(b.visual)
            ref = ref_video.get(b.visual)
            if win and ref:
                # Reference video INSIDE the card's window. No camera move (a
                # zoompan would drag the window off its drawn frame), the clip is
                # fitted whole (decrease+pad) and NOT precropped: unlike a
                # full-screen quote, a broadcaster's burnt-in caption doesn't
                # collide with our layout here, and cropping it half-away looks
                # worse AND removes the channel logo that shows where the quote
                # came from.
                rx, ry, rw, rh = win
                rsrc, rat = (ref if isinstance(ref, (tuple, list)) else (ref, 0.0))
                rat = float(rat or 0.0)
                have = _dur(rsrc) - rat
                assert have >= sd - 0.05, (
                    f"{b.id}: reference video {pathlib.Path(str(rsrc)).name!r} has "
                    f"{have:.2f}s left after in-point {rat}s but the window must run "
                    f"{sd:.2f}s — grab a longer source (~75s) or pick an earlier "
                    f"in-point.")
                # The opposite direction used to pass silently: a source much
                # longer than the beat renders only its head, so a GENERATED
                # diagram whose payoff lands at the end never reaches the screen
                # (that shipped in several episodes). A quoted broadcast clip is
                # legitimately long, hence a warning, not an assert.
                if have > sd * 2:
                    print(f"[beats] {b.id}: reference "
                          f"{pathlib.Path(str(rsrc)).name!r} has {have:.1f}s after "
                          f"the in-point but the window only shows {sd:.1f}s "
                          f"({sd / have * 100:.0f}%) — the last {have - sd:.1f}s never "
                          f"render. Fine for a quoted clip; if this is a generated "
                          f"diagram, match its framerate/length to the beat "
                          f"(fps = frames / {sd:.1f}s).")
                vf = (f"[1:v]scale={rw}:{rh}:force_original_aspect_ratio=decrease,"
                      f"pad={rw}:{rh}:(ow-iw)/2:(oh-ih)/2:color=0x08080a,"
                      f"setsar=1,fps={fps}[win];"
                      f"[0:v][win]overlay={rx}:{ry}[base];"
                      f"[base][2:v]overlay=0:0,fps={fps}[v]")
                _run(["-loop", "1", "-i", str(gfx / f"{b.visual}.png"),
                      "-ss", str(rat), "-t", str(sd), "-i", str(rsrc), "-i", ov,
                      "-filter_complex", vf, "-map", "[v]", "-t", str(sd), "-an",
                      *config.encode_args(), "-r", str(fps), str(seg)])
            else:
                vf = (f"[0:v]scale={int(W * 1.04)}:{int(H * 1.04)},"
                      f"zoompan=z='min(zoom+0.00018,1.045)':d={nf}:"
                      f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={fps},"
                      f"trim=duration={sd}[v0];[v0][1:v]overlay=0:0,fps={fps}[v]")
                _run(["-loop", "1", "-i", str(gfx / f"{b.visual}.png"), "-i", ov,
                      "-filter_complex", vf, "-map", "[v]", "-t", str(sd), "-an",
                      *config.encode_args(), "-r", str(fps), str(seg)])

        elif b.kind == "photo":
            vf = (f"[0:v]split[a][b];[a]{blur}[bg];[b]scale={W}:-2[fg];"
                  f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base];"
                  f"[base]scale={int(W * 1.1)}:{int(H * 1.1)},"
                  f"zoompan=z='min(zoom+0.00016,1.06)':d={nf}:"
                  f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={fps},"
                  f"trim=duration={sd}[kb];[kb][1:v]overlay=0:0,fps={fps}[v]")
            _run(["-loop", "1", "-i", b.visual, "-i", ov, "-filter_complex", vf,
                  "-map", "[v]", "-t", str(sd), "-an", *config.encode_args(),
                  "-r", str(fps), str(seg)])

        else:  # clip
            src = clips[b.visual]
            have = _dur(src) - b.at
            assert have >= sd - 0.05, (
                f"{b.id}: clip '{b.visual}' has {have:.2f}s left after in-point "
                f"{b.at}s but the beat needs {sd:.2f}s — pick an earlier in-point "
                f"or shorten the narration.")
            pc = precrop.get(b.visual, "")
            vf = (f"[0:v]{pc}fps={fps},setsar=1,split[a][b];[a]{blur}[bg];"
                  f"[b]scale={W}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[m];"
                  f"[m][1:v]overlay=0:0[v]")
            _run(["-ss", str(b.at), "-t", str(sd), "-i", str(src), "-i", ov,
                  "-filter_complex", vf, "-map", "[v]", "-an",
                  *config.encode_args(), "-r", str(fps), str(seg)])

        got = _dur(seg)
        assert abs(got - sd) < 0.25, f"{b.id}: segment is {got:.2f}s, wanted {sd:.2f}s"
        new_cache[b.id] = fp
        segs.append((b.id, sd))

    if reuse_segments:
        cache_path.write_text(json.dumps(new_cache))

    # 3. silent outro card
    if outro:
        nf = int(outro_dur * fps)
        _run(["-loop", "1", "-i", outro, "-filter_complex",
              f"[0:v]scale={int(W * 1.04)}:{int(H * 1.04)},"
              f"zoompan=z='min(zoom+0.00015,1.04)':d={nf}:"
              f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={fps},"
              f"trim=duration={outro_dur},fade=t=in:st=0:d=0.4[v]",
              "-map", "[v]", "-t", str(outro_dur), "-an", *config.encode_args(),
              "-r", str(fps), str(wd / "seg_outro.mp4")])

    # 4. concat video, then build the matching VO track (pad each beat to its segment)
    with open(wd / "concat.txt", "w") as f:
        for bid, _ in segs:
            f.write(f"file 'seg_{bid}.mp4'\n")
        if outro:
            f.write("file 'seg_outro.mp4'\n")
    vtrack = wd / "vtrack.mp4"
    # Every segment came out of the same encoder with the same params/res/fps,
    # so the concat demuxer's stream-copy precondition always holds — no re-encode
    # (a whole-video generation saved, ~100x faster, zero quality loss).
    _run(["-f", "concat", "-safe", "0", "-i", str(wd / "concat.txt"),
          "-c", "copy", str(vtrack)])

    parts = []
    for bid, sd in segs:
        vo = wd / "vo" / f"{bid}.mp3"
        ap = wd / f"a_{bid}.wav"
        _run(["-i", str(vo), "-af", f"apad=pad_dur={round(sd - _dur(vo), 3)},aresample=48000",
              "-t", str(sd), "-ac", "2", "-ar", "48000", str(ap)])
        parts.append(ap)
    if outro:
        sil = wd / "a_outro.wav"
        _run(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", str(outro_dur), str(sil)])
        parts.append(sil)
    with open(wd / "aconcat.txt", "w") as f:
        for p in parts:
            f.write(f"file '{p.name}'\n")
    votrack = wd / "vo_track.wav"
    _run(["-f", "concat", "-safe", "0", "-i", str(wd / "aconcat.txt"),
          "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(votrack)])

    # 5. mix: BGM ducked under the voice, then normalise
    total = _dur(vtrack)
    norm = f",loudnorm=I={loudnorm}:TP=-1.5:LRA=11" if loudnorm is not None else ""
    if bgm:
        fo = max(0.0, total - 2)
        fc = (f"[1:a]aresample=48000,apad=whole_dur={total},asplit=2[vo1][vo2];"
              f"[2:a]volume={bgm_volume},lowpass=f=10500,afade=t=in:st=0:d=1.2,"
              f"afade=t=out:st={fo}:d=2,aresample=48000[bg];"
              f"[bg][vo1]sidechaincompress=threshold=0.03:ratio=9:attack=15:release=320[bgd];"
              f"[vo2][bgd]amix=inputs=2:duration=first:normalize=0[am];[am]anull{norm}[a]")
        _run(["-i", str(vtrack), "-i", str(votrack), "-stream_loop", "-1", "-i", bgm,
              "-filter_complex", fc, "-map", "0:v", "-map", "[a]", "-t", str(total),
              "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(wd / "muxed.mp4")])
    else:
        _run(["-i", str(vtrack), "-i", str(votrack), "-filter_complex",
              f"[1:a]aresample=48000,apad=whole_dur={total}[am];[am]anull{norm}[a]",
              "-map", "0:v", "-map", "[a]", "-t", str(total),
              "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(wd / "muxed.mp4")])

    if final_encode == "copy":
        # Preview build: stream-copy the already-muxed file (no whole-video
        # libx264 pass). ~60x faster, larger file — re-encode once before publish.
        _run(["-i", str(wd / "muxed.mp4"), "-c", "copy",
              "-movflags", "+faststart", out])
    else:
        _run(["-i", str(wd / "muxed.mp4"), "-c:v", "libx264", "-preset", "veryfast",
              "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
              "-c:a", "aac", "-b:a", "160k", out])

    return {"final": out, "duration": _dur(out), "vo": str(votrack),
            "reused_segments": reused,
            "segments": [{"id": b, "dur": d, "path": str(wd / f"seg_{b}.mp4")}
                         for b, d in segs]}
