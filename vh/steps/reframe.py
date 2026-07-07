"""Aspect-ratio reframing for Shorts (9:16) and long-form (16:9).

Aspect-aware: the source width/aspect is inspected FIRST. A source that already
matches the target aspect is filled edge-to-edge (no blurred bars) and is never
upscaled past its native resolution — so an already-vertical clip stays crisp
instead of being blown up to 1080p with pointless side bars.
"""
from __future__ import annotations

from pathlib import Path

from .. import config
from ..probe import run, probe

# How close (relative) src aspect must be to target to count as "already correct".
ASPECT_TOL = 0.06


def _even(n: int) -> int:
    return n - (n % 2)


def reframe(src: str, dst: str, preset) -> str:
    """Fit source into target WxH, considering the source aspect first.

    mode="pad"  -> aspect mismatch: blurred background fills the bars (Shorts look)
                   aspect match:    fill exactly, no bars, no upscale
    mode="crop" -> scale to cover, center-crop (fills frame, loses edges)
    mode="none" -> letterbox scale to target
    """
    w, h, mode = preset.target_w, preset.target_h, preset.reframe_mode
    allow_upscale = getattr(preset, "allow_upscale", False)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)

    info = probe(src)
    src_ar = info.width / info.height
    tgt_ar = w / h
    rel = abs(src_ar - tgt_ar) / tgt_ar
    matches = rel <= ASPECT_TOL

    if mode == "pad" and matches:
        # Already the right shape. Fill exactly (crop the sub-pixel excess),
        # and don't upscale a smaller source — keep it crisp at native size.
        if info.width <= w and not allow_upscale:
            vf = f"scale={_even(info.width)}:{_even(info.height)}"
            note = f"native {_even(info.width)}x{_even(info.height)} (aspect match, no upscale)"
        else:
            vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
            note = f"{w}x{h} (aspect match, fill)"
    elif mode == "crop":
        vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
        note = f"{w}x{h} (cover-crop)"
    elif mode == "pad":
        # Genuine aspect mismatch (e.g. landscape -> vertical): blurred bars.
        vf = (
            f"split[a][b];"
            f"[a]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},boxblur=luma_radius=40:luma_power=2[bg];"
            f"[b]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )
        note = f"{w}x{h} (blur-pad, aspect diff {rel*100:.0f}%)"
    else:  # none / letterbox-safe scale
        vf = (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
        )
        note = f"{w}x{h} (letterbox)"

    print(f"[reframe] src {info.width}x{info.height} -> {note}")
    from .. import remote
    remote.ffmpeg_run([
        config.FFMPEG, "-y", "-i", str(src),
        "-vf", vf,
        *config.encode_args(),
        str(dst),
    ], reads=[str(src)], write=str(dst))
    return dst
