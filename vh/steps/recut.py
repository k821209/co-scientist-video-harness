"""Speech-paced re-cut helpers for dubbing across big duration gaps.

When a dub's target language runs much shorter than the source (e.g. KO->EN is
~60%), padding the slack as silence gives dead air and slowing the voice sounds
draggy. Instead, CUT the video to the voice: each sentence shows its footage for
its dubbed-audio length, jump-cutting the slack — but NOT where the screen is
actively changing (a scroll, a click-through, a file opening), which the viewer
still needs to see.

Two primitives:
  screen_activity(src)  -> per-frame visual-change signal (frame diff)
  paced_cut(...)        -> the clip end for one segment: at least long enough to
                           cover the voice, extended to hold a changing screen
                           (bounded), otherwise cut at the voice end.

The caller pairs each segment's clip [start, paced_cut(...)] with audio
[voice + silence(clip_len - voice_len)] so clip length == audio length per
segment; concatenated (with cards spliced via titlecards.build_with_interstitials
`bounds=`), video and dub stay locked with zero drift.
"""
from __future__ import annotations

import subprocess

from .. import config


def screen_activity(src: str, fps: float = 2.0, w: int = 96, h: int = 54):
    """Mean absolute frame-to-frame pixel difference at `fps`, as a numpy array
    indexed by frame (t = i / fps). High where the screen is changing, ~0 when
    static. Cheap: decodes to tiny grayscale."""
    import numpy as np
    cmd = [config.FFMPEG, "-nostdin", "-loglevel", "error", "-i", str(src),
           "-vf", f"fps={fps},scale={w}:{h},format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (w * h)
    frames = np.frombuffer(raw, dtype=np.uint8)[:n * w * h].reshape(n, w * h).astype(np.int16)
    diff = np.abs(np.diff(frames, axis=0)).mean(axis=1)
    return np.concatenate([[0.0], diff]), fps


def paced_cut(start: float, vo_dur: float, slot_end: float, activity, afps: float,
              src_dur: float, pause: float = 0.35, act_thr: float = 4.0,
              keep_tail: float = 0.8, max_keep: float = 7.0) -> float:
    """End time of the clip that starts at `start` and carries a `vo_dur`-second
    dubbed voice. Always covers the voice (+ `pause`); if the screen keeps
    changing past that within [., slot_end], hold it up to `max_keep` longer so a
    scroll/click isn't chopped; else cut at voice end."""
    import numpy as np
    vo_end = start + vo_dur
    base = min(src_dur, vo_end + pause)          # never shorter than the voice
    end = min(src_dur, slot_end)
    i0, i1 = int(base * afps), int(end * afps)
    if i1 > i0:
        hot = np.where(activity[i0:i1] > act_thr)[0]
        if hot.size:
            last = base + (hot[-1] + 1) / afps + keep_tail
            return min(end, base + max_keep, max(base, last))
    return base
