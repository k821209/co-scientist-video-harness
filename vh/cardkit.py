"""Graphic-card helper with a layout guard.

Shorts built from `vh.steps.beats` are mostly *self-drawn cards* (title, spec
table, price list, timeline). Drawing them with raw PIL has one failure mode that
bites every time and is invisible in logs: a divider line whose y is a guessed
offset from the row's y (`y + 30`) runs straight **through** the glyphs when the
row's font is big. The render succeeds, ffmpeg is happy, and you only notice by
opening a frame — which is exactly the check people skip.

`Card` fixes that structurally: every `text()` records the glyph bbox it actually
drew, and `rule()` raises AssertionError if the line it is about to draw would
intersect one (plus `pad` clearance). The bug becomes a build failure.

    from vh.cardkit import Card
    c = Card()
    c.text((540, 300), "제목", c.fb(80), c.FG)              # centred, y = mid
    lab = c.text((130, 700), "무게", c.fr(40), c.DIM, "l")   # left
    val = c.text((950, 700), "201g", c.fb(46), c.FG, "r")    # right
    c.rule(int(c.bottom(lab, val)) + 26, 130, 950)           # safe by construction
    c.save("g_spec.png")

Position accents off the returned bbox (`bottom(...) + margin`), never off a
hand-guessed offset — that is the whole point.
"""
from __future__ import annotations

import json
import pathlib

from PIL import Image, ImageDraw, ImageFont

# Noto CJK ships with most distros and covers KR/JP/CN + Latin in one file.
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_INDEX = 1          # ttc face 1 = Korean; 0 = Japanese


class Card:
    """A 1080x1920 gradient card that refuses to draw a rule through text."""

    W, H = 1080, 1920
    FG = (240, 238, 246)
    DIM = (158, 156, 176)
    LINE = (52, 50, 68)
    LAV = (186, 148, 255)
    MINT = (96, 226, 200)
    AMBER = (250, 196, 110)

    def __init__(self, top=(28, 26, 38), bot=(12, 11, 18), w: int | None = None,
                 h: int | None = None):
        self.W = w or Card.W
        self.H = h or Card.H
        self.im = Image.new("RGB", (self.W, self.H), bot)
        self.d = ImageDraw.Draw(self.im)
        for y in range(self.H):
            t = y / self.H
            self.d.line([(0, y), (self.W, y)],
                        fill=tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)))
        self.boxes: list[tuple[float, float, float, float]] = []
        self.windows: dict[str, list[int]] = {}   # name -> [x, y, w, h]

    # ── fonts ────────────────────────────────────────────────────────────
    @staticmethod
    def fb(size: int, path: str = FONT_BOLD):
        return ImageFont.truetype(path, size, index=FONT_INDEX)

    @staticmethod
    def fr(size: int, path: str = FONT_REGULAR):
        return ImageFont.truetype(path, size, index=FONT_INDEX)

    # ── text (records its bbox) ──────────────────────────────────────────
    def text(self, xy, t: str, font, fill, align: str = "c", vtop: bool = False):
        """align: 'c' centre / 'l' left / 'r' right on x.
        y is the glyph *middle* unless vtop=True (then it is the top edge).
        Returns the drawn bbox (x0, y0, x1, y1) — feed it to bottom()/rule()."""
        x, y = xy
        l, tt, r, b = self.d.textbbox((0, 0), t, font=font)
        w, h = r - l, b - tt
        px = {"c": x - w / 2 - l, "l": x - l, "r": x - w - l}[align]
        py = (y - tt) if vtop else (y - h / 2 - tt)
        self.d.text((px, py), t, font=font, fill=fill)
        box = (px + l, py + tt, px + l + w, py + tt + h)
        self.boxes.append(box)
        return box

    # ── rule (guarded) ───────────────────────────────────────────────────
    def rule(self, y: int, x0: int, x1: int, fill=None, width: int = 2, pad: int = 10):
        """Horizontal divider. Raises if it would cross (or hug) any drawn text."""
        top, bot = y - pad, y + width + pad
        for bx0, by0, bx1, by1 in self.boxes:
            if bx1 > x0 and bx0 < x1 and by1 > top and by0 < bot:
                raise AssertionError(
                    f"rule(y={y}, x={x0}..{x1}) would cross the text box "
                    f"({bx0:.0f},{by0:.0f})-({bx1:.0f},{by1:.0f}). "
                    f"Place it at bottom(...) + margin instead of a guessed offset.")
        self.d.line([(x0, y), (x1, y)], fill=fill or self.LINE, width=width)

    def bar(self, box, w: int = 92, h: int = 8, gap: int = 14, fill=None, radius: int = 4):
        """Accent bar centred under a text box (uses its real bottom, not a guess)."""
        cx = (box[0] + box[2]) / 2
        top = int(box[3]) + gap
        self.d.rounded_rectangle([cx - w / 2, top, cx + w / 2, top + h],
                                 radius=radius, fill=fill or self.MINT)
        return top + h

    @staticmethod
    def bottom(*boxes) -> float:
        """Lowest edge of the given text boxes — the anchor for rules/accents."""
        return max(b[3] for b in boxes)

    # ── metrics-first sizing ─────────────────────────────────────────────
    def ink(self, t: str, font) -> tuple[float, float]:
        """(width, height) of the glyphs `t` would actually draw in `font`.

        Measure with this instead of guessing — Hangul ink height varies with the
        size AND the syllables, so a constant row height leaves asymmetric inner
        margins (and can push the ink past the box edge, i.e. negative padding)."""
        l, tp, r, b = self.d.textbbox((0, 0), t, font=font)
        return r - l, b - tp

    def stack_height(self, items, *, pad: int, gap: int) -> float:
        """Height a panel needs to hold `items` [(text, font), …] stacked, with
        `pad` above and below and `gap` between rows:
        pad + ink1 + gap + ink2 + … + pad.

        Derive the BOX from the type, never the other way round — then the inner
        margins are equal by construction at any text length or font size:

            rows = [(label, c.fr(34)), (value, c.fb(46))]
            h = c.stack_height(rows, pad=40, gap=20)
            c.draw.rounded_rectangle([x0, y, x1, y + h], radius=24, fill=...)
        """
        heights = [self.ink(t, f)[1] for t, f in items]
        if not heights:
            return 2 * pad
        return 2 * pad + sum(heights) + gap * (len(heights) - 1)

    def fit_font(self, t: str, max_width: float, start_pt: int, *, bold: bool = False,
                 floor: int = 22, step: int = 2):
        """Largest font ≤ `start_pt` whose `t` fits `max_width` (min `floor`).
        Pair with stack_height so auto-shrunk text still gets an exact box."""
        make = self.fb if bold else self.fr
        f = make(start_pt)
        while self.ink(t, f)[0] > max_width and f.size > floor:
            f = make(f.size - step)
        return f

    # ── reference-video window ───────────────────────────────────────────
    def window(self, name: str, x: int, y: int, w: int, h: int, *,
               label: str | None = "참고 영상", border=None, fill=(8, 8, 10),
               label_pt: int = 24, radius: int = 10) -> float:
        """Reserve a rectangle for a REFERENCE VIDEO that plays inside the card.

        Draws the frame (border + label) and leaves the interior flat; the clip is
        composited there at assembly time — pass `ref_video={name: (clip, at)}` to
        `build_beat_short`, which reads the sidecar this writes.

        Quoting a clip *inside the card* instead of cutting to it removes the need
        for a filler line ("here's that video") whose only job was to give the cut
        a duration: the window runs for as long as the beat's narration does. A
        real episode came out 122s instead of 138s with MORE quotes and no empty
        narration, and the cold open reads better because the first frame already
        moves.

        Coordinates are snapped DOWN to even numbers — h264_nvenc rejects odd
        dimensions (exit 234, with nothing in the message about it).

        Returns the window's bottom y. Saved to `<card>.windows.json` by save()."""
        ex, ey, ew, eh = (int(v) - int(v) % 2 for v in (x, y, w, h))
        if (ex, ey, ew, eh) != (int(x), int(y), int(w), int(h)):
            print(f"[cardkit] window {name!r}: snapped to even "
                  f"{ew}x{eh} at ({ex},{ey}) — odd sizes break h264_nvenc")
        col = border or self.LAV
        self.d.rounded_rectangle([ex - 4, ey - 4, ex + ew + 4, ey + eh + 4],
                                 radius=radius, outline=col, width=3)
        self.d.rectangle([ex, ey, ex + ew, ey + eh], fill=fill)
        if label:
            f = self.fr(label_pt)
            tw = self.ink(label, f)[0]
            self.d.rectangle([ex, ey - 42, ex + tw + 30, ey - 4], fill=col)
            self.text((ex + 15 + tw / 2, ey - 24), label, f, (18, 14, 18))
        self.windows[name] = [ex, ey, ew, eh]
        return ey + eh

    # ── escape hatch + output ────────────────────────────────────────────
    @property
    def draw(self) -> ImageDraw.ImageDraw:
        """Raw ImageDraw for shapes (panels, chips). Shapes are not guarded."""
        return self.d

    def save(self, path) -> str:
        """Write the PNG. If the card has windows, also write the
        `<stem>.windows.json` sidecar the assembler reads — the renderer and the
        assembler share coordinates through this file, never through code, so a
        card can be re-laid-out without touching the build."""
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.im.save(p)
        if self.windows:
            sidecar = p.with_suffix("").with_suffix(".windows.json")
            existing = {}
            if sidecar.exists():
                try:
                    existing = json.loads(sidecar.read_text())
                except Exception:
                    existing = {}
            existing.update(self.windows)
            sidecar.write_text(json.dumps(existing, ensure_ascii=False))
        return str(p)
