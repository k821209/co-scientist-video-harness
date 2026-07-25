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

    # ── escape hatch + output ────────────────────────────────────────────
    @property
    def draw(self) -> ImageDraw.ImageDraw:
        """Raw ImageDraw for shapes (panels, chips). Shapes are not guarded."""
        return self.d

    def save(self, path) -> str:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.im.save(p)
        return str(p)
