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
                 h: int | None = None, *, check_hierarchy: bool = True,
                 hierarchy_ratio: float = 1.6):
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
        self.check_hierarchy = check_hierarchy
        self.hierarchy_ratio = hierarchy_ratio
        self._kickers: list[tuple] = []
        self._headlines: list[tuple] = []

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

    # ── type hierarchy: kicker + headline as a NAMED pair ────────────────
    #
    # The third failure mode, after collisions (rule) and margins (ink): the card
    # renders cleanly, nothing overlaps, and it still can't be read — because it
    # has an eyebrow and a table but no big line saying what the card is about.
    # A hand-written `text(..., fb(76))` headline can be forgotten silently; a
    # named pair makes "this card has no headline" visible in the code, and lets
    # save() warn about it.

    def kicker(self, y: float, text: str, *, size: int = 34, color=None,
               x: float | None = None, align: str = "c", font=None):
        """Small eyebrow label ("기상청 53년 분석"). Pair it with headline()."""
        f = font or self.fr(size)
        box = self.text((self.W / 2 if x is None else x, y), text,
                        f, color or self.DIM, align)
        self._kickers.append(box)
        return box

    def headline(self, y: float, text: str, *, size: int = 72, color=None,
                 x: float | None = None, align: str = "c", bold: bool = True,
                 max_width: float | None = None, rule_gap: int | None = None):
        """The card's own sentence, set big ("여름은 실제로 길어졌다").

        Ask of every card: **is the biggest text on it what the card is trying to
        say?** If not, the headline is missing. `max_width` auto-shrinks to fit;
        `rule_gap` also draws the guarded divider that far below it."""
        f = (self.fit_font(text, max_width, size, bold=bold) if max_width
             else (self.fb(size) if bold else self.fr(size)))
        box = self.text((self.W / 2 if x is None else x, y), text,
                        f, color or self.FG, align)
        self._headlines.append(box)
        if rule_gap is not None:
            self.rule(int(box[3]) + rule_gap, int(box[0]), int(box[2]))
        return box

    def row(self, y: float, left: str, right: str | None = None, *,
            sub: str | None = None, x: float = 148, gap: int = 26,
            size: int = 40, right_size: int | None = None, sub_size: int = 30,
            color=None, right_color=None, sub_color=None):
        """One list row: `left` + `right` set ADJACENT (not pushed to opposite
        edges) with `sub` left-aligned underneath.

        Splitting a row's parts to the far margins leaves the middle empty and the
        row reads as three scattered fragments instead of one item — pinning them
        into a left-anchored group fixed exactly that (and cut the row height
        250 → 186). Returns the row's bottom y."""
        fl = self.fb(size)
        lb = self.text((x, y), left, fl, color or self.FG, "l")
        bottom = lb[3]
        if right:
            fr_ = self.fr(right_size or size - 6)
            rb = self.text((lb[2] + gap, y), right, fr_, right_color or self.DIM, "l")
            bottom = max(bottom, rb[3])
        if sub:
            sb = self.text((x, bottom + 14), sub, self.fr(sub_size),
                           sub_color or self.DIM, "l", vtop=True)
            bottom = sb[3]
        return bottom

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

    def _warn_if_no_headline(self, name: str) -> None:
        """Soft check: a card with a kicker but nothing much bigger than it
        probably lost its headline.

        Deliberately a WARNING, never an exception — plenty of good cards let a
        big number ("71만 5천 명") or a single big word ("북한") BE the headline,
        and those pass this test naturally. Disable with
        `Card(..., check_hierarchy=False)`; tune with `hierarchy_ratio`."""
        if not (self.check_hierarchy and self._kickers) or self._headlines:
            return
        height = lambda b: b[3] - b[1]           # noqa: E731
        kick = max(height(b) for b in self._kickers)
        biggest = max((height(b) for b in self.boxes), default=0.0)
        if biggest < kick * self.hierarchy_ratio:
            print(f"[cardkit] {name}: no headline? the biggest text is "
                  f"{biggest:.0f}px vs a {kick:.0f}px kicker "
                  f"(< {self.hierarchy_ratio}x). A kicker + a table with no big "
                  f"line reads as an untitled table — add headline(), or pass "
                  f"check_hierarchy=False if a big number IS the headline.")

    def save(self, path) -> str:
        """Write the PNG. If the card has windows, also write the
        `<stem>.windows.json` sidecar the assembler reads — the renderer and the
        assembler share coordinates through this file, never through code, so a
        card can be re-laid-out without touching the build."""
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._warn_if_no_headline(p.name)
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
