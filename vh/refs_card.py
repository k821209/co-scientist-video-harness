"""Science-short references → (1) on-screen "참고문헌" card PNG, (2) YouTube
description citation block.

Takes the reference dicts the co-scientist reference store returns
(verify_doi / add_reference_by_doi / list_references) verbatim — no hand-typed
citations, only DOI-verified records:

    refs = [ {citation_key, title, authors[], journal, journal_short?,
              year, volume, issue, pages, doi}, ... ]
    build_refs_card(refs, "gfx/g_refs.png")     # on-screen card (deep-dive theme)
    print(format_description(refs))              # description block (full DOIs)

The card shows compact citations (author · journal-abbrev · year;vol:pages, font
auto-shrinks); the description carries the full bibliography + DOI. Journal is
abbreviated from the ref's `journal_short` (CrossRef short-container-title) when
present, else a small built-in map, else the full name.
"""
from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw, ImageFont

from . import config

# Fallback journal abbreviations (used only when a ref has no `journal_short`).
_ABBR = {
    "Frontiers in Pharmacology": "Front Pharmacol",
    "Annual Review of Immunology": "Annu Rev Immunol",
    "Current Issues in Molecular Biology": "Curr Issues Mol Biol",
    "Journal of the American Academy of Dermatology": "J Am Acad Dermatol",
    "New England Journal of Medicine": "N Engl J Med",
}

_FONTDIR = config.CAPTION_FONTSDIR
_FB = f"{_FONTDIR}/NotoSansCJK-Bold.ttc"
_FR = f"{_FONTDIR}/NotoSansCJK-Regular.ttc"


def _fmt_author(name) -> str:
    """'Francesco Squadrito' -> 'Squadrito F' (surname assumed last)."""
    parts = str(name or "").split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    surname = parts[-1]
    initials = "".join(p[0] for p in parts[:-1] if p)
    return f"{surname} {initials}".strip()


def _journal(ref) -> str:
    """Abbreviated journal: ref's own short title → built-in map → full name."""
    return (ref.get("journal_short")
            or _ABBR.get(ref.get("journal", ""), ref.get("journal", "")) or "")


def _volpage(ref) -> str:
    v, p = ref.get("volume"), ref.get("pages")
    if v and p:
        return f";{v}:{p}"
    return f";{v}" if v else ""


def _tidy(s: str) -> str:
    """Collapse accidental double punctuation ('et al..' → 'et al.')."""
    while ".." in s:
        s = s.replace("..", ".")
    return s.replace(" .", ".").strip()


def short_cite(ref) -> str:
    """Card line: 'Squadrito F et al. Front Pharmacol 2017;8:224'."""
    au = ref.get("authors") or []
    a = _fmt_author(au[0]) if au else ""
    if len(au) > 1:
        a += " et al."
    return _tidy(f"{a} {_journal(ref)} {ref.get('year','')}{_volpage(ref)}".strip())


def full_cite(ref, idx=None) -> str:
    """Description line: up to 6 authors (Surname Initials) + et al., DOI."""
    au = ref.get("authors") or []
    names = ", ".join(_fmt_author(a) for a in au[:6]) + (", et al" if len(au) > 6 else "")
    vip = ""
    if ref.get("volume"):
        vip = f";{ref['volume']}"
        if ref.get("issue"):
            vip += f"({ref['issue']})"
        if ref.get("pages"):
            vip += f":{ref['pages']}"
    doi = f" doi:{ref['doi']}" if ref.get("doi") else ""
    pref = f"{idx}. " if idx else ""
    journal = ref.get("journal") or _journal(ref)
    return _tidy(f"{pref}{names}. {ref.get('title','')}. {journal}. {ref.get('year','')}{vip}.") + doi


def format_description(refs, header="■ 참고문헌 (DOI 검증 완료)") -> str:
    return "\n".join([header] + [full_cite(r, i) for i, r in enumerate(refs, 1)])


def build_refs_card(refs, out_png, *, title="참고문헌 (주요)",
                    accent=(255, 138, 115), teal=(70, 212, 200),
                    W=1620, H=1584, bg=(14, 22, 30), fg=(238, 244, 248),
                    dim=(140, 156, 168), max_items=6,
                    note="전체 DOI·PMID는 설명란(더보기)에", disclaimer="의학 정보 · 의료 조언 아님"):
    """Deep-dive-theme reference card PNG. Returns the output path."""
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    def f(p, s): return ImageFont.truetype(p, s, index=0)

    def C(xy, t, fo, fill):
        l, tt, r, b = d.textbbox((0, 0), t, font=fo)
        d.text((xy[0] - (r - l) / 2 - l, xy[1] - (b - tt) / 2 - tt), t, font=fo, fill=fill)

    C((W // 2, 190), title, f(_FB, 78), fg)
    y = 400
    for r in refs[:max_items]:
        s = short_cite(r)
        fs = 46; fo = f(_FR, fs)
        while fs > 30 and d.textlength(s, font=fo) > W - 260:
            fs -= 2; fo = f(_FR, fs)
        d.ellipse([160, y - 9, 178, y + 9], fill=accent)
        l, tt, r2, b = d.textbbox((0, 0), s, font=fo)
        d.text((210, y - (b - tt) / 2 - tt), s, font=fo, fill=fg)
        y += 135
    C((W // 2, 1180), note, f(_FR, 46), teal)
    if disclaimer:
        C((W // 2, 1300), disclaimer, f(_FR, 42), dim)
    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png
