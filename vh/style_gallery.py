"""Reusable video-style catalog + gallery — portable across projects.

Every project that installs vh gets a base catalog of engine-level video styles
(`vh/assets/style_catalog_base.json`). A project extends it with its OWN
`<styles_dir>/catalog.json` — augmenting a base style (by matching `id`) with a
thumbnail + example links, adding project-only styles, or hiding base ones.

    from vh.style_gallery import load_catalog, build_style_gallery

    load_catalog("proj/styles")                 # merged base+project dict (read in chat / code)
    build_style_gallery("proj/styles")          # -> proj/styles/gallery.html (self-contained)

CLI:
    python -m vh.style_gallery proj/styles [--out gallery.html] [--no-base] [--title "..."]

catalog.json (project) schema:
    {"title": "...", "updated": "YYYY-MM-DD",
     "hide": ["<base-id>", ...],
     "styles": [
       {"id": "rank-race", "thumb": "thumbs/x.png", "examples": [{"title","url"}], "variants":[...]},  # augment base
       {"id": "my-style", "title_ko","title_en","tagline","description","best_for","engine","aspect","thumb","examples"}  # new
     ]}

Thumbnails: `thumb` is relative to <styles_dir>. Missing/absent -> an auto
placeholder tile (accent + initials) is generated, so a brand-new project with
no example videos still renders a usable gallery.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import pathlib


def _load_base_catalog() -> dict:
    """Load the packaged base catalog. Uses importlib.resources so it resolves
    whether vh is installed as a plain directory or a zip (zip-safe)."""
    from importlib.resources import files
    return json.loads(
        (files("vh.assets") / "style_catalog_base.json").read_text(encoding="utf-8")
    )


# ── catalog merge ────────────────────────────────────────────────────────────
def load_catalog(styles_dir=None, include_base: bool = True) -> dict:
    """Merge the vh base catalog with a project's <styles_dir>/catalog.json.

    Project entries with a base `id` deep-overlay that base style; new ids are
    appended; `hide` drops base styles. Returns {"styles":[...], "title", "updated"}.
    """
    base = _load_base_catalog() if include_base else {"styles": []}
    proj = {"styles": []}
    if styles_dir:
        cat = pathlib.Path(styles_dir) / "catalog.json"
        if cat.exists():
            proj = json.loads(cat.read_text())

    by_id = {s["id"]: dict(s) for s in base.get("styles", [])}
    order = [s["id"] for s in base.get("styles", [])]
    for s in proj.get("styles", []):
        if s["id"] in by_id:
            by_id[s["id"]].update({k: v for k, v in s.items() if v is not None})
        else:
            by_id[s["id"]] = dict(s); order.append(s["id"])
    hide = set(proj.get("hide", []))
    styles = [by_id[i] for i in order if i not in hide]
    return {"styles": styles,
            "title": proj.get("title") or "비디오 스타일 카탈로그",
            "updated": proj.get("updated") or base.get("updated", "")}


# ── thumbnails (embedded as data URIs; placeholder when missing) ─────────────
def _placeholder_png(style: dict, w: int = 360, h: int = 480) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    from . import config
    accent = style.get("accent", "#48BEFF")
    ac = tuple(int(accent.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    img = Image.new("RGB", (w, h), (14, 17, 25))
    d = ImageDraw.Draw(img)
    for y in range(h):  # subtle vertical accent wash
        t = y / h
        d.line([(0, y), (w, y)], fill=(int(14 + ac[0] * .10 * (1 - t)),
                                       int(17 + ac[1] * .10 * (1 - t)),
                                       int(25 + ac[2] * .16 * (1 - t))))
    d.rounded_rectangle([16, 16, w - 16, h - 16], radius=18, outline=ac, width=3)
    fdir = config.CAPTION_FONTSDIR
    try:
        fb = ImageFont.truetype(f"{fdir}/NotoSansCJK-Bold.ttc", 132, index=0)
        fr = ImageFont.truetype(f"{fdir}/NotoSansCJK-Regular.ttc", 26, index=0)
    except Exception:
        fb = fr = ImageFont.load_default()
    initials = "".join(w0[0] for w0 in style.get("title_en", "?").split()[:2]).upper() or "?"
    d.text((w / 2, h / 2 - 26), initials, font=fb, fill=ac, anchor="mm")
    d.text((w / 2, h - 60), style.get("title_ko", ""), font=fr, fill=(180, 190, 205), anchor="mm")
    buf = io.BytesIO(); img.save(buf, "PNG"); return buf.getvalue()


def _datauri(style: dict, styles_dir: pathlib.Path) -> str:
    thumb = style.get("thumb")
    if thumb:
        p = styles_dir / thumb
        if p.exists():
            return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
    return "data:image/png;base64," + base64.b64encode(_placeholder_png(style)).decode()


def _esc(s) -> str:
    return html.escape(str(s or ""))


# ── gallery html ─────────────────────────────────────────────────────────────
_CSS = """
:root{--ground:#0E1119;--surface:#161B26;--surface2:#1E2634;--text:#E8EDF6;--muted:#8B94A7;
--line:#28303D;--accent:#48BEFF;--accent2:#FFCE3C;--radius:16px;--maxw:1180px;
--sans:system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans KR",sans-serif;
--mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,monospace;}
@media (prefers-color-scheme:light){:root{--ground:#F4F6FB;--surface:#FFFFFF;--surface2:#EEF2F8;
--text:#131824;--muted:#5C6474;--line:#DFE5EE;}}
:root[data-theme="dark"]{--ground:#0E1119;--surface:#161B26;--surface2:#1E2634;--text:#E8EDF6;--muted:#8B94A7;--line:#28303D;}
:root[data-theme="light"]{--ground:#F4F6FB;--surface:#FFFFFF;--surface2:#EEF2F8;--text:#131824;--muted:#5C6474;--line:#DFE5EE;}
*{box-sizing:border-box;}
body{margin:0;background:var(--ground);color:var(--text);font-family:var(--sans);line-height:1.55;-webkit-font-smoothing:antialiased;}
.wrap{max-width:var(--maxw);margin:0 auto;padding:clamp(24px,5vw,64px) clamp(18px,4vw,40px);}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.22em;text-transform:uppercase;color:var(--accent);margin:0 0 14px;}
h1{font-size:clamp(30px,6vw,52px);line-height:1.05;letter-spacing:-.02em;margin:0 0 14px;text-wrap:balance;font-weight:800;}
.lede{max-width:60ch;color:var(--muted);font-size:clamp(15px,2.2vw,18px);margin:0 0 20px;}
.howto{display:inline-flex;gap:10px;align-items:baseline;background:var(--surface);border:1px solid var(--line);
border-radius:999px;padding:9px 16px;font-size:14px;flex-wrap:wrap;}
.howto b{color:var(--accent2);}
.howto code{font-family:var(--mono);font-size:12.5px;color:var(--muted);}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:clamp(16px,2.5vw,26px);}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);overflow:hidden;
display:flex;flex-direction:column;transition:transform .25s ease,border-color .25s ease;}
.card:hover{transform:translateY(-4px);border-color:color-mix(in srgb,var(--accent) 55%,var(--line));}
.thumb{aspect-ratio:3/4;background:#05070C;overflow:hidden;}
.thumb img{width:100%;height:100%;object-fit:cover;object-position:center;display:block;transition:transform .4s ease;}
.card:hover .thumb img{transform:scale(1.04);}
.body{padding:18px 18px 20px;display:flex;flex-direction:column;gap:12px;flex:1;}
.titlerow{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;}
.titlerow h2{font-size:21px;margin:0;letter-spacing:-.01em;font-weight:750;}
.en{font-family:var(--mono);font-size:11.5px;letter-spacing:.04em;color:var(--muted);text-transform:uppercase;}
.tagline{margin:0;color:var(--accent);font-size:14.5px;font-weight:600;}
.desc{margin:0;color:var(--muted);font-size:14px;}
.variants{display:flex;gap:10px;}
.var{margin:0;flex:1;}
.var img{width:100%;aspect-ratio:9/16;object-fit:cover;border-radius:9px;border:1px solid var(--line);display:block;}
.var figcaption{font-size:11.5px;margin-top:6px;display:flex;flex-direction:column;}
.var small{color:var(--muted);font-size:10.5px;}
.chips{display:flex;flex-wrap:wrap;gap:7px;}
.chip{font-size:11.5px;background:var(--surface2);border:1px solid var(--line);border-radius:999px;padding:4px 11px;}
.meta{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.aspect{font-family:var(--mono);font-size:11px;letter-spacing:.05em;color:var(--ground);background:var(--accent2);border-radius:5px;padding:2px 8px;font-weight:700;}
.engine{font-family:var(--mono);font-size:11px;color:var(--muted);word-break:break-all;}
.examples{display:flex;flex-direction:column;gap:5px;border-top:1px solid var(--line);padding-top:12px;}
.ex{font-size:13px;color:var(--text);text-decoration:none;}
a.ex:hover{color:var(--accent);text-decoration:underline;}
.ex--demo{color:var(--muted);}
.foot{margin-top:44px;color:var(--muted);font-size:12.5px;font-family:var(--mono);}
@media (prefers-reduced-motion:reduce){*{transition:none!important;}}
"""


def _card(s: dict, styles_dir: pathlib.Path) -> str:
    chips = "".join(f'<span class="chip">{_esc(c)}</span>' for c in s.get("best_for", []))
    ex = "".join(
        (f'<a class="ex" href="{_esc(e["url"])}" target="_blank" rel="noopener">▸ {_esc(e.get("title"))}</a>'
         if e.get("url") else f'<span class="ex ex--demo">▸ {_esc(e.get("title"))}</span>')
        for e in s.get("examples", []))
    variants = ""
    if s.get("variants"):
        vt = "".join(
            f'<figure class="var"><img loading="lazy" src="{_datauri(v, styles_dir)}" alt="{_esc(v.get("name"))}">'
            f'<figcaption>{_esc(v.get("name"))}<small>{_esc(v.get("note",""))}</small></figcaption></figure>'
            for v in s["variants"])
        variants = f'<div class="variants">{vt}</div>'
    return f"""<article class="card" id="{_esc(s['id'])}">
  <div class="thumb"><img loading="lazy" src="{_datauri(s, styles_dir)}" alt="{_esc(s.get('title_ko'))} 예시"></div>
  <div class="body">
    <header class="titlerow"><h2>{_esc(s.get('title_ko'))}</h2><span class="en">{_esc(s.get('title_en'))}</span></header>
    <p class="tagline">{_esc(s.get('tagline'))}</p>
    <p class="desc">{_esc(s.get('description'))}</p>
    {variants}
    <div class="chips">{chips}</div>
    <footer class="meta"><span class="aspect">{_esc(s.get('aspect'))}</span><code class="engine">{_esc(s.get('engine'))}</code></footer>
    <div class="examples">{ex}</div>
  </div>
</article>"""


def build_style_gallery(styles_dir, out=None, title=None, include_base: bool = True) -> str:
    """Render <styles_dir>/gallery.html (self-contained) from merged base+project
    catalog. Returns the output path. `out` overrides the default location."""
    styles_dir = pathlib.Path(styles_dir)
    cat = load_catalog(styles_dir, include_base=include_base)
    ttl = title or cat["title"]
    cards = "\n".join(_card(s, styles_dir) for s in cat["styles"])
    page = f"""<title>{_esc(ttl)}</title>
<style>{_CSS}</style>
<div class="wrap">
  <header>
    <p class="eyebrow">Video Styles</p>
    <h1>{_esc(ttl)}</h1>
    <p class="lede">고를 수 있는 쇼츠·영상 포맷입니다. 마음에 드는 스타일로 “이 스타일로 만들어줘”라고 하면 그 포맷으로 제작합니다.</p>
    <span class="howto"><b>고르는 법</b><span>스타일 이름 + 소재를 알려주세요</span><code>예: “순위 레이스로 시청률 만들어줘”</code></span>
  </header>
  <main class="grid">
{cards}
  </main>
  <div class="foot">{len(cat['styles'])}개 스타일 · {_esc(cat['updated'])} · vh.style_gallery</div>
</div>
"""
    dst = pathlib.Path(out) if out else styles_dir / "gallery.html"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(page, encoding="utf-8")
    return str(dst)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a self-contained video-style gallery from a project styles/ dir.")
    ap.add_argument("styles_dir", help="project styles directory (holds catalog.json + thumbs/)")
    ap.add_argument("--out", default=None, help="output html path (default <styles_dir>/gallery.html)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-base", action="store_true", help="project catalog only, skip vh base styles")
    a = ap.parse_args(argv)
    path = build_style_gallery(a.styles_dir, out=a.out, title=a.title, include_base=not a.no_base)
    print(f"[done] {path}")


if __name__ == "__main__":
    main()
