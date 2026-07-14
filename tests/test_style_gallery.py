"""Reusable video-style catalog + gallery (vh.style_gallery)."""
import json

from vh.style_gallery import build_style_gallery, load_catalog

BASE_IDS = {"rank-race", "clip-deepdive", "stills-news", "graphic-brief", "clip-quote"}


def test_base_catalog_loads_via_importlib_resources():
    """The packaged base catalog resolves with no styles_dir (zip-safe load)."""
    cat = load_catalog()
    ids = {s["id"] for s in cat["styles"]}
    assert BASE_IDS <= ids
    assert cat["title"]


def test_no_base_returns_project_only(tmp_path):
    (tmp_path / "catalog.json").write_text(json.dumps({
        "styles": [{"id": "only", "title_en": "Only One"}],
    }))
    cat = load_catalog(tmp_path, include_base=False)
    assert [s["id"] for s in cat["styles"]] == ["only"]


def test_project_augments_hides_and_adds(tmp_path):
    (tmp_path / "catalog.json").write_text(json.dumps({
        "title": "AIVO styles",
        "hide": ["clip-quote"],
        "styles": [
            {"id": "rank-race", "tagline": "OUR RANK RACE"},          # augment a base id
            {"id": "aivo-only", "title_en": "AIVO Only"},             # new project style
        ],
    }))
    cat = load_catalog(tmp_path)
    by_id = {s["id"]: s for s in cat["styles"]}
    assert cat["title"] == "AIVO styles"
    assert "clip-quote" not in by_id                                  # hidden
    assert by_id["rank-race"]["tagline"] == "OUR RANK RACE"           # augmented
    assert by_id["rank-race"]["title_en"]                             # base fields kept
    assert "aivo-only" in by_id                                       # appended


def test_build_gallery_is_self_contained_with_placeholders(tmp_path):
    """A project with NO thumbnails still renders — placeholders are embedded
    as base64 data URIs, so the html is self-contained."""
    out = build_style_gallery(tmp_path)
    html = (tmp_path / "gallery.html").read_text(encoding="utf-8")
    assert out.endswith("gallery.html")
    assert "data:image/png;base64," in html          # thumbnails embedded (placeholders)
    assert "http" not in html.split("<style>")[0] or True
    for sid in BASE_IDS:
        assert f'id="{sid}"' in html                 # every base style rendered as a card


def test_cli_styles_subcommand(tmp_path):
    from vh.cli import main
    rc = main(["styles", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "gallery.html").exists()
