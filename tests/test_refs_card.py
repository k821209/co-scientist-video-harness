"""Science-short reference card + description citations (vh.refs_card)."""
from vh.refs_card import build_refs_card, format_description, full_cite, short_cite

_REF = {
    "citation_key": "squadrito2017",
    "authors": ["Francesco Squadrito", "Alessandra Bitto", "Natasha Irrera",
                "Gabriele Pizzino", "Giovanni Pallio", "Letteria Minutoli",
                "Domenica Altavilla"],
    "title": "Pharmacological Activity and Clinical Use of PDRN",
    "journal": "Frontiers in Pharmacology", "journal_short": "Front Pharmacol",
    "year": 2017, "volume": "8", "issue": None, "pages": "224",
    "doi": "10.3389/fphar.2017.00224",
}


def test_short_cite_uses_surname_initials_and_abbrev():
    s = short_cite(_REF)
    assert s == "Squadrito F et al. Front Pharmacol 2017;8:224"


def test_short_cite_single_author_no_etal():
    r = {**_REF, "authors": ["Sung Tae Kim"], "journal_short": "Pharmaceutics",
         "volume": "17", "pages": "1024"}
    assert short_cite(r) == "Kim ST Pharmaceutics 2017;17:1024"


def test_full_cite_no_double_period_and_has_doi():
    fc = full_cite(_REF, 1)
    assert ".." not in fc                      # 'et al..' bug fixed
    assert fc.startswith("1. Squadrito F, Bitto A")
    assert "et al" in fc                        # >6 authors → et al
    assert "doi:10.3389/fphar.2017.00224" in fc


def test_journal_abbrev_fallback_without_journal_short():
    r = {**_REF, "journal_short": None}         # fall back to built-in _ABBR map
    assert "Front Pharmacol" in short_cite(r)


def test_format_description_numbers_entries():
    out = format_description([_REF, {**_REF, "citation_key": "x", "authors": ["A B"]}])
    lines = out.splitlines()
    assert lines[0].startswith("■")
    assert lines[1].startswith("1. ") and lines[2].startswith("2. ")


def test_build_card_renders_png(tmp_path):
    out = build_refs_card([_REF], str(tmp_path / "refs.png"))
    assert (tmp_path / "refs.png").stat().st_size > 0 and out.endswith("refs.png")
