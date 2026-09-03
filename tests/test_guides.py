"""Guards for the generated guide pages and sitemap.

The guides are static HTML committed to the repo (Caddy serves files, it never
runs Python). That means the committed output can silently drift from the
generator: someone edits `scripts/build_guides.py`, forgets to rebuild, and the
live site keeps serving the old text. These tests make that failure loud.

They also cover the SEO invariants that are easy to break by hand and
impossible to notice by eye — hreflang reciprocity, canonical/URL agreement,
title and description length budgets.
"""

from __future__ import annotations

import html
import json
import re
import sys
from datetime import date
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_guides  # noqa: E402  (needs the sys.path tweak above)


# --- Drift ------------------------------------------------------------------


def test_committed_output_matches_generator():
    """Every generated file on disk matches what the generator produces now.

    Failure means: run `python scripts/build_guides.py` and commit the result.
    """
    stale = []
    for path, expected in build_guides.build().items():
        if not path.exists():
            stale.append(f"{path.relative_to(ROOT)} (missing)")
        elif path.read_text(encoding="utf-8") != expected:
            stale.append(str(path.relative_to(ROOT)))
    assert not stale, (
        "Generated files are out of date: "
        + ", ".join(stale)
        + ". Run: python scripts/build_guides.py"
    )


# --- Structural SEO invariants ----------------------------------------------


def _all_guides():
    """(lang, Guide) for every guide in every language."""
    return [(lang, g) for lang, guides in build_guides.GUIDES.items() for g in guides]


def _all_tools():
    """(lang, Tool) for every checker tool in every language."""
    return [(lang, t) for lang, tools in build_guides.TOOLS.items() for t in tools]


@pytest.mark.parametrize("lang,guide", _all_guides(), ids=lambda v: getattr(v, "slug", v))
def test_title_and_description_fit_serp_budget(lang, guide):
    """Titles and descriptions stay inside what Google actually displays.

    Beyond roughly 60 / 160 characters the text is truncated with an ellipsis,
    so the tail is wasted. These are display budgets, not hard limits, hence
    the generous ceilings.
    """
    assert len(guide.title) <= 65, f"title too long ({len(guide.title)}): {guide.title}"
    assert 80 <= len(guide.description) <= 175, (
        f"description length {len(guide.description)} outside 80-175: {guide.slug}"
    )


@pytest.mark.parametrize("lang", ["fr", "en"])
def test_landing_descriptions_fit_their_budgets_and_reach_the_head(lang):
    """The crawler landing pages obey the same display budgets as the guides.

    They are not Guide objects, so the parametrized budget test above skips
    them — which is how the French meta description reached 211 characters
    unnoticed. The OG / Twitter variants exist because link previews truncate
    harder than a results page; check they are the ones actually emitted.
    """
    chrome = build_guides.CHROME[lang]
    assert 80 <= len(chrome["landing_description"]) <= 175
    assert len(chrome["landing_og_description"]) <= 175
    assert len(chrome["landing_twitter_description"]) <= 175
    assert len(chrome["landing_title"]) <= 65
    rendered = build_guides.render_landing(lang)
    esc = html.escape
    assert f'<meta name="description" content="{esc(chrome["landing_description"])}">' in rendered
    assert f'<meta property="og:description" content="{esc(chrome["landing_og_description"])}">' in rendered
    assert f'<meta name="twitter:description" content="{esc(chrome["landing_twitter_description"])}">' in rendered


@pytest.mark.parametrize("lang,guide", _all_guides(), ids=lambda v: getattr(v, "slug", v))
def test_slug_is_url_safe(lang, guide):
    """Slugs are lowercase ASCII with hyphens — no accents, no spaces.

    Accented URLs get percent-encoded, which is legal but unreadable in a
    result listing and easy to mangle when copied.
    """
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", guide.slug), guide.slug


def test_counterparts_are_reciprocal():
    """If FR guide A points at EN guide B, B must point back at A.

    Google discards a whole hreflang cluster when the annotations disagree, so
    a one-way link is worse than none.
    """
    by_lang = {
        lang: {g.slug: g for g in guides}
        for lang, guides in build_guides.GUIDES.items()
    }
    for lang, guides in build_guides.GUIDES.items():
        other = "en" if lang == "fr" else "fr"
        for guide in guides:
            if guide.counterpart is None:
                continue
            assert guide.counterpart in by_lang[other], (
                f"{lang}/{guide.slug} points at unknown {other} slug "
                f"{guide.counterpart!r}"
            )
            back = by_lang[other][guide.counterpart].counterpart
            assert back == guide.slug, (
                f"{lang}/{guide.slug} <-> {other}/{guide.counterpart} not reciprocal"
            )


@pytest.mark.parametrize("lang,guide", _all_guides(), ids=lambda v: getattr(v, "slug", v))
def test_page_declares_its_own_canonical(lang, guide):
    """The canonical URL matches where the page is actually served."""
    rendered = build_guides.render_guide(lang, guide)
    path = build_guides.CHROME[lang]["guides_path"]
    expected = f'<link rel="canonical" href="{build_guides.SITE}{path}/{guide.slug}">'
    assert expected in rendered


@pytest.mark.parametrize("lang,guide", _all_guides(), ids=lambda v: getattr(v, "slug", v))
def test_page_has_exactly_one_h1(lang, guide):
    """One <h1> per page: more than one muddies what the page is about."""
    rendered = build_guides.render_guide(lang, guide)
    assert rendered.count("<h1>") == 1


@pytest.mark.parametrize("lang,guide", _all_guides(), ids=lambda v: getattr(v, "slug", v))
def test_section_anchors_are_unique(lang, guide):
    """Duplicate anchors would make table-of-contents links ambiguous."""
    anchors = [build_guides._slugify_heading(s.heading) for s in guide.sections]
    assert len(anchors) == len(set(anchors)), anchors
    assert all(anchors), "a heading produced an empty anchor"


# --- Checker tools ----------------------------------------------------------


@pytest.mark.parametrize("lang,tool", _all_tools(), ids=lambda v: getattr(v, "slug", v))
def test_tool_title_description_slug_fit_the_same_budgets(lang, tool):
    assert len(tool.title) <= 65, f"title too long ({len(tool.title)}): {tool.title}"
    assert 80 <= len(tool.description) <= 175, (
        f"description length {len(tool.description)} outside 80-175: {tool.slug}"
    )
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", tool.slug), tool.slug


@pytest.mark.parametrize("lang,tool", _all_tools(), ids=lambda v: getattr(v, "slug", v))
def test_tool_page_canonical_h1_and_inline_script(lang, tool):
    """Canonical matches the served path; one h1; the shared validator is inlined verbatim.

    The inline copy and tools/checksums.js are the same string by
    construction, but the Node test only exercises the file — so pin the
    pages to it, or a future edit could ship a browser validator nobody tests.
    """
    rendered = build_guides.render_tool(lang, tool)
    path = build_guides.CHROME[lang]["tools_path"]
    assert f'<link rel="canonical" href="{build_guides.SITE}{path}/{tool.slug}">' in rendered
    assert rendered.count("<h1>") == 1
    assert build_guides.CHECKSUM_JS.rstrip("\n") in rendered
    assert f'"kind": "{tool.kind}"' in rendered
    assert "</script>" not in json.dumps(build_guides.CHROME[lang]["tool_i18n"])


def test_tool_counterparts_are_reciprocal():
    by_lang = {lang: {t.slug: t for t in tools} for lang, tools in build_guides.TOOLS.items()}
    for lang, tools in build_guides.TOOLS.items():
        other = "en" if lang == "fr" else "fr"
        for tool in tools:
            assert tool.counterpart in by_lang[other], (lang, tool.slug, tool.counterpart)
            assert by_lang[other][tool.counterpart].counterpart == tool.slug
            assert by_lang[other][tool.counterpart].kind == tool.kind


def test_tool_messages_exist_in_both_languages():
    fr, en = build_guides.CHROME["fr"]["tool_i18n"], build_guides.CHROME["en"]["tool_i18n"]
    assert set(fr) == set(en), set(fr) ^ set(en)
    for kind in ("iban", "siret", "siren"):
        for prefix in ("format_", "valid_", "invalid_"):
            assert prefix + kind in fr


@pytest.mark.parametrize("lang", ["fr", "en"])
def test_guides_index_links_every_tool(lang):
    page = build_guides.render_index(lang)
    for tool in build_guides.TOOLS[lang]:
        assert f'href="{build_guides.CHROME[lang]["tools_path"]}/{tool.slug}"' in page


# --- Sitemap ----------------------------------------------------------------

SITEMAP_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "xhtml": "http://www.w3.org/1999/xhtml",
}


def test_sitemap_is_well_formed_and_complete():
    """Every guide URL appears in the sitemap exactly once."""
    root = ET.fromstring((ROOT / "sitemap.xml").read_text(encoding="utf-8"))
    locs = [el.text for el in root.findall("sm:url/sm:loc", SITEMAP_NS)]
    assert len(locs) == len(set(locs)), "duplicate <loc> in sitemap"

    for lang, guide in _all_guides():
        url = f"{build_guides.SITE}{build_guides.CHROME[lang]['guides_path']}/{guide.slug}"
        assert url in locs, f"{url} missing from sitemap.xml"
    for lang, tool in _all_tools():
        url = f"{build_guides.SITE}{build_guides.CHROME[lang]['tools_path']}/{tool.slug}"
        assert url in locs, f"{url} missing from sitemap.xml"


def test_sitemap_urls_all_resolve_to_a_committed_file():
    """No sitemap entry points at a page that does not exist.

    A sitemap listing URLs that 404 is a crawl-budget leak and shows up as an
    error in Search Console. The two app landing pages are served by Streamlit
    rather than from a file, so they are skipped.
    """
    root = ET.fromstring((ROOT / "sitemap.xml").read_text(encoding="utf-8"))
    app_pages = {build_guides.SITE + "/", build_guides.SITE + "/en"}

    for el in root.findall("sm:url/sm:loc", SITEMAP_NS):
        url = el.text
        if url in app_pages:
            continue
        path = url.removeprefix(build_guides.SITE)
        if path.startswith("/en/guides"):
            rel = path.removeprefix("/en/guides").strip("/")
            base = ROOT / "guides" / "en"
        elif path.startswith("/en/tools"):
            rel = path.removeprefix("/en/tools").strip("/")
            base = ROOT / "tools" / "en"
        elif path.startswith("/outils"):
            rel = path.removeprefix("/outils").strip("/")
            base = ROOT / "tools" / "fr"
        else:
            rel = path.removeprefix("/guides").strip("/")
            base = ROOT / "guides" / "fr"
        target = base / "index.html" if not rel else base / f"{rel}.html"
        assert target.exists(), f"{url} has no file at {target.relative_to(ROOT)}"


def test_sitemap_hreflang_annotations_are_self_referential():
    """Each URL's alternates include the URL itself, as the spec requires."""
    root = ET.fromstring((ROOT / "sitemap.xml").read_text(encoding="utf-8"))
    for url_el in root.findall("sm:url", SITEMAP_NS):
        loc = url_el.find("sm:loc", SITEMAP_NS).text
        alternates = {
            link.get("href")
            for link in url_el.findall("xhtml:link", SITEMAP_NS)
        }
        assert loc in alternates, f"{loc} does not list itself as an alternate"


# --- Cross-file consistency -------------------------------------------------


def test_seo_landing_pages_link_to_the_guides():
    """The crawler-facing pages link into /guides.

    Without an internal link the guides are reachable only through the sitemap,
    which crawls far more slowly and passes no link signal.
    """
    fr = (ROOT / "seo-fr.html").read_text(encoding="utf-8")
    en = (ROOT / "seo-en.html").read_text(encoding="utf-8")
    assert 'href="/guides/' in fr
    assert 'href="/en/guides/' in en
    # Same for the checker tools: a page only the hub links to is one hop
    # further from the home page than it needs to be.
    for tool in build_guides.TOOLS["fr"]:
        assert f'href="/outils/{tool.slug}"' in fr
    for tool in build_guides.TOOLS["en"]:
        assert f'href="/en/tools/{tool.slug}"' in en


class _VisibleText(HTMLParser):
    """Collect the text nodes a reader sees in <body>, skipping script/style."""

    def __init__(self) -> None:
        super().__init__()
        self.in_body = False
        self.skipping = 0
        self.texts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "body":
            self.in_body = True
        if tag in ("script", "style"):
            self.skipping += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skipping -= 1

    def handle_data(self, data):
        if self.in_body and not self.skipping:
            # Separator glyphs and whitespace carry no content of their own.
            text = data.strip(" \n\t·")
            if text:
                self.texts.append(text)


@pytest.mark.parametrize("lang", ["fr", "en"])
def test_landing_page_says_nothing_the_app_does_not(lang):
    """The crawler landing page is built from the app's own strings — and only them.

    Every visible text node must be a translation value app.py renders on its
    landing (headline, tagline, disclaimers, steps, footer), a guide title, or
    fixed chrome (brand, step numbers, punctuation, contact address, language
    switch). Anything else is prose a human never sees: the cloaking bug this
    generator exists to prevent (fixed 2026-08-27).
    """
    t = build_guides._translations(lang)
    colon = " :" if lang == "fr" else ":"
    allowed = {
        t[key]
        for key in (
            "subtitle", "landing_headline", "landing_tagline", "disclaimer_upload",
            "step1_title", "step1_desc", "step2_title", "step2_desc",
            "step3_title", "step3_desc", "disclaimer_api",
            "footer_guides", "footer_tools", "footer_contact",
        )
    }
    allowed |= {t["footer_guides"] + colon, t["footer_tools"] + colon}
    allowed |= {guide.title for guide in build_guides.GUIDES[lang]}
    allowed |= {tool.title for tool in build_guides.TOOLS[lang]}
    allowed |= {
        "TrustyFile", "1", "2", "3", "contact@trustyfile.com",
        build_guides.CHROME[lang]["lang_switch"],
    }

    parser = _VisibleText()
    parser.feed(build_guides.render_landing(lang))
    stray = [text for text in parser.texts if text not in allowed]
    assert not stray, f"crawler page carries text the app landing does not: {stray}"


@pytest.mark.parametrize("lang", ["fr", "en"])
def test_faq_structured_data_matches_visible_faq(lang):
    """FAQPage JSON-LD must describe exactly the <dt>/<dd> pairs on the page.

    Structured data for content that is not visible is a Google spam-policy
    violation in its own right; the two are generated from one list today,
    but nothing else would notice if a refactor let them diverge.
    """
    page = build_guides.render_index(lang)
    blocks = [
        json.loads(block)
        for block in re.findall(r'<script type="application/ld\+json">(.*?)</script>', page, re.S)
    ]
    faq = next(block for block in blocks if block.get("@type") == "FAQPage")
    from_ld = [(q["name"], q["acceptedAnswer"]["text"]) for q in faq["mainEntity"]]
    visible = [
        (html.unescape(q), html.unescape(a))
        for q, a in re.findall(r"<dt>(.*?)</dt>\s*<dd>(.*?)</dd>", page, re.S)
    ]
    assert len(visible) >= 3
    assert from_ld == visible


def test_robots_does_not_block_the_guides():
    """A Disallow rule covering /guides would undo the whole exercise."""
    robots = (ROOT / "robots.txt").read_text(encoding="utf-8")
    disallowed = [
        line.split(":", 1)[1].strip()
        for line in robots.splitlines()
        if line.lower().startswith("disallow:")
    ]
    for rule in disallowed:
        for public in ("/guides", "/en/guides", "/outils", "/en/tools", "/llms.txt"):
            assert rule and not public.startswith(rule.rstrip("*")), (
                f"robots.txt rule {rule!r} blocks {public}"
            )


# --- Dates, tool callouts, llms.txt -----------------------------------------


@pytest.mark.parametrize("lang,guide", _all_guides(), ids=lambda v: getattr(v, "slug", v))
def test_guide_dates_are_real_and_rendered(lang, guide):
    """published <= modified, both ISO, and the page shows the same date it declares.

    The visible "updated" line, the Article JSON-LD and the sitemap must agree:
    a dateModified that says one thing while the page says another is the
    inconsistency Google treats as an unreliable freshness signal.
    """
    published = date.fromisoformat(guide.published)
    modified = date.fromisoformat(guide.modified)
    assert published <= modified <= date.today()
    rendered = build_guides.render_guide(lang, guide)
    assert f'<time datetime="{guide.modified}">' in rendered
    assert rendered.count("<time ") == 1
    article = next(
        json.loads(block)
        for block in re.findall(r'<script type="application/ld\+json">(.*?)</script>', rendered, re.S)
        if json.loads(block).get("@type") == "Article"
    )
    assert article["datePublished"] == guide.published
    assert article["dateModified"] == guide.modified


def test_sitemap_lastmod_comes_from_each_page():
    """Every guide and tool URL carries its own modified date, not a global one."""
    tree = ET.parse(ROOT / "sitemap.xml")
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    lastmod = {
        url.find("sm:loc", ns).text: url.find("sm:lastmod", ns).text
        for url in tree.getroot().findall("sm:url", ns)
    }
    for lang in ("fr", "en"):
        chrome = build_guides.CHROME[lang]
        for guide in build_guides.GUIDES[lang]:
            loc = f"{build_guides.SITE}{chrome['guides_path']}/{guide.slug}"
            assert lastmod[loc] == guide.modified, loc
        for tool in build_guides.TOOLS[lang]:
            loc = f"{build_guides.SITE}{chrome['tools_path']}/{tool.slug}"
            assert lastmod[loc] == tool.modified, loc
    # Hubs and landings list everything, so they moved when the newest page did.
    newest = build_guides.latest_content_date()
    for loc in (f"{build_guides.SITE}/", f"{build_guides.SITE}/en"):
        assert lastmod[loc] == newest
    for value in lastmod.values():
        date.fromisoformat(value)


@pytest.mark.parametrize("lang,guide", _all_guides(), ids=lambda v: getattr(v, "slug", v))
def test_guide_tool_callouts_point_at_real_tools(lang, guide):
    """A guide may offer checker tools; each slug must exist and be linked."""
    known = {tool.slug for tool in build_guides.TOOLS[lang]}
    rendered = build_guides.render_guide(lang, guide)
    path = build_guides.CHROME[lang]["tools_path"]
    for slug in guide.tools:
        assert slug in known, f"{guide.slug} offers unknown tool {slug!r}"
        assert f'href="{path}/{slug}"' in rendered
    assert len(guide.tools) == len(set(guide.tools))


def test_llms_txt_lists_every_public_page():
    """llms.txt (llmstxt.org) names every guide and tool, and only public URLs.

    Committed-output parity with the generator is covered by
    test_committed_output_matches_generator, since build() emits it.
    """
    text = build_guides.render_llms_txt()
    assert text.startswith("# TrustyFile\n\n> ")
    linked = set(re.findall(r"\]\((https://[^)]+)\)", text))
    for lang in ("fr", "en"):
        chrome = build_guides.CHROME[lang]
        assert f"{build_guides.SITE}{chrome['index_path']}" in linked
        for guide in build_guides.GUIDES[lang]:
            assert f"{build_guides.SITE}{chrome['guides_path']}/{guide.slug}" in linked
        for tool in build_guides.TOOLS[lang]:
            assert f"{build_guides.SITE}{chrome['tools_path']}/{tool.slug}" in linked
    sitemap_locs = {entry.loc for entry in build_guides.sitemap_entries()}
    assert linked - sitemap_locs == {f"{build_guides.SITE}/sitemap.xml"}


_LINK_ROOTS = {
    "/guides/": ROOT / "guides" / "fr",
    "/en/guides/": ROOT / "guides" / "en",
    "/outils/": ROOT / "tools" / "fr",
    "/en/tools/": ROOT / "tools" / "en",
}


def test_internal_links_resolve_to_generated_files():
    """Every site-relative link in every generated page points at a real page.

    Guides cross-link each other by hand-typed slug inside prose; a typo
    there renders fine and 404s in production, which nothing else catches.
    """
    broken = []
    for path, content in build_guides.build().items():
        if path.suffix != ".html":
            continue
        for href in re.findall(r'href="(/[^"#]*)', content):
            for prefix, root in _LINK_ROOTS.items():
                if href.startswith(prefix):
                    slug = href[len(prefix):]
                    target = root / "index.html" if slug == "" else root / f"{slug}.html"
                    if not target.exists():
                        broken.append((path.relative_to(ROOT).as_posix(), href))
                    break
            else:
                # Not a generated section: the app root, /en, or a static asset.
                assert href in ("/", "/en") or "." in href.rsplit("/", 1)[-1], href
    assert not broken, broken


@pytest.mark.parametrize(
    "lang,iso,expected",
    [
        ("fr", "2026-08-13", "13 août 2026"),
        ("fr", "2026-09-01", "1er septembre 2026"),
        ("en", "2026-08-13", "13 August 2026"),
        ("en", "2026-09-01", "1 September 2026"),
    ],
)
def test_format_date(lang, iso, expected):
    assert build_guides._format_date(lang, iso) == expected
