"""Guards for the generated guide pages and sitemap.

The guides are static HTML committed to the repo (Caddy serves files, it never
runs Python). That means the committed output can silently drift from the
generator: someone edits `src/site/`, forgets to rebuild, and the
live site keeps serving the old text. These tests make that failure loud.

They also cover the SEO invariants that are easy to break by hand and
impossible to notice by eye — hreflang reciprocity, canonical/URL agreement,
title and description length budgets.
"""

from __future__ import annotations

import dataclasses
import html
import json
import re
from datetime import date
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
from src.site import assets, content, render


# --- Drift ------------------------------------------------------------------


def test_committed_output_matches_generator():
    """Every generated file on disk matches what the generator produces now.

    Failure means: run `python scripts/build_guides.py` and commit the result.
    """
    stale = []
    for path, expected in render.build().items():
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


def _landing(lang: str) -> str:
    """The landing page as the web app serves it, to visitors and crawlers."""
    from fastapi.testclient import TestClient

    import src.web.app as web_app

    return TestClient(web_app.app).get("/", params={"lang": lang}).text


def _all_guides():
    """(lang, Guide) for every guide in every language."""
    return [(lang, g) for lang, guides in content.GUIDES.items() for g in guides]


def _all_tools():
    """(lang, Tool) for every checker tool in every language."""
    return [(lang, t) for lang, tools in content.TOOLS.items() for t in tools]


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
    chrome = content.CHROME[lang]
    assert 80 <= len(chrome["landing_description"]) <= 175
    assert len(chrome["landing_og_description"]) <= 175
    assert len(chrome["landing_twitter_description"]) <= 175
    assert len(chrome["landing_title"]) <= 65
    rendered = _landing(lang)
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
        for lang, guides in content.GUIDES.items()
    }
    for lang, guides in content.GUIDES.items():
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
    rendered = render.render_guide(lang, guide)
    path = content.CHROME[lang]["guides_path"]
    expected = f'<link rel="canonical" href="{content.SITE}{path}/{guide.slug}">'
    assert expected in rendered


@pytest.mark.parametrize("lang,guide", _all_guides(), ids=lambda v: getattr(v, "slug", v))
def test_page_has_exactly_one_h1(lang, guide):
    """One <h1> per page: more than one muddies what the page is about."""
    rendered = render.render_guide(lang, guide)
    assert rendered.count("<h1>") == 1


@pytest.mark.parametrize("lang,guide", _all_guides(), ids=lambda v: getattr(v, "slug", v))
def test_section_anchors_are_unique(lang, guide):
    """Duplicate anchors would make table-of-contents links ambiguous."""
    anchors = [render._slugify_heading(s.heading) for s in guide.sections]
    assert len(anchors) == len(set(anchors)), anchors
    assert all(anchors), "a heading produced an empty anchor"


# --- "In short" box (key_points) --------------------------------------------


def test_key_points_box_renders_escaped_under_the_intro():
    """The box sits between the intro and the body, its points escaped."""
    guide = dataclasses.replace(
        content.GUIDES["fr"][0],
        key_points=["Montant < 1000 € : rien de suspect.", "Deuxième point."],
    )
    rendered = render.render_guide("fr", guide)

    assert '<aside class="key-points">' in rendered
    assert "<p>En bref</p>" in rendered
    # Escaped: the "<" is shown as text, it does not open a tag.
    assert "<li>Montant &lt; 1000 € : rien de suspect.</li>" in rendered
    assert "<li>Deuxième point.</li>" in rendered
    # Placed after the intro and before the first section heading.
    box = rendered.index('<aside class="key-points">')
    assert rendered.index('<p class="intro">') < box
    assert box < rendered.index(f'id="{render._slugify_heading(guide.sections[0].heading)}"')


@pytest.mark.parametrize("lang,guide", _all_guides(), ids=lambda v: getattr(v, "slug", v))
def test_every_guide_answers_up_front_and_in_a_faq(lang, guide):
    """Each guide carries the two blocks answer engines quote most easily.

    A new guide without them still renders, so nothing else would notice.
    """
    assert 3 <= len(guide.key_points) <= 5, guide.slug
    assert len(guide.faq) >= 2, guide.slug
    for point in guide.key_points:
        assert "<" not in point, f"key points are plain text: {point}"


@pytest.mark.parametrize("lang,guide", _all_guides(), ids=lambda v: getattr(v, "slug", v))
def test_guide_faq_structured_data_matches_visible_faq(lang, guide):
    """Same rule as the hub FAQ: the JSON-LD describes exactly what is shown."""
    page = render.render_guide(lang, guide)
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
    assert from_ld == visible == list(guide.faq)


def test_indexnow_key_file_is_published_and_routed():
    """The key the script sends is the one the site serves, at the path Caddy routes.

    A mismatch makes every IndexNow ping fail with a 403, silently from the
    site's point of view.
    """
    from scripts import indexnow

    key_file = ROOT / "public" / f"{indexnow.KEY}.txt"
    assert key_file.read_text(encoding="utf-8").strip() == indexnow.KEY
    caddyfile = (ROOT / "Caddyfile").read_text(encoding="utf-8")
    assert f"handle /{indexnow.KEY}.txt {{" in caddyfile
    assert indexnow.build_payload(["u"])["keyLocation"] == f"{content.SITE}/{indexnow.KEY}.txt"


def test_no_key_points_no_box():
    """A guide without key points renders no empty box."""
    guide = dataclasses.replace(content.GUIDES["en"][0], key_points=[])
    # The class name also lives in the inlined stylesheet: look for the tag.
    assert '<aside class="key-points">' not in render.render_guide("en", guide)


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
    rendered = render.render_tool(lang, tool)
    path = content.CHROME[lang]["tools_path"]
    assert f'<link rel="canonical" href="{content.SITE}{path}/{tool.slug}">' in rendered
    assert rendered.count("<h1>") == 1
    assert assets.CHECKSUM_JS.rstrip("\n") in rendered
    assert f'"kind": "{tool.kind}"' in rendered


def test_tool_i18n_json_cannot_close_its_own_script_tag(monkeypatch):
    """A translation containing </script> must not break out of the JSON block.

    render_tool() escapes "</" as "<\/" inside the inline JSON (json.dumps
    does not). Seed a hostile value and check the *rendered page*: the block
    still ends at the real closing tag and parses back to the same string.
    """
    hostile = "</script><img src=x onerror=alert(1)>"
    i18n = dict(content.CHROME["fr"]["tool_i18n"], reminder=hostile)
    monkeypatch.setitem(content.CHROME["fr"], "tool_i18n", i18n)
    rendered = render.render_tool("fr", content.TOOLS["fr"][0])
    block = re.search(r'<script type="application/json" id="tf-i18n">(.*?)</script>', rendered, re.S).group(1)
    assert "</script" not in block
    assert json.loads(block)["reminder"] == hostile


def test_tool_counterparts_are_reciprocal():
    by_lang = {lang: {t.slug: t for t in tools} for lang, tools in content.TOOLS.items()}
    for lang, tools in content.TOOLS.items():
        other = "en" if lang == "fr" else "fr"
        for tool in tools:
            assert tool.counterpart in by_lang[other], (lang, tool.slug, tool.counterpart)
            assert by_lang[other][tool.counterpart].counterpart == tool.slug
            assert by_lang[other][tool.counterpart].kind == tool.kind


def test_tool_messages_exist_in_both_languages():
    fr, en = content.CHROME["fr"]["tool_i18n"], content.CHROME["en"]["tool_i18n"]
    assert set(fr) == set(en), set(fr) ^ set(en)
    for kind in ("iban", "siret", "siren", "vat", "nir"):
        for prefix in ("format_", "valid_", "invalid_"):
            assert prefix + kind in fr
    assert {tool.kind for tools in content.TOOLS.values() for tool in tools} == {"iban", "siret", "siren", "vat", "nir"}


@pytest.mark.parametrize("lang", ["fr", "en"])
def test_guides_index_links_every_tool(lang):
    page = render.render_index(lang)
    for tool in content.TOOLS[lang]:
        assert f'href="{content.CHROME[lang]["tools_path"]}/{tool.slug}"' in page


# --- Sitemap ----------------------------------------------------------------

SITEMAP_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "xhtml": "http://www.w3.org/1999/xhtml",
}


def test_sitemap_is_well_formed_and_complete():
    """Every guide URL appears in the sitemap exactly once."""
    root = ET.fromstring((ROOT / "public" / "sitemap.xml").read_text(encoding="utf-8"))
    locs = [el.text for el in root.findall("sm:url/sm:loc", SITEMAP_NS)]
    assert len(locs) == len(set(locs)), "duplicate <loc> in sitemap"

    for lang, guide in _all_guides():
        url = f"{content.SITE}{content.CHROME[lang]['guides_path']}/{guide.slug}"
        assert url in locs, f"{url} missing from sitemap.xml"
    for lang, tool in _all_tools():
        url = f"{content.SITE}{content.CHROME[lang]['tools_path']}/{tool.slug}"
        assert url in locs, f"{url} missing from sitemap.xml"


def test_sitemap_urls_all_resolve_to_a_committed_file():
    """No sitemap entry points at a page that does not exist.

    A sitemap listing URLs that 404 is a crawl-budget leak and shows up as an
    error in Search Console. The two landing pages are rendered by the web
    app rather than read from a file, so they are skipped.
    """
    root = ET.fromstring((ROOT / "public" / "sitemap.xml").read_text(encoding="utf-8"))
    app_pages = {content.SITE + "/", content.SITE + "/en"}

    for el in root.findall("sm:url/sm:loc", SITEMAP_NS):
        url = el.text
        if url in app_pages:
            continue
        path = url.removeprefix(content.SITE)
        if path.startswith("/en/guides"):
            rel = path.removeprefix("/en/guides").strip("/")
            base = ROOT / "public" / "guides" / "en"
        elif path.startswith("/en/tools"):
            rel = path.removeprefix("/en/tools").strip("/")
            base = ROOT / "public" / "tools" / "en"
        elif path.startswith("/outils"):
            rel = path.removeprefix("/outils").strip("/")
            base = ROOT / "public" / "tools" / "fr"
        else:
            rel = path.removeprefix("/guides").strip("/")
            base = ROOT / "public" / "guides" / "fr"
        target = base / "index.html" if not rel else base / f"{rel}.html"
        assert target.exists(), f"{url} has no file at {target.relative_to(ROOT)}"


def test_sitemap_hreflang_annotations_are_self_referential():
    """Each URL's alternates include the URL itself, as the spec requires."""
    root = ET.fromstring((ROOT / "public" / "sitemap.xml").read_text(encoding="utf-8"))
    for url_el in root.findall("sm:url", SITEMAP_NS):
        loc = url_el.find("sm:loc", SITEMAP_NS).text
        alternates = {
            link.get("href")
            for link in url_el.findall("xhtml:link", SITEMAP_NS)
        }
        assert loc in alternates, f"{loc} does not list itself as an alternate"


# --- Cross-file consistency -------------------------------------------------


def test_landing_pages_link_to_the_guides():
    """The landing pages link into /guides.

    Without an internal link the guides are reachable only through the sitemap,
    which crawls far more slowly and passes no link signal.
    """
    fr, en = _landing("fr"), _landing("en")
    assert 'href="/guides/' in fr
    assert 'href="/en/guides/' in en
    # Same for the checker tools: a page only the hub links to is one hop
    # further from the home page than it needs to be.
    for tool in content.TOOLS["fr"]:
        assert f'href="/outils/{tool.slug}"' in fr
    for tool in content.TOOLS["en"]:
        assert f'href="/en/tools/{tool.slug}"' in en


@pytest.mark.parametrize("lang", ["fr", "en"])
def test_faq_structured_data_matches_visible_faq(lang):
    """FAQPage JSON-LD must describe exactly the <dt>/<dd> pairs on the page.

    Structured data for content that is not visible is a Google spam-policy
    violation in its own right; the two are generated from one list today,
    but nothing else would notice if a refactor let them diverge.
    """
    page = render.render_index(lang)
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
    robots = (ROOT / "public" / "robots.txt").read_text(encoding="utf-8")
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
    rendered = render.render_guide(lang, guide)
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
    tree = ET.parse(ROOT / "public" / "sitemap.xml")
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    lastmod = {
        url.find("sm:loc", ns).text: url.find("sm:lastmod", ns).text
        for url in tree.getroot().findall("sm:url", ns)
    }
    for lang in ("fr", "en"):
        chrome = content.CHROME[lang]
        for guide in content.GUIDES[lang]:
            loc = f"{content.SITE}{chrome['guides_path']}/{guide.slug}"
            assert lastmod[loc] == guide.modified, loc
        for tool in content.TOOLS[lang]:
            loc = f"{content.SITE}{chrome['tools_path']}/{tool.slug}"
            assert lastmod[loc] == tool.modified, loc
    # Hubs and landings list everything, so they moved when the newest page did.
    newest = render.latest_content_date()
    for loc in (f"{content.SITE}/", f"{content.SITE}/en"):
        assert lastmod[loc] == newest
    for value in lastmod.values():
        date.fromisoformat(value)


@pytest.mark.parametrize("lang,guide", _all_guides(), ids=lambda v: getattr(v, "slug", v))
def test_guide_tool_callouts_point_at_real_tools(lang, guide):
    """A guide may offer checker tools; each slug must exist and be linked."""
    known = {tool.slug for tool in content.TOOLS[lang]}
    # Check the slugs before rendering: render_guide() would raise KeyError
    # on an unknown one, which is a worse failure message than this assert.
    for slug in guide.tools:
        assert slug in known, f"{guide.slug} offers unknown tool {slug!r}"
    assert len(guide.tools) == len(set(guide.tools))
    rendered = render.render_guide(lang, guide)
    path = content.CHROME[lang]["tools_path"]
    for slug in guide.tools:
        assert f'href="{path}/{slug}"' in rendered


def test_llms_txt_lists_every_public_page():
    """llms.txt (llmstxt.org) names every guide and tool, and only public URLs.

    Committed-output parity with the generator is covered by
    test_committed_output_matches_generator, since build() emits it.
    """
    text = render.render_llms_txt()
    assert text.startswith("# TrustyFile\n\n> ")
    linked = set(re.findall(r"\]\((https://[^)]+)\)", text))
    for lang in ("fr", "en"):
        chrome = content.CHROME[lang]
        assert f"{content.SITE}{chrome['index_path']}" in linked
        for guide in content.GUIDES[lang]:
            assert f"{content.SITE}{chrome['guides_path']}/{guide.slug}" in linked
        for tool in content.TOOLS[lang]:
            assert f"{content.SITE}{chrome['tools_path']}/{tool.slug}" in linked
    sitemap_locs = {entry.loc for entry in render.sitemap_entries()}
    # Two public URLs are linked on purpose without a sitemap entry: the
    # sitemap itself, and the API docs — a Swagger UI app served to everyone
    # under /api/*, worth a pointer for assistants but not an indexable page.
    assert linked - sitemap_locs == {
        f"{content.SITE}/sitemap.xml",
        f"{content.SITE}/api/docs",
    }


_LINK_ROOTS = {
    "/guides/": ROOT / "public" / "guides" / "fr",
    "/en/guides/": ROOT / "public" / "guides" / "en",
    "/outils/": ROOT / "public" / "tools" / "fr",
    "/en/tools/": ROOT / "public" / "tools" / "en",
}


def test_internal_links_resolve_to_generated_files():
    """Every site-relative link in every generated page points at a real page.

    Guides cross-link each other by hand-typed slug inside prose; a typo
    there renders fine and 404s in production, which nothing else catches.
    """
    broken = []
    for path, page in render.build().items():
        if path.suffix != ".html":
            continue
        for href in re.findall(r'href="(/[^"#]*)', page):
            for prefix, root in _LINK_ROOTS.items():
                if href.startswith(prefix):
                    slug = href[len(prefix):]
                    target = root / "index.html" if slug == "" else root / f"{slug}.html"
                    if not target.exists():
                        broken.append((path.relative_to(ROOT).as_posix(), href))
                    break
            else:
                # Not a generated section: an app route (the landing pages, the
                # API docs the method page links to) or a static asset.
                assert href in ("/", "/en", "/api/docs") or "." in href.rsplit("/", 1)[-1], href
    assert not broken, broken


@pytest.mark.parametrize("lang", ["fr", "en"])
def test_hub_groups_every_guide_under_exactly_one_audience(lang):
    for guide in content.GUIDES[lang]:
        assert guide.audience in content.AUDIENCES, (guide.slug, guide.audience)
    page = render.render_index(lang)
    headings = content.CHROME[lang]["audience_headings"]
    for audience in content.AUDIENCES:
        assert f'<h2 id="{audience}">{html.escape(headings[audience])}</h2>' in page
    for guide in content.GUIDES[lang]:
        assert page.count(f'href="{content.CHROME[lang]["guides_path"]}/{guide.slug}"') == 1
    assert len(content.CHROME[lang]["index_description"]) <= 175


@pytest.mark.parametrize("lang,tool", _all_tools(), ids=lambda v: getattr(v, "slug", v))
def test_tool_related_guides_exist_and_are_linked(lang, tool):
    known = {guide.slug for guide in content.GUIDES[lang]}
    assert tool.guides, f"{tool.slug} offers no guide"
    for slug in tool.guides:
        assert slug in known, (tool.slug, slug)
    rendered = render.render_tool(lang, tool)
    for slug in tool.guides:
        assert f'href="{content.CHROME[lang]["guides_path"]}/{slug}"' in rendered


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
    assert render._format_date(lang, iso) == expected
