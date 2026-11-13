"""
Render the static pages from the site's content.

Guides (/guides, /en/guides), the crawler landing pages (seo-*.html), the
checker tools (/outils, /en/tools), sitemap.xml and llms.txt. `build()`
returns every generated file as {path: content} without writing anything;
`scripts/build_guides.py` is the command that writes them.

The reasons this is generated at all (anti-cloaking, one copy of the head
boilerplate) are explained in `scripts/build_guides.py`.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

from src.site.assets import CHECKSUM_JS, STYLESHEET, TOOL_GLUE_JS
from src.site.content import (
    AUDIENCES,
    CHROME,
    GUIDES,
    OUTPUT_DIR,
    ROOT,
    SITE,
    TOOLS,
    Guide,
    Tool,
)

_MONTHS = {
    "fr": [
        "janvier", "février", "mars", "avril", "mai", "juin", "juillet",
        "août", "septembre", "octobre", "novembre", "décembre",
    ],
    "en": [
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December",
    ],
}


def _format_date(lang: str, iso_date: str) -> str:
    """Spell out an ISO date the way a reader of that language expects.

    French writes the day first with no leading zero, and "1er" for the
    first; English here follows the same day-month-year order, which is what
    the previous hand-written labels used.

    Example:
        >>> _format_date("fr", "2026-09-01")
        '1er septembre 2026'
        >>> _format_date("en", "2026-08-13")
        '13 August 2026'
    """
    year, month, day = (int(part) for part in iso_date.split("-"))
    day_text = "1er" if lang == "fr" and day == 1 else str(day)
    return f"{day_text} {_MONTHS[lang][month - 1]} {year}"


def _updated_line(lang: str, iso_date: str) -> str:
    """The visible "updated on" line, with a machine-readable <time>.

    Example:
        >>> _updated_line("fr", "2026-08-13")
        '<p class="updated"><time datetime="2026-08-13">Mis à jour le 13 août 2026</time></p>'
    """
    label = CHROME[lang]["updated"].format(date=_format_date(lang, iso_date))
    return f'<p class="updated"><time datetime="{iso_date}">{label}</time></p>'


def _guide_cards(lang: str, guides: list["Guide"]) -> list[str]:
    """Link cards for guides — shared by the hub and the tool pages."""
    chrome = CHROME[lang]
    parts: list[str] = []
    for guide in guides:
        parts.append(f'    <a class="card" href="{chrome["guides_path"]}/{guide.slug}">')
        parts.append(f"        <h3>{html.escape(guide.title)}</h3>")
        parts.append(f"        <p>{html.escape(guide.description)}</p>")
        parts.append(f'        <p class="more">{chrome["read_more"]} &rarr;</p>')
        parts.append("    </a>")
    return parts


def _tool_cards(lang: str, tools: list["Tool"]) -> list[str]:
    """Link cards for checker tools — shared by the hub, the guides and the tools."""
    chrome = CHROME[lang]
    parts: list[str] = []
    for tool in tools:
        parts.append(f'    <a class="card" href="{chrome["tools_path"]}/{tool.slug}">')
        parts.append(f"        <h3>{html.escape(tool.title)}</h3>")
        parts.append(f"        <p>{html.escape(tool.description)}</p>")
        parts.append(f'        <p class="more">{chrome["open_tool"]} &rarr;</p>')
        parts.append("    </a>")
    return parts


def _slugify_heading(heading: str) -> str:
    """Turn a section heading into a stable #anchor.

    Anchors let the table of contents link into the page, and let Google show
    "jump to" links in the results. Accents and punctuation are dropped rather
    than percent-encoded, which keeps the URLs readable.

    Example:
        >>> _slugify_heading("1. Regarder qui a produit le fichier")
        'regarder-qui-a-produit-le-fichier'
    """
    import unicodedata

    # NFKD splits "é" into "e" + combining accent; the ASCII encode then drops
    # the accent rather than the whole letter.
    normalised = unicodedata.normalize("NFKD", heading)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii").lower()
    kept = [c if c.isalnum() else "-" for c in ascii_only]
    slug = "".join(kept)
    # Collapse runs of separators left by punctuation ("2d-doc :" -> "2d-doc-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def head_inner(
    *,
    lang: str,
    title: str,
    description: str,
    canonical: str,
    alternates: dict[str, str],
    json_ld: list[dict],
    og_type: str = "article",
    extra_head: str = "",
    og_description: str | None = None,
    twitter_description: str | None = None,
    style: str | None = None,
) -> str:
    """Build what goes between <head> and </head> on every page of the site.

    Args:
        lang: "fr" or "en".
        title: <title> text, already plain (escaped here).
        description: meta description text; also the Open Graph and Twitter
            description unless the two overrides below are given.
        og_description: shorter copy for link previews (Facebook, LinkedIn,
            Slack...), which truncate harder than a results page.
        twitter_description: same for Twitter/X cards.
        canonical: absolute URL of this page.
        alternates: {hreflang: absolute URL}, including x-default.
        json_ld: structured-data objects, one <script> each.
        og_type: Open Graph type — "article" for guides, "website" for the
            landing pages.
        extra_head: extra lines inserted before </head> (already escaped).
        style: the stylesheet block. Default: the guides' stylesheet,
            inlined. The web app passes a <link> to its own instead.
    """
    if style is None:
        style = f"    <style>\n{STYLESHEET}    </style>"
    chrome = CHROME[lang]
    esc = html.escape
    alt_links = "\n".join(
        f'    <link rel="alternate" hreflang="{code}" href="{url}">'
        for code, url in alternates.items()
    )
    ld_blocks = "\n".join(
        '    <script type="application/ld+json">\n'
        + json.dumps(block, ensure_ascii=False, indent=8)
        + "\n    </script>"
        for block in json_ld
    )
    return f"""    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{esc(title)}</title>
    <meta name="description" content="{esc(description)}">
    <meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1">
    <meta name="theme-color" content="#1a1a2e">
    <link rel="icon" href="/favicon.ico" type="image/x-icon">
    <link rel="icon" href="/favicon-48x48.png" type="image/png" sizes="48x48">
    <link rel="apple-touch-icon" href="/apple-touch-icon.png">
    <link rel="canonical" href="{canonical}">
{alt_links}
    <meta property="og:type" content="{og_type}">
    <meta property="og:title" content="{esc(title)}">
    <meta property="og:description" content="{esc(og_description or description)}">
    <meta property="og:url" content="{canonical}">
    <meta property="og:image" content="{SITE}/og-image.png">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="630">
    <meta property="og:site_name" content="TrustyFile">
    <meta property="og:locale" content="{chrome['locale']}">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{esc(title)}">
    <meta name="twitter:description" content="{esc(twitter_description or description)}">
    <meta name="twitter:image" content="{SITE}/og-image.png">
{style}
{ld_blocks}
{extra_head}"""


def _head(*, lang: str, **kwargs) -> str:
    """A generated page from <!DOCTYPE> to <body> (see head_inner for the arguments)."""
    return (
        f'<!DOCTYPE html>\n<html lang="{lang}">\n<head>\n'
        f"{head_inner(lang=lang, **kwargs)}</head>\n<body>"
    )


def _nav(lang: str, other_url: str) -> str:
    """Breadcrumb-ish top nav plus the language switch."""
    chrome = CHROME[lang]
    return f"""    <nav class="top">
        <a href="{chrome['app_url']}">{chrome['nav_home']}</a>
        <span class="sep">/</span>
        <a href="{chrome['index_path']}">{chrome['nav_guides']}</a>
        <a class="lang" href="{other_url}">{chrome['other_lang_label']}</a>
    </nav>
"""


def _footer(lang: str) -> str:
    chrome = CHROME[lang]
    return f"""    <footer>
        <p><a href="{chrome['app_url']}">TrustyFile</a>
        &middot; <a href="{chrome['index_path']}">{chrome['nav_guides']}</a>
        &middot; <a href="mailto:contact@trustyfile.com">{chrome['footer_contact']}</a></p>
    </footer>
</body>
</html>
"""


def _key_points_html(lang: str, points: list[str]) -> list[str]:
    """The "In short" box shown right under a guide's intro.

    Answer engines (ChatGPT, Claude, Gemini) cut a page into passages and
    quote the ones that answer the question on their own. A short list of
    self-contained sentences near the top is the easiest thing to quote.

    The points are plain text and are escaped: a "<" in "amount < 1000" must
    show as a character, not open a tag. The intro, by contrast, holds links
    and is inserted as HTML.

    Args:
        lang: "fr" or "en", picks the box title.
        points: the guide's key_points; may be empty.

    Returns:
        HTML lines to append to the page, or [] when there are no points
        (the guide then renders exactly as before).

    Example:
        _key_points_html("en", ["Check the Producer field."]) ->
        ['    <aside class="key-points">', '        <p>In short</p>',
         '        <ul>', '            <li>Check the Producer field.</li>',
         '        </ul>', '    </aside>']
    """
    if not points:
        return []
    lines = [
        '    <aside class="key-points">',
        f"        <p>{html.escape(CHROME[lang]['key_points'])}</p>",
        "        <ul>",
    ]
    for point in points:
        lines.append(f"            <li>{html.escape(point)}</li>")
    lines += ["        </ul>", "    </aside>"]
    return lines


def render_guide(lang: str, guide: Guide) -> str:
    """Render one guide page to a complete HTML document."""
    chrome = CHROME[lang]
    other_lang = "en" if lang == "fr" else "fr"
    canonical = f"{SITE}{chrome['guides_path']}/{guide.slug}"

    # hreflang set. A guide without a translation still declares itself, which
    # is required: hreflang annotations must be reciprocal and self-referential.
    alternates = {lang: canonical}
    if guide.counterpart:
        other_canonical = (
            f"{SITE}{CHROME[other_lang]['guides_path']}/{guide.counterpart}"
        )
        alternates[other_lang] = other_canonical
        # French is the site's default language, hence x-default.
        alternates["x-default"] = canonical if lang == "fr" else other_canonical
        other_url = f"{CHROME[other_lang]['guides_path']}/{guide.counterpart}"
    else:
        other_url = CHROME[other_lang]["index_path"]

    article_ld = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": guide.title,
        "description": guide.description,
        "inLanguage": lang,
        "datePublished": guide.published,
        "dateModified": guide.modified,
        "mainEntityOfPage": {"@type": "WebPage", "@id": canonical},
        "image": f"{SITE}/og-image.png",
        # "@id" points at the Organization node the landing page declares
        # (landing_head_args): search and answer engines then read every guide
        # as written by that one entity, not by a string that happens to match.
        "author": {
            "@type": "Organization",
            "@id": f"{SITE}/#organization",
            "name": "TrustyFile",
            "url": SITE,
        },
        "publisher": {
            "@type": "Organization",
            "@id": f"{SITE}/#organization",
            "name": "TrustyFile",
            "url": SITE,
            "logo": {
                "@type": "ImageObject",
                "url": f"{SITE}/favicon-192x192.png",
            },
        },
    }
    breadcrumb_ld = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {
                "@type": "ListItem",
                "position": 1,
                "name": "TrustyFile",
                "item": chrome["app_url"],
            },
            {
                "@type": "ListItem",
                "position": 2,
                "name": chrome["nav_guides"],
                "item": f"{SITE}{chrome['index_path']}",
            },
            {"@type": "ListItem", "position": 3, "name": guide.title},
        ],
    }

    parts = [
        _head(
            lang=lang,
            title=guide.title,
            description=guide.description,
            canonical=canonical,
            alternates=alternates,
            json_ld=[article_ld, breadcrumb_ld]
            + ([_faq_ld(guide.faq)] if guide.faq else []),
        ),
        _nav(lang, other_url),
        f"    <h1>{html.escape(guide.heading)}</h1>",
        f"    {_updated_line(lang, guide.modified)}",
        f'    <p class="intro">{guide.intro}</p>',
    ]
    parts.extend(_key_points_html(lang, guide.key_points))

    # Table of contents. Only worth rendering when there are enough sections
    # for a reader to actually want to skip around.
    if len(guide.sections) >= 4:
        parts.append(f"    <h2>{chrome['toc']}</h2>")
        parts.append('    <ol class="toc">')
        for section in guide.sections:
            anchor = _slugify_heading(section.heading)
            parts.append(
                f'        <li><a href="#{anchor}">{html.escape(section.heading)}</a></li>'
            )
        parts.append("    </ol>")

    for section in guide.sections:
        anchor = _slugify_heading(section.heading)
        parts.append(f'    <h2 id="{anchor}">{html.escape(section.heading)}</h2>')
        for paragraph in section.paragraphs:
            parts.append(f"    <p>{paragraph}</p>")
        if section.bullets:
            parts.append("    <ul>")
            for bullet in section.bullets:
                parts.append(f"        <li>{bullet}</li>")
            parts.append("    </ul>")

    # The guide's own FAQ, phrased the way people ask an assistant ("Does a
    # valid IBAN prove...?"). Same component and look as the hub's FAQ.
    if guide.faq:
        parts.extend(_faq_html(lang, guide.faq))

    # Checker tools the reader is likely to want next (the fake-RIB guide
    # offers the IBAN checker, the invoice guide the SIRET one). A guide is the
    # page a searcher lands on; the tool is what they came to do. Also the
    # only inbound links the tools get besides the hub and the landing.
    if guide.tools:
        by_slug = {tool.slug: tool for tool in TOOLS[lang]}
        parts.append(f"    <h2>{html.escape(chrome['tools_heading'])}</h2>")
        parts.extend(_tool_cards(lang, [by_slug[slug] for slug in guide.tools]))

    # Related guides: every sibling in the same language. Internal links are
    # the main way crawlers discover pages and how link signal flows between
    # them; the sitemap alone is a much weaker channel. h3 (not h2) so the
    # cards do not pollute the article's own outline.
    siblings = [g for g in GUIDES[lang] if g.slug != guide.slug]
    if siblings:
        parts.append(f"    <h2>{chrome['related']}</h2>")
        for other in siblings:
            url = f"{chrome['guides_path']}/{other.slug}"
            parts.append(f'    <a class="card" href="{url}">')
            parts.append(f"        <h3>{html.escape(other.title)}</h3>")
            parts.append(f"        <p>{html.escape(other.description)}</p>")
            parts.append("    </a>")

    parts.append(f'    <a class="cta" href="{chrome["app_url"]}">{chrome["cta"]}</a>')
    parts.append(_footer(lang))
    return "\n".join(parts)


def _faq_ld(faq: list[tuple[str, str]]) -> dict:
    """FAQPage structured data for a FAQ rendered on the same page.

    Used by the guides index (CHROME[lang]["faq"]) and by each guide
    (guide.faq). Only valid because the same questions and answers are
    visible on the page to every visitor: structured data describing content
    that is not shown is a Google Search spam-policy violation in its own
    right. Both are generated from the one list passed here.

    Args:
        faq: (question, answer) pairs, plain text.

    Returns:
        A schema.org FAQPage dict, ready for json.dumps.
    """
    return {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": question,
                "acceptedAnswer": {"@type": "Answer", "text": answer},
            }
            for question, answer in faq
        ],
    }


def _faq_html(lang: str, faq: list[tuple[str, str]]) -> list[str]:
    """The visible FAQ: a heading and a definition list (same look everywhere).

    Args:
        lang: picks the "Frequently asked questions" heading.
        faq: (question, answer) pairs, plain text, escaped here.
    """
    chrome = CHROME[lang]
    parts = [f"    <h2>{html.escape(chrome['faq_heading'])}</h2>", '    <dl class="faq">']
    for question, answer in faq:
        parts.append(f"        <dt>{html.escape(question)}</dt>")
        parts.append(f"        <dd>{html.escape(answer)}</dd>")
    parts.append("    </dl>")
    return parts


def render_index(lang: str) -> str:
    """Render the guides index: the hub that links to every guide, plus the FAQ."""
    chrome = CHROME[lang]
    other_lang = "en" if lang == "fr" else "fr"
    canonical = f"{SITE}{chrome['index_path']}"
    other_canonical = f"{SITE}{CHROME[other_lang]['index_path']}"

    alternates = {
        lang: canonical,
        other_lang: other_canonical,
        "x-default": canonical if lang == "fr" else other_canonical,
    }

    collection_ld = {
        "@context": "https://schema.org",
        "@type": "CollectionPage",
        "name": chrome["index_heading"],
        "description": chrome["index_description"],
        "url": canonical,
        "inLanguage": lang,
        "hasPart": [
            {
                "@type": "Article",
                "headline": guide.title,
                "description": guide.description,
                "url": f"{SITE}{chrome['guides_path']}/{guide.slug}",
            }
            for guide in GUIDES[lang]
        ],
    }

    parts = [
        _head(
            lang=lang,
            title=chrome["index_title"],
            description=chrome["index_description"],
            canonical=canonical,
            alternates=alternates,
            json_ld=[collection_ld, _faq_ld(chrome["faq"])],
        ),
        _nav(lang, CHROME[other_lang]["index_path"]),
        f"    <h1>{html.escape(chrome['index_heading'])}</h1>",
        f'    <p class="intro">{chrome["index_intro"]}</p>',
    ]
    # One section per audience: with ten guides a flat list makes a landlord
    # scan past supplier fraud and an accounts-payable clerk past pay slips.
    for audience in AUDIENCES:
        section = [g for g in GUIDES[lang] if g.audience == audience]
        if not section:
            continue
        parts.append(f'    <h2 id="{audience}">{html.escape(chrome["audience_headings"][audience])}</h2>')
        parts.extend(_guide_cards(lang, section))
    # Free checker tools: the hub is the one place that links every tool.
    parts.append(f'    <h2 id="outils">{html.escape(chrome["tools_heading"])}</h2>')
    parts.append(f'    <p>{html.escape(chrome["tools_intro"])}</p>')
    parts.extend(_tool_cards(lang, TOOLS[lang]))
    parts.extend(_faq_html(lang, chrome["faq"]))
    parts.append(f"    <p>{chrome['index_outro']}</p>")
    parts.append(f'    <a class="cta" href="{chrome["app_url"]}">{chrome["cta"]}</a>')
    parts.append(_footer(lang))
    return "\n".join(parts)


def landing_head_args(lang: str) -> dict:
    """Arguments of `head_inner` for the landing page: title, canonical,
    hreflang alternates, structured data. Used by the web app's landing
    page (src/web/app.py), which is served to visitors and crawlers alike."""
    chrome = CHROME[lang]
    other_lang = "en" if lang == "fr" else "fr"
    esc = html.escape

    fr_home = f"{SITE}{CHROME['fr']['landing_path']}"
    en_home = f"{SITE}{CHROME['en']['landing_path']}"
    canonical = fr_home if lang == "fr" else en_home
    alternates = {"fr": fr_home, "en": en_home, "x-default": fr_home}
    id_suffix = "" if lang == "fr" else f"-{lang}"

    graph_ld = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "Organization",
                "@id": f"{SITE}/#organization",
                "name": "TrustyFile",
                "url": SITE,
                "logo": f"{SITE}/favicon-192x192.png",
                "email": "contact@trustyfile.com",
                "description": chrome["landing_webapp_description"],
                # schema.org's slot for "how this organisation works": the
                # method page, so an engine describing TrustyFile can cite it.
                "publishingPrinciples": (
                    f"{SITE}{chrome['guides_path']}/{chrome['method_slug']}"
                ),
            },
            {
                "@type": "WebSite",
                "@id": f"{SITE}/#website{id_suffix}",
                "name": "TrustyFile",
                "url": canonical,
                "inLanguage": lang,
                "publisher": {"@id": f"{SITE}/#organization"},
            },
            {
                "@type": "WebApplication",
                "@id": f"{SITE}/#webapp{id_suffix}",
                "name": "TrustyFile",
                "url": canonical,
                "description": chrome["landing_webapp_description"],
                "applicationCategory": "SecurityApplication",
                "operatingSystem": "Any",
                "inLanguage": lang,
                "featureList": chrome["landing_feature_list"],
                "offers": {"@type": "Offer", "price": "0", "priceCurrency": "EUR"},
                "image": f"{SITE}/og-image.png",
                "publisher": {"@id": f"{SITE}/#organization"},
            },
        ],
    }

    extra_head = "\n".join(
        [
            '    <link rel="icon" href="/favicon-192x192.png" type="image/png" sizes="192x192">',
            '    <link rel="manifest" href="/site.webmanifest">',
            f'    <meta property="og:locale:alternate" content="{CHROME[other_lang]["locale"]}">',
            f'    <meta property="og:image:alt" content="{esc(chrome["landing_og_image_alt"])}">',
            "",
        ]
    )

    return dict(
        lang=lang,
        title=chrome["landing_title"],
        description=chrome["landing_description"],
        canonical=canonical,
        alternates=alternates,
        json_ld=[graph_ld],
        og_type="website",
        extra_head=extra_head,
        og_description=chrome["landing_og_description"],
        twitter_description=chrome["landing_twitter_description"],
    )


def render_tool(lang: str, tool: Tool) -> str:
    """Render one checker page: guide chrome around a client-side form.

    The validator runs in the visitor's browser (CHECKSUM_JS, inlined), so the
    page is fully static and the number typed never reaches a server — which
    is also what the page says, so keep that true.
    """
    chrome = CHROME[lang]
    other_lang = "en" if lang == "fr" else "fr"
    esc = html.escape
    canonical = f"{SITE}{chrome['tools_path']}/{tool.slug}"

    alternates = {lang: canonical}
    if tool.counterpart:
        other_canonical = f"{SITE}{CHROME[other_lang]['tools_path']}/{tool.counterpart}"
        alternates[other_lang] = other_canonical
        alternates["x-default"] = canonical if lang == "fr" else other_canonical
        other_url = f"{CHROME[other_lang]['tools_path']}/{tool.counterpart}"
    else:
        other_url = CHROME[other_lang]["index_path"]

    app_ld = {
        "@context": "https://schema.org",
        "@type": "WebApplication",
        "name": tool.title,
        "url": canonical,
        "description": tool.description,
        "applicationCategory": "UtilityApplication",
        "operatingSystem": "Any",
        "browserRequirements": "Requires JavaScript",
        "isAccessibleForFree": True,
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "EUR"},
        "inLanguage": lang,
        "publisher": {"@type": "Organization", "name": "TrustyFile", "url": SITE},
    }
    breadcrumb_ld = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {"@type": "ListItem", "position": 1, "name": "TrustyFile", "item": chrome["app_url"]},
            {"@type": "ListItem", "position": 2, "name": chrome["tools_heading"],
             "item": f"{SITE}{chrome['index_path']}#outils"},
            {"@type": "ListItem", "position": 3, "name": tool.title},
        ],
    }

    # Messages for the page script. "</" must not appear inside a <script>
    # block, even a JSON one; json.dumps does not escape it, so do it here.
    i18n = dict(chrome["tool_i18n"], kind=tool.kind)
    i18n_json = json.dumps(i18n, ensure_ascii=False).replace("</", "<\\/")

    parts = [
        _head(
            lang=lang,
            title=tool.title,
            description=tool.description,
            canonical=canonical,
            alternates=alternates,
            json_ld=[app_ld, breadcrumb_ld],
            og_type="website",
        ),
        _nav(lang, other_url),
        f"    <h1>{esc(tool.heading)}</h1>",
        f'    <p class="intro">{tool.intro}</p>',
        '    <form class="tool" id="tf-form" autocomplete="off">',
        # An MRZ is two or three lines, so it gets a textarea; every other
        # number fits on one line.
        (
            f'        <textarea id="tf-input" rows="3" spellcheck="false" placeholder="{esc(tool.placeholder)}" aria-label="{esc(tool.heading)}"></textarea>'
            if tool.kind == "mrz"
            else f'        <input id="tf-input" type="text" spellcheck="false" placeholder="{esc(tool.placeholder)}" aria-label="{esc(tool.heading)}">'
        ),
        f'        <button type="submit">{esc(chrome["tool_button"])}</button>',
        "    </form>",
        '    <div id="tf-result" class="result" role="status" aria-live="polite"></div>',
        f'    <p class="privacy">{esc(chrome["tool_privacy"])}</p>',
    ]
    for section in tool.sections:
        anchor = _slugify_heading(section.heading)
        parts.append(f'    <h2 id="{anchor}">{html.escape(section.heading)}</h2>')
        for paragraph in section.paragraphs:
            parts.append(f"    <p>{paragraph}</p>")
        if section.bullets:
            parts.append("    <ul>")
            for bullet in section.bullets:
                parts.append(f"        <li>{bullet}</li>")
            parts.append("    </ul>")

    siblings = [t for t in TOOLS[lang] if t.slug != tool.slug]
    if siblings:
        parts.append(f"    <h2>{chrome['related_tools']}</h2>")
        parts.extend(_tool_cards(lang, siblings))

    # The guides this tool serves (the IBAN checker -> fake RIB, supplier
    # fraud). A visitor who came for one number often has the whole document
    # in front of them; and these are the tools' only links back to the guides.
    if tool.guides:
        by_slug = {guide.slug: guide for guide in GUIDES[lang]}
        parts.append(f"    <h2>{chrome['related']}</h2>")
        parts.extend(_guide_cards(lang, [by_slug[slug] for slug in tool.guides]))

    parts.append(f'    <a class="cta" href="{chrome["app_url"]}">{chrome["cta"]}</a>')
    parts.append(f'    <script type="application/json" id="tf-i18n">{i18n_json}</script>')
    parts.append("    <script>")
    parts.append(CHECKSUM_JS.rstrip("\n"))
    parts.append(TOOL_GLUE_JS.rstrip("\n"))
    parts.append("    </script>")
    parts.append(_footer(lang))
    return "\n".join(parts)


@dataclass
class SitemapEntry:
    """One <url> in sitemap.xml, with its hreflang alternates.

    Attributes:
        loc: the URL itself.
        fr: French equivalent, for the xhtml:link annotation.
        en: English equivalent.
        priority: relative importance within this site only. Google largely
            ignores it; kept because other engines still read it.
        lastmod: ISO date of the last substantive change to that URL. Google
            does use this one — provided it is truthful, so it comes from the
            page's own date, not from the build time.
    """

    loc: str
    fr: str
    en: str
    priority: str
    lastmod: str


def sitemap_entries() -> list[SitemapEntry]:
    """Every indexable URL: the two app landing pages plus all guides.

    hreflang annotations must be reciprocal — each URL lists both languages,
    including itself — or Google discards the whole cluster.
    """
    fr_home, en_home = SITE + "/", SITE + "/en"
    # The landing pages and the hubs list every guide and tool, so they last
    # changed when the newest of those did.
    newest = latest_content_date()
    entries = [
        SitemapEntry(fr_home, fr_home, en_home, "1.0", newest),
        SitemapEntry(en_home, fr_home, en_home, "0.9", newest),
    ]
    for lang in ("fr", "en"):
        other = "en" if lang == "fr" else "fr"
        index = f"{SITE}{CHROME[lang]['index_path']}"
        entries.append(
            SitemapEntry(
                index,
                f"{SITE}{CHROME['fr']['index_path']}",
                f"{SITE}{CHROME['en']['index_path']}",
                "0.8",
                newest,
            )
        )
        for guide in GUIDES[lang]:
            loc = f"{SITE}{CHROME[lang]['guides_path']}/{guide.slug}"
            if guide.counterpart:
                other_loc = f"{SITE}{CHROME[other]['guides_path']}/{guide.counterpart}"
                fr_url, en_url = (loc, other_loc) if lang == "fr" else (other_loc, loc)
            else:
                # No translation yet: the page is its own only alternate.
                fr_url = en_url = loc
            entries.append(SitemapEntry(loc, fr_url, en_url, "0.7", guide.modified))
        for tool in TOOLS[lang]:
            loc = f"{SITE}{CHROME[lang]['tools_path']}/{tool.slug}"
            other_loc = f"{SITE}{CHROME[other]['tools_path']}/{tool.counterpart}"
            fr_url, en_url = (loc, other_loc) if lang == "fr" else (other_loc, loc)
            entries.append(SitemapEntry(loc, fr_url, en_url, "0.7", tool.modified))
    return entries


def latest_content_date() -> str:
    """The most recent `modified` date across every guide and tool.

    ISO dates sort as strings, so max() is enough.
    """
    dates = [g.modified for guides in GUIDES.values() for g in guides]
    dates += [t.modified for tools in TOOLS.values() for t in tools]
    return max(dates)


def render_sitemap() -> str:
    """Render sitemap.xml.

    Generated rather than hand-edited: a guide added to GUIDES but forgotten in
    the sitemap is invisible to search engines, and that mistake is silent.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!-- Generated by scripts/build_guides.py - do not edit by hand. -->',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"',
        '        xmlns:xhtml="http://www.w3.org/1999/xhtml">',
    ]
    for entry in sitemap_entries():
        lines += [
            "  <url>",
            f"    <loc>{entry.loc}</loc>",
            f'    <xhtml:link rel="alternate" hreflang="fr" href="{entry.fr}" />',
            f'    <xhtml:link rel="alternate" hreflang="en" href="{entry.en}" />',
            f'    <xhtml:link rel="alternate" hreflang="x-default" href="{entry.fr}" />',
            f"    <lastmod>{entry.lastmod}</lastmod>",
            "    <changefreq>monthly</changefreq>",
            f"    <priority>{entry.priority}</priority>",
            "  </url>",
        ]
    lines.append("</urlset>")
    return "\n".join(lines) + "\n"


def render_llms_txt() -> str:
    """Render /llms.txt: a Markdown map of the site for AI answer engines.

    The llms.txt convention (llmstxt.org) is a plain Markdown file at the site
    root: an h1, a one-line summary in a blockquote, then h2 sections of
    "- [title](url): description" links. Assistants that browse (ChatGPT,
    Perplexity, Claude — all already in the Caddyfile's crawler list) read it
    to find the pages worth citing, and citations are a growing referral
    source. Everything listed is a public page served identically to every
    visitor, so this adds no content the site does not already show.
    """
    lines = [
        "# TrustyFile",
        "",
        "> Vérificateur de documents PDF gratuit : déposez un PDF et obtenez un "
        "score de confiance fondé sur des indices techniques (métadonnées, "
        "révisions, polices, texte masqué, 2D-Doc, identifiants d'entreprise). "
        "Free online PDF document fraud detector: upload a PDF and get a trust "
        "score built from technical evidence.",
        "",
        "TrustyFile est un outil d'aide à la décision, pas une autorité de "
        "certification : un score élevé signale peu de traces de manipulation, "
        "il ne prouve pas l'authenticité. Les documents ne sont pas conservés. "
        "Decision support, not certification: a high score means few signs of "
        "tampering, not proof of authenticity. Documents are not retained.",
        "",
        "## Application",
        "",
        f"- [TrustyFile (français)]({SITE}/): analyse gratuite d'un PDF, "
        "sans inscription, résultat en quelques secondes.",
        f"- [TrustyFile (English)]({SITE}/en): free PDF analysis, no sign-up, "
        "results in seconds.",
        "",
        "## API",
        "",
        f"- [Documentation de l'API / API docs]({SITE}/api/docs): "
        "POST /api/v1/analyze — un PDF en entrée, le rapport d'analyse JSON en "
        "sortie (le même que le téléchargement de la page de résultats) ; "
        "clé d'API requise, quota mensuel. "
        "One PDF in, the JSON analysis report out — same schema as the "
        "results-page download; API key required, monthly quota.",
        "",
    ]
    headings = {
        "fr": ("Guides (français)", "Outils gratuits (français)"),
        "en": ("Guides (English)", "Free tools (English)"),
    }
    for lang in ("fr", "en"):
        chrome = CHROME[lang]
        guides_heading, tools_heading = headings[lang]
        lines.append(f"## {guides_heading}")
        lines.append("")
        lines.append(f"- [{chrome['index_heading']}]({SITE}{chrome['index_path']})")
        for guide in GUIDES[lang]:
            url = f"{SITE}{chrome['guides_path']}/{guide.slug}"
            lines.append(f"- [{guide.title}]({url}): {guide.description}")
        lines.append("")
        lines.append(f"## {tools_heading}")
        lines.append("")
        for tool in TOOLS[lang]:
            url = f"{SITE}{chrome['tools_path']}/{tool.slug}"
            lines.append(f"- [{tool.title}]({url}): {tool.description}")
        lines.append("")
    lines.append("## Optional")
    lines.append("")
    lines.append(f"- [Sitemap]({SITE}/sitemap.xml)")
    lines.append("- [Contact](mailto:contact@trustyfile.com)")
    return "\n".join(lines) + "\n"


def build() -> dict[Path, str]:
    """Every generated file. Returns {absolute path: content}, without writing."""
    pages: dict[Path, str] = {}
    for lang in ("fr", "en"):
        pages[OUTPUT_DIR / lang / "index.html"] = render_index(lang)
        for guide in GUIDES[lang]:
            pages[OUTPUT_DIR / lang / f"{guide.slug}.html"] = render_guide(lang, guide)
        for tool in TOOLS[lang]:
            pages[ROOT / "public" / "tools" / lang / f"{tool.slug}.html"] = render_tool(lang, tool)
    # The same validators the pages inline, as a Node module for the parity
    # test (tests/test_checksum_vectors.py). Not served by Caddy.
    pages[ROOT / "public" / "tools" / "checksums.js"] = CHECKSUM_JS
    pages[ROOT / "public" / "sitemap.xml"] = render_sitemap()
    pages[ROOT / "public" / "llms.txt"] = render_llms_txt()
    return pages
