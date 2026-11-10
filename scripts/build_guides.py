"""Render the static pages: guides (/guides, /en/guides), the crawler landing
pages (seo-*.html), the checker tools (/outils, /en/tools), sitemap.xml
and llms.txt.

WHY THIS EXISTS
---------------
`seo-fr.html` / `seo-en.html` are served *only to crawlers* — real users get
proxied to Streamlit, whose landing page is built by JavaScript. Serving a
crawler different *content* from what a human sees is cloaking, which Google
can penalise site-wide. Two things keep us on the right side of that line:

1. The crawler pages are generated here (`render_landing`) from the very same
   `src/translations/*.json` strings that `app.py` renders on its landing page
   — headline, tagline, the three steps, both disclaimers, the footer links.
   Same words, static HTML: that is "dynamic rendering", not cloaking. Until
   2026-08-27 they were hand-written and had drifted to ~800 words of prose,
   a feature list and a FAQ that no human ever saw.
2. The guides are real URLs served to **everyone**, linked from the app
   footer. A crawler and a human see the same bytes. The FAQ that used to
   live only on the crawler pages now lives on the guides index, visibly.

They also carry the long-tail queries the single-page app can never rank for
("comment savoir si un PDF a été modifié", "vérifier un justificatif de
domicile"), each landing on a page that answers that one question.

WHY A GENERATOR AND NOT TEN HAND-WRITTEN FILES
----------------------------------------------
Every guide needs the same ~40 lines of head boilerplate: canonical, hreflang,
Open Graph, JSON-LD, the shared stylesheet. Hand-maintaining ten copies means
ten places to forget an update. Here the boilerplate lives once in
`src/site/render.py` and the *content* lives in `src/site/content.py` as plain
data. This file is only the command that writes the result to disk.

The rendered HTML is committed to the repo — Caddy serves static files and
never runs Python. Regenerate after editing `src/site/`:

    python scripts/build_guides.py

`tests/test_guides.py` re-renders and fails if the committed HTML has drifted,
so a forgotten rebuild cannot ship.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run as `python scripts/build_guides.py`: Python puts scripts/ on the import
# path, not the repository root, so `src` would not be found without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.site.content import ROOT  # noqa: E402  (needs the sys.path tweak above)
from src.site.render import build  # noqa: E402


def main() -> None:
    """Write every generated file that changed, and say which."""
    pages = build()
    written = 0
    for path, content in pages.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Skip identical writes so rebuilding does not churn file mtimes.
        if path.exists() and path.read_text(encoding="utf-8") == content:
            continue
        path.write_text(content, encoding="utf-8")
        written += 1
        print(f"wrote {path.relative_to(ROOT)}")
    print(f"{written} file(s) updated, {len(pages)} total")


if __name__ == "__main__":
    main()
