"""
The static site: guides, checker tools, crawler landing pages, sitemap.

- `content.py` — the pages' content as plain data (GUIDES, TOOLS, CHROME).
- `assets.py`  — the CSS and JavaScript the pages inline.
- `render.py`  — content -> HTML; `build()` returns every generated file.

`python scripts/build_guides.py` writes the result to disk. The HTML is
committed: Caddy serves static files and never runs Python.
"""
