"""
Paint the page dark before Streamlit's JavaScript bundle loads.

Streamlit ships a static index.html with no theme styling in its <head>: the
theme only applies once the JS bundle has downloaded, parsed and mounted. On a
cold mobile load that is several seconds of **white** screen on a dark-themed
site — the first thing a visitor sees is a flash of blank white.

Nothing in Streamlit's config fixes this ([theme] is applied by the bundle, so
it is subject to the same delay), and Caddy cannot rewrite a response body
without a plugin that is not in the stock binary. The remaining lever is the
served file itself, so we inject a tiny <style> into its <head>.

Run at boot, before Streamlit starts (see start.sh). Idempotent: it looks for
its own marker and does nothing if the file is already patched, so repeated
deploys and local re-runs are safe.

Usage:
    python scripts/patch_streamlit_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Keep in sync with the dark background used by the app's theme.
BACKGROUND = "#0e1117"

MARKER = "tf-preboot-style"

SNIPPET = f"""    <style id="{MARKER}">
      /* Injected by scripts/patch_streamlit_index.py — see that file.
         Paints the shell dark from the first frame so a cold load does not
         flash white before Streamlit's bundle applies the theme. */
      html, body {{ background: {BACKGROUND} !important; }}
    </style>
"""


def find_index_html() -> Path:
    """Locate the index.html of the installed Streamlit package.

    Returns:
        Path to streamlit/static/index.html.

    Raises:
        FileNotFoundError: if Streamlit is not importable or the file is gone.
    """
    try:
        import streamlit
    except ImportError as exc:  # pragma: no cover - environment problem
        raise FileNotFoundError("streamlit is not installed") from exc

    path = Path(streamlit.__file__).parent / "static" / "index.html"
    if not path.is_file():
        raise FileNotFoundError(f"no index.html at {path}")
    return path


def patch(path: Path) -> bool:
    """Insert the pre-boot style into <head>, unless already present.

    Args:
        path: index.html to patch, modified in place.

    Returns:
        True if the file was modified, False if it was already patched.
    """
    html = path.read_text(encoding="utf-8")
    if MARKER in html:
        return False

    # Anchor on the opening <head> so the rule lands before anything paints.
    needle = "<head>"
    if needle not in html:
        raise ValueError(f"no <head> in {path}; Streamlit's layout changed")

    html = html.replace(needle, needle + "\n" + SNIPPET, 1)
    path.write_text(html, encoding="utf-8")
    return True


def main() -> int:
    try:
        path = find_index_html()
        changed = patch(path)
    except (FileNotFoundError, ValueError) as exc:
        # Never abort the boot over cosmetics — a white flash beats no site.
        print(f"[patch_streamlit_index] skipped: {exc}", file=sys.stderr)
        return 0

    print(f"[patch_streamlit_index] {'patched' if changed else 'already patched'}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
