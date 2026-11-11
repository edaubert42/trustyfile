"""Tell search engines which pages changed, through IndexNow.

WHY THIS EXISTS
---------------
ChatGPT's web answers lean heavily on Bing's index, and Bing learns about a
changed page in one of two ways: it recrawls on its own schedule, or the site
tells it. IndexNow (indexnow.org) is the "site tells it" channel, shared by
Bing, Yandex, Seznam, Naver and others: one POST with a list of URLs, and every
participating engine gets it.

HOW THE ENGINE KNOWS IT IS US
-----------------------------
We publish a random key in a file at the site root, public/<KEY>.txt, served
by Caddy at https://www.trustyfile.com/<KEY>.txt. The engine fetches that file
and checks it contains the key we sent. Only someone who controls the site can
put it there. The key is not a secret: it is public by design.

USAGE (after a deploy, so the pages and the key file are live)
-----
    python scripts/indexnow.py --since 2026-11-11   # pages changed since a date
    python scripts/indexnow.py --since-commit <sha>  # pages changed since a commit
    python scripts/indexnow.py --all                # every page in the sitemap
    add --dry-run to print the list without sending anything

.github/workflows/indexnow.yml runs --since-commit on its own after every
production deploy, with the commit of the previous production deploy.

Submit what changed, not everything on every deploy: IndexNow asks for changed
URLs only, and an engine may start ignoring a site that pings for nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

# Run as `python scripts/indexnow.py`: Python puts scripts/ on the import
# path, not the repository root, so `src` would not be found without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.site.content import SITE  # noqa: E402
from src.site.render import sitemap_entries  # noqa: E402

# Must match public/<KEY>.txt and the Caddyfile route that serves it.
KEY = "6c2a8b428c23f3f77b0dc6a5a4229379"
KEY_URL = f"{SITE}/{KEY}.txt"
ENDPOINT = "https://api.indexnow.org/indexnow"


def changed_urls(since: str | None) -> list[str]:
    """URLs from the sitemap whose lastmod is on or after `since`.

    Args:
        since: ISO date "YYYY-MM-DD", or None for every URL.

    Returns:
        The URLs, in sitemap order.

    Example:
        changed_urls("2026-11-11") -> ["https://www.trustyfile.com/", ...]
    """
    entries = sitemap_entries()
    if since is None:
        return [e.loc for e in entries]
    # ISO dates compare correctly as strings: "2026-11-11" >= "2026-09-03".
    return [e.loc for e in entries if e.lastmod >= since]


# Generated page file (under public/) -> its URL path. Pages served by the app
# itself (the landing pages) have no file here and are never submitted by
# --since-commit: their changes come from code, not from the generator.
_PAGE_DIRS = {
    "public/guides/fr/": "/guides/",
    "public/guides/en/": "/en/guides/",
    "public/tools/fr/": "/outils/",
    "public/tools/en/": "/en/tools/",
}


def url_for_page_file(path: str) -> str | None:
    """The public URL of a generated page file, or None if it is not a page.

    Example:
        url_for_page_file("public/guides/fr/detecter-un-faux-rib.html")
        -> "https://www.trustyfile.com/guides/detecter-un-faux-rib"
        url_for_page_file("public/guides/en/index.html")
        -> "https://www.trustyfile.com/en/guides/"
        url_for_page_file("public/sitemap.xml") -> None
    """
    for prefix, url_path in _PAGE_DIRS.items():
        if path.startswith(prefix) and path.endswith(".html"):
            slug = path[len(prefix):-len(".html")]
            if "/" in slug:
                return None
            return SITE + url_path + ("" if slug == "index" else slug)
    return None


def urls_changed_since_commit(commit: str) -> list[str]:
    """Sitemap URLs whose page changed in substance since `commit`.

    Two filters, both needed:
      - the page file under public/ differs between `commit` and HEAD (a
        deploy that touched no page sends nothing);
      - its lastmod is on or after the commit's date (a page regenerated only
        because the shared CSS or the related-guides cards moved keeps its
        date, so it is not re-announced as new).

    Args:
        commit: the commit the previous production deploy ran.

    Returns:
        The URLs, in sitemap order.
    """
    diff = subprocess.run(
        ["git", "diff", "--name-only", commit, "HEAD", "--", "public/"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    touched = {url for url in map(url_for_page_file, diff) if url}
    commit_date = subprocess.run(
        ["git", "show", "-s", "--format=%cs", commit],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return [e.loc for e in sitemap_entries() if e.loc in touched and e.lastmod >= commit_date]


def build_payload(urls: list[str]) -> dict:
    """The JSON body IndexNow expects (one host, its key, the URL list)."""
    host = SITE.removeprefix("https://")
    return {"host": host, "key": KEY, "keyLocation": KEY_URL, "urlList": urls}


def key_is_live() -> bool:
    """True when the key file is deployed and holds the key.

    Without it the engine answers 403, so checking first gives a clear error
    instead of a cryptic one.
    """
    try:
        with urllib.request.urlopen(KEY_URL, timeout=10) as response:
            return response.read().decode().strip() == KEY
    except OSError:
        return False


def submit(payload: dict) -> int:
    """POST the payload; return the HTTP status (200 or 202 mean accepted)."""
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--since", help="ISO date: submit pages modified on or after it")
    which.add_argument("--since-commit", help="commit SHA: submit pages changed since it")
    which.add_argument("--all", action="store_true", help="submit every sitemap URL")
    parser.add_argument("--dry-run", action="store_true", help="list, do not send")
    args = parser.parse_args()

    if args.since_commit:
        urls = urls_changed_since_commit(args.since_commit)
    else:
        urls = changed_urls(None if args.all else args.since)
    print(f"{len(urls)} URL(s)")
    for url in urls:
        print("  " + url)
    if not urls or args.dry_run:
        return 0

    if not key_is_live():
        print(f"Key file not live at {KEY_URL}: deploy first.", file=sys.stderr)
        return 1
    status = submit(build_payload(urls))
    print(f"IndexNow answered HTTP {status}")
    return 0 if status in (200, 202) else 1


if __name__ == "__main__":
    sys.exit(main())
