#!/bin/bash
# Start the app in the background on port 8000 (internal only)
# Then start Caddy in the foreground on $PORT (exposed to internet)

# Caddy: the release the Dockerfile copies into the image, called by its full
# path, never looked up on PATH. Two past outages came from a caddy other than
# the expected one (an older one in the image, 2026-11-10) or none at all
# (nixPkgs, commit e86a8f1). Failing here, loudly, beats a crash loop.
CADDY="/usr/local/bin/caddy"
if [ ! -x "$CADDY" ]; then
    echo "Caddy not found at $CADDY: the image was not built from the Dockerfile." >&2
    exit 1
fi
"$CADDY" version

# The site: one uvicorn process serves the pages and the API
# (src/web/app.py) on 127.0.0.1:8000; Caddy proxies to it. One worker on
# purpose: the results of an upload are kept in that process's memory
# (src/web/store.py), so a second worker would not find them; and analyses
# are CPU bound, the app caps their concurrency itself.
# --no-access-log: every result URL (/r/<token>) is a key to someone's
# document; the access log would copy each one into the Railway logs.
uvicorn src.web.app:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log &

# Give the app a moment to start before Caddy begins proxying
sleep 2

"$CADDY" run --config /app/Caddyfile --adapter caddyfile
