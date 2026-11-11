#!/bin/bash
# Start the app in the background on port 8000 (internal only)
# Then start Caddy in the foreground on $PORT (exposed to internet)

# Caddy: one exact release, checked against its published SHA-512, installed
# under its own name and always called by that path.
#
# Why not whatever `caddy` is on PATH: an image can ship its own, older Caddy
# (staging did, 2026-11-10: it rejected `trusted_proxies` in the Caddyfile,
# an option every release since 2.6.3 accepts, and the site crash-looped).
# Why not nixPkgs: adding caddy there built fine while the binary was absent
# from the runtime image PATH, which took the site down.
# To upgrade: change both values below (the checksum is in the release's
# caddy_<version>_checksums.txt on GitHub).
CADDY_VERSION="2.11.9"
CADDY_SHA512="0e85544340290a16ce4bfb76fe7d65e9de1bf3a6e67f5ccfff30b30d27ea4c73e078077b7bf3e58ae60b6b12e44327705287059c775bca5782e8ee85ea7edd75"
CADDY="/usr/local/bin/caddy-${CADDY_VERSION}"

if [ ! -x "$CADDY" ]; then
    echo "Installing Caddy ${CADDY_VERSION}..."
    archive="$(mktemp)"
    curl -sfL "https://github.com/caddyserver/caddy/releases/download/v${CADDY_VERSION}/caddy_${CADDY_VERSION}_linux_amd64.tar.gz" -o "$archive"
    # Refuse to run a download that is not byte for byte the release.
    if ! echo "${CADDY_SHA512}  ${archive}" | sha512sum -c - > /dev/null; then
        echo "Caddy ${CADDY_VERSION}: download failed or checksum mismatch, not starting." >&2
        exit 1
    fi
    tar -xzf "$archive" -C "$(dirname "$CADDY")" caddy
    mv "$(dirname "$CADDY")/caddy" "$CADDY"
    chmod +x "$CADDY"
    rm -f "$archive"
fi
"$CADDY" version

# The site: one uvicorn process serves the pages and the API
# (src/web/app.py) on 127.0.0.1:8000; Caddy proxies to it. One worker on
# purpose: the results of an upload are kept in that process's memory
# (src/web/store.py), so a second worker would not find them; and analyses
# are CPU bound, the app caps their concurrency itself.
uvicorn src.web.app:app --host 127.0.0.1 --port 8000 --workers 1 &

# Give the app a moment to start before Caddy begins proxying
sleep 2

"$CADDY" run --config /app/Caddyfile --adapter caddyfile
