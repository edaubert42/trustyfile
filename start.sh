#!/bin/bash
# Start the app in the background on port 8000 (internal only)
# Then start Caddy in the foreground on $PORT (exposed to internet)

# Install Caddy if not already present.
#
# This download is not ideal — unpinned and unverified — but adding caddy to
# nixPkgs did NOT work: the build succeeded while the binary was absent from
# the runtime image PATH, which took the site down. Do not "fix" this by
# moving it back to nixPkgs without first confirming `caddy` resolves at
# RUNTIME, not just at build time.
if ! command -v caddy &> /dev/null; then
    echo "Installing Caddy..."
    curl -sL "https://caddyserver.com/api/download?os=linux&arch=amd64" -o /usr/local/bin/caddy
    chmod +x /usr/local/bin/caddy
fi

# The site: one uvicorn process serves the pages and the API
# (src/web/app.py) on 127.0.0.1:8000; Caddy proxies to it. One worker on
# purpose: the results of an upload are kept in that process's memory
# (src/web/store.py), so a second worker would not find them; and analyses
# are CPU bound, the app caps their concurrency itself.
uvicorn src.web.app:app --host 127.0.0.1 --port 8000 --workers 1 &

# Give the app a moment to start before Caddy begins proxying
sleep 2

caddy run --config /app/Caddyfile --adapter caddyfile
