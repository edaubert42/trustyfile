#!/bin/bash
# Start Streamlit in the background on port 8501 (internal only)
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

# Paint the shell dark before Streamlit's JS bundle loads, so a cold load does
# not flash white. Patches the installed package's index.html in place; it is
# idempotent and never fails the boot. See the script's docstring.
python scripts/patch_streamlit_index.py

# API (roadmap #13): uvicorn serves src/api.py on 127.0.0.1:8000 and Caddy
# proxies /api/* to it. One worker on purpose: analyses are CPU bound and the
# app caps concurrency itself (MAX_CONCURRENT_ANALYSES); more workers would
# only multiply that cap on a container shared with Streamlit.
uvicorn src.api:api --host 127.0.0.1 --port 8000 --workers 1 &

streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &

# Give Streamlit a moment to start before Caddy begins proxying
sleep 2

caddy run --config /app/Caddyfile --adapter caddyfile
