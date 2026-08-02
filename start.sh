#!/bin/bash
# Start Streamlit in the background on port 8501 (internal only)
# Then start Caddy in the foreground on $PORT (exposed to internet)

# Caddy comes from nixpacks.toml (nixPkgs), version-pinned and hash-verified at
# build time — see the comment there for why downloading it at boot was wrong.
# Fail loudly if the build did not provide it, rather than starting Streamlit
# and then dying with nothing listening on $PORT.
if ! command -v caddy &> /dev/null; then
    echo "FATAL: caddy not found — check nixPkgs in nixpacks.toml" >&2
    exit 1
fi

streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &

# Give Streamlit a moment to start before Caddy begins proxying
sleep 2

caddy run --config /app/Caddyfile --adapter caddyfile
