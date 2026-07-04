#!/bin/bash
# Start Streamlit in the background on port 8501 (internal only)
# Then start Caddy in the foreground on $PORT (exposed to internet)

# Install Caddy if not already present
if ! command -v caddy &> /dev/null; then
    echo "Installing Caddy..."
    curl -sL "https://caddyserver.com/api/download?os=linux&arch=amd64" -o /usr/local/bin/caddy
    chmod +x /usr/local/bin/caddy
fi

streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &

# Give Streamlit a moment to start before Caddy begins proxying
sleep 2

caddy run --config /app/Caddyfile --adapter caddyfile
