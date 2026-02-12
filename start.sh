#!/bin/bash
# Start Streamlit in the background on port 8501 (internal only)
# Then start Caddy in the foreground on $PORT (exposed to internet)

streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &

# Give Streamlit a moment to start before Caddy begins proxying
sleep 2

caddy run --config /app/Caddyfile --adapter caddyfile
