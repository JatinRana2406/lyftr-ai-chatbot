#!/bin/sh
# Writes the backend URL from the API_BASE_URL env var into config.js at container start,
# so one image works against any backend without rebuilding.
set -eu
if [ -n "${API_BASE_URL:-}" ]; then
  printf 'window.__APP_CONFIG__ = { API_BASE_URL: "%s" };\n' "$API_BASE_URL" > /usr/share/nginx/html/config.js
  echo "runtime-config: API_BASE_URL=$API_BASE_URL"
fi
