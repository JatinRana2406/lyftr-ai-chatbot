// Overwritten at container start from the API_BASE_URL env var (see docker/40-runtime-config.sh).
// Leave empty to fall back to the build-time VITE_API_BASE_URL.
window.__APP_CONFIG__ = window.__APP_CONFIG__ || {};
