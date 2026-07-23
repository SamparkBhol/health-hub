#!/usr/bin/env bash
set -euo pipefail

cd /app

# These protect internal mutation routes and are intentionally regenerated for
# each ephemeral Space runtime. Users do not need to configure credentials.
export DEMO_API_TOKEN="${DEMO_API_TOKEN:-$(python -c 'import secrets; print(secrets.token_hex(32))')}"
export COLLECTOR_API_TOKEN="${COLLECTOR_API_TOKEN:-$(python -c 'import secrets; print(secrets.token_hex(32))')}"

refresh_environment() {
  while true; do
    python scripts/collect_environment.py \
      --mode current-conditions \
      --output-layer data/environment/current_conditions.json || true
    python scripts/collect_seasonal.py || true
    sleep 21600
  done
}

cleanup() {
  kill "${api_pid:-}" "${caddy_pid:-}" "${environment_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

refresh_environment &
environment_pid=$!

uvicorn services.api.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --proxy-headers &
api_pid=$!

for _ in $(seq 1 120); do
  if curl --fail --silent http://127.0.0.1:8000/api/v1/readyz >/dev/null; then
    break
  fi
  sleep 1
done
curl --fail --silent http://127.0.0.1:8000/api/v1/readyz >/dev/null

# Prime live acquisition without delaying the public page. The in-process
# scheduler continues across all enabled source routes while the Space is awake.
(
  for _ in $(seq 1 5); do
    curl --fail --silent \
      -X POST "http://127.0.0.1:8000/api/v1/internal/collector/tick?maximum_jobs=1" \
      -H "X-Collector-Token: $COLLECTOR_API_TOKEN" >/dev/null || true
  done
) &

caddy run --config /app/Caddyfile.huggingface --adapter caddyfile &
caddy_pid=$!

wait -n "$api_pid" "$caddy_pid" "$environment_pid"
