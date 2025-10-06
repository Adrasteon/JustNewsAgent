#!/usr/bin/env bash
# e2e_smoke_archive.sh - simple smoke test to verify archive services are reachable
set -euo pipefail

# Load global env if present to pick up custom ports
if [[ -r /etc/justnews/global.env ]]; then
  # shellcheck disable=SC1091
  source /etc/justnews/global.env
fi

# Defaults based on manifest (archive_graphql -> 8020, archive_api -> 8021)
GQL_PORT="${ARCHIVE_GRAPHQL_PORT:-8020}"
API_PORT="${ARCHIVE_API_PORT:-8021}"

GQL_URL="http://127.0.0.1:${GQL_PORT}/health"
API_URL="http://127.0.0.1:${API_PORT}/health"

echo "Checking archive GraphQL health: $GQL_URL"
if curl -fsS --max-time 5 "$GQL_URL" >/dev/null; then
  echo "[OK] archive_graphql healthy"
else
  echo "[FAIL] archive_graphql failed health check: $GQL_URL" >&2
  exit 2
fi

echo "Checking archive API health: $API_URL"
if curl -fsS --max-time 5 "$API_URL" >/dev/null; then
  echo "[OK] archive_api healthy"
else
  echo "[FAIL] archive_api failed health check: $API_URL" >&2
  exit 2
fi

echo "Archive smoke test passed"
exit 0
