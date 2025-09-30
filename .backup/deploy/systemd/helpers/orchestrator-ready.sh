#!/usr/bin/env bash
# Poll gpu_orchestrator readiness endpoint with backoff
set -euo pipefail
HOST="${1:-127.0.0.1}"
PORT="${2:-8014}"
ENDPOINT="/ready"
TIMEOUT="${TIMEOUT:-60}"
url="http://$HOST:$PORT$ENDPOINT"
end_ts=$(( $(date +%s) + TIMEOUT ))
backoff=1
while [[ $(date +%s) -le $end_ts ]]; do
  if curl -fsS --max-time 5 "$url" >/dev/null; then
    echo "READY"; exit 0
  fi
  sleep "$backoff"; if [[ $backoff -lt 8 ]]; then backoff=$((backoff*2)); fi

done
echo "NOT READY after ${TIMEOUT}s: $url" >&2
exit 1
