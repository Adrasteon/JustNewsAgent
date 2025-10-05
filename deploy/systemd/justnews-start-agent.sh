#!/bin/bash
# Wrapper that delegates to the scripts/ helper to start an agent. Placed at
# deploy/systemd/justnews-start-agent.sh so installer scripts can copy it to
# /usr/local/bin/justnews-start-agent.sh as needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL="$SCRIPT_DIR/scripts/justnews-start-agent.sh"
if [[ -x "$REAL" ]]; then
  exec "$REAL" "$@"
else
  echo "ERROR: helper script missing: $REAL" >&2
  exit 2
fi
