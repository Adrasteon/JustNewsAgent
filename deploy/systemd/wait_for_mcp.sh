#!/bin/bash
# Delegating wrapper for wait_for_mcp script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL="$SCRIPT_DIR/scripts/wait_for_mcp.sh"
if [[ -x "$REAL" ]]; then
  exec "$REAL" "$@"
else
  echo "ERROR: helper script missing: $REAL" >&2
  exit 2
fi
