#!/bin/bash
# Wrapper script to invoke the canonical enable_all helper from scripts/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/scripts/enable_all.sh"

if [[ ! -x "$TARGET" ]]; then
  echo "Expected helper not found or not executable: $TARGET" >&2
  exit 1
fi

exec "$TARGET" "$@"
