#!/bin/bash
# Lightweight wrapper to call the real enable_all helper in scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/scripts/enable_all.sh" "$@"
