#!/usr/bin/env bash
# Wrapper to run repo boot_smoke_test.sh from a stable path
set -euo pipefail
REPO_DIR="/home/adra/justnewsagent/JustNewsAgent"
SCRIPT="$REPO_DIR/deploy/systemd/helpers/boot_smoke_test.sh"
if [[ -x "$SCRIPT" ]]; then
  exec "$SCRIPT"
else
  echo "boot_smoke_test.sh not found or not executable at $SCRIPT" >&2
  exit 0
fi
