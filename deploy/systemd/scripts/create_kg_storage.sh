#!/usr/bin/env bash
# create_kg_storage.sh - ensure archive KG storage directory exists and has correct perms

set -euo pipefail

ENV_FILE="/etc/justnews/global.env"
# Resolve variables from env file if present
if [[ -r "$ENV_FILE" ]]; then
  # shellcheck disable=SC1091
  source "$ENV_FILE"
fi

# Determine destination
DEST="${ARCHIVE_KG_STORAGE:-${MODEL_STORE_ROOT:-/var/lib/justnews/model_store}/kg_storage}"

# Default group name used by reset_and_start.sh
GROUP_NAME="justnews"

# Allow override via env
GROUP_NAME="${JUSTNEWS_GROUP:-$GROUP_NAME}"

if [[ -z "$DEST" ]]; then
  echo "[ERROR] Could not determine ARCHIVE_KG_STORAGE path; check $ENV_FILE" >&2
  exit 2
fi

mkdir -p "$DEST"
chown root:"$GROUP_NAME" "$DEST" || true
chmod 2775 "$DEST" || true

echo "[INFO] Ensured archive KG storage at: $DEST (owner root:$GROUP_NAME, mode 2775)"
exit 0
