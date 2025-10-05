#!/usr/bin/env bash
# CI validation for systemd unit template and example env files
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNIT="$REPO_ROOT/deploy/systemd/units/justnews@.service"
ENV_EXAMPLE="$REPO_ROOT/deploy/systemd/env/global.env.example"

fail() { echo "[ERROR] $*" >&2; exit 1; }
warn() { echo "[WARN] $*" >&2; }
info() { echo "[INFO] $*"; }

info "Validating unit template exists"
if [[ ! -f "$UNIT" ]]; then
  fail "Unit template not found: $UNIT"
fi

# Ensure recommended dedicated service user/group are used in the template
RECOMMENDED_USER="justnews"
RECOMMENDED_GROUP="justnews"
unit_user=$(grep -E '^User=' "$UNIT" | head -n1 | cut -d= -f2 || true)
unit_group=$(grep -E '^Group=' "$UNIT" | head -n1 | cut -d= -f2 || true)

if [[ -z "$unit_user" ]]; then
  fail "Unit template missing User= entry"
fi
if [[ -z "$unit_group" ]]; then
  fail "Unit template missing Group= entry"
fi

if [[ "$unit_user" != "$RECOMMENDED_USER" ]]; then
  fail "Unit template should use User=$RECOMMENDED_USER (found: $unit_user)"
fi
if [[ "$unit_group" != "$RECOMMENDED_GROUP" ]]; then
  fail "Unit template should use Group=$RECOMMENDED_GROUP (found: $unit_group)"
fi
info "Unit template user/group validated: $unit_user/$unit_group"

# Validate env example does not contain concrete secrets
if [[ ! -f "$ENV_EXAMPLE" ]]; then
  warn "Env example not found: $ENV_EXAMPLE"; exit 0
fi

# Check for obvious hard-coded passwords (simple heuristics)
if grep -E "password\s*=\s*[A-Za-z0-9]{6,}" -n "$ENV_EXAMPLE" >/dev/null; then
  fail "Env example contains a likely hard-coded password; replace with placeholder"
fi
info "Env example sanitized (no obvious hard-coded passwords)"

echo "[OK] Unit and env example validation passed"
