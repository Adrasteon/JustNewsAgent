#!/usr/bin/env bash
# Canonical_Test_RUNME.sh
# Single entrypoint to run the canonical test order for the JustNewsAgent repo.
# Usage: ./Canonical_Test_RUNME.sh [--all|--unit|--smoke|--tensorrt|--gpu|--integration|--preflight|--start-dry|--start|--precommit|--coverage|--help]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
# Allow caller to override the Python command (e.g. PY='conda run -n justnews-v2-py312 python')
PY=${PY:-python3}
PYTEST=pytest

# Helpers
info(){ printf "[info] %s\n" "$*"; }
warn(){ printf "[warn] %s\n" "$*"; }
error(){ printf "[error] %s\n" "$*"; }

# Defaults
DO_UNIT=0
DO_SMOKE=0
DO_TRT=0
DO_GPU=0
DO_INTEGRATION=0
DO_PREFLIGHT=0
DO_START_DRY=0
DO_START=0
DO_PRECOMMIT=0
DO_COVERAGE=0

# Simple port check
is_port_in_use(){ ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN || return 1; }

usage(){
  cat <<EOF
Usage: $0 [options]
Options:
  --all            Run all safe tests (unit + smoke + tensorrt)
  --unit           Run unit tests (excludes integration)
  --smoke          Run smoke E2E stub
  --tensorrt       Run tensorrt stub test
  --gpu            Run GPU-marked tests (only if GPU available or --force-gpu)
  --integration    Run integration tests (requires services/DB)
  --preflight      Run ./square-one.sh preflight
  --start-dry      Run ./square-one.sh start --dry-run --yes
  --start          Actually start the system via square-one (interactive unless --yes)
  --precommit      Run pre-commit hooks locally before tests
  --coverage       Run tests with coverage
  --help           Show this help
EOF
}

# Parse args
while [ $# -gt 0 ]; do
  case "$1" in
    --all) DO_UNIT=1; DO_SMOKE=1; DO_TRT=1; shift;;
    --unit) DO_UNIT=1; shift;;
    --smoke) DO_SMOKE=1; shift;;
    --tensorrt) DO_TRT=1; shift;;
    --gpu) DO_GPU=1; shift;;
    --integration) DO_INTEGRATION=1; shift;;
    --preflight) DO_PREFLIGHT=1; shift;;
    --start-dry) DO_START_DRY=1; shift;;
    --start) DO_START=1; shift;;
    --precommit) DO_PRECOMMIT=1; shift;;
    --coverage) DO_COVERAGE=1; shift;;
    -h|--help) usage; exit 0;;
    *) error "Unknown arg: $1"; usage; exit 1;;
  esac
done

# Aggregate exit codes
FAIL_COUNT=0

# Optional: run pre-commit locally
if [ "$DO_PRECOMMIT" -eq 1 ]; then
  info "Running pre-commit hooks (this may install hooks if not present)"
  if command -v pre-commit >/dev/null 2>&1; then
    pre-commit run --all-files || { warn "pre-commit reported issues"; FAIL_COUNT=$((FAIL_COUNT+1)); }
  else
    warn "pre-commit not found in PATH; install in the project environment: conda install -n justnews-v2-py312 -c conda-forge pre-commit (or mamba install -n justnews-v2-py312 -c conda-forge pre-commit)"
    FAIL_COUNT=$((FAIL_COUNT+1))
  fi
fi

# Preflight
if [ "$DO_PREFLIGHT" -eq 1 ]; then
  info "Running preflight via ./square-one.sh --dry-run"
  if [ -x "$ROOT/square-one.sh" ]; then
    "$ROOT/square-one.sh" preflight --dry-run || { warn "preflight (dry-run) reported issues"; FAIL_COUNT=$((FAIL_COUNT+1)); }
  else
    warn "square-one.sh not present or not executable"
    FAIL_COUNT=$((FAIL_COUNT+1))
  fi
fi

# Start dry-run
if [ "$DO_START_DRY" -eq 1 ]; then
  info "Computing start plan via square-one start --dry-run --yes"
  if [ -x "$ROOT/square-one.sh" ]; then
    "$ROOT/square-one.sh" start --dry-run --yes || { warn "start dry-run reported issues"; FAIL_COUNT=$((FAIL_COUNT+1)); }
  else
    warn "square-one.sh not present or not executable"
    FAIL_COUNT=$((FAIL_COUNT+1))
  fi
fi

# Unit tests
if [ "$DO_UNIT" -eq 1 ]; then
  info "Running unit tests (excluding integration)"
  if [ "$DO_COVERAGE" -eq 1 ]; then
    $PY -m pytest --maxfail=5 -q -k "not integration" || { warn "Unit tests failed"; FAIL_COUNT=$((FAIL_COUNT+1)); }
  else
    $PY -m pytest -q -k "not integration" || { warn "Unit tests failed"; FAIL_COUNT=$((FAIL_COUNT+1)); }
  fi
fi

# Smoke E2E stub
if [ "$DO_SMOKE" -eq 1 ]; then
  info "Running smoke E2E stub tests"
  if [ -f "$ROOT/tests/smoke_e2e_stub.py" ]; then
    $PY "$ROOT/tests/smoke_e2e_stub.py" || { warn "Smoke E2E stub failed"; FAIL_COUNT=$((FAIL_COUNT+1)); }
  else
    warn "Smoke E2E stub not found: tests/smoke_e2e_stub.py"
    FAIL_COUNT=$((FAIL_COUNT+1))
  fi
fi

# Tensorrt stub
if [ "$DO_TRT" -eq 1 ]; then
  info "Running Tensorrt stub test"
  if [ -f "$ROOT/tests/test_tensorrt_stub.py" ]; then
    $PY -m pytest -q "$ROOT/tests/test_tensorrt_stub.py" || { warn "Tensorrt stub test failed"; FAIL_COUNT=$((FAIL_COUNT+1)); }
  else
    warn "Tensorrt test not found"
    FAIL_COUNT=$((FAIL_COUNT+1))
  fi
fi

# GPU tests
if [ "$DO_GPU" -eq 1 ]; then
  info "Running GPU-marked tests (ensure GPU and drivers are present)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    info "nvidia-smi present"
    $PY -m pytest -q -k gpu || { warn "GPU-marked tests failed"; FAIL_COUNT=$((FAIL_COUNT+1)); }
  else
    warn "nvidia-smi not found (no NVIDIA GPU detected) — skipping GPU tests"
    FAIL_COUNT=$((FAIL_COUNT+1))
  fi
fi

# Integration tests
if [ "$DO_INTEGRATION" -eq 1 ]; then
  info "Running integration tests (ensure Postgres and services are running)"
  if is_port_in_use 5432; then
    $PY -m pytest -q -k integration || { warn "Integration tests failed"; FAIL_COUNT=$((FAIL_COUNT+1)); }
  else
    warn "Postgres (port 5432) does not appear to be running — integration tests skipped"
    FAIL_COUNT=$((FAIL_COUNT+1))
  fi
fi

# Start actual system (dangerous) — require interactive confirmation
if [ "$DO_START" -eq 1 ]; then
  info "About to start the entire system via square-one.sh (this will try to stop running services first)."
  read -r -p "Proceed? [y/N] " yn
  case "$yn" in
    [Yy]*) "$ROOT/square-one.sh" start ;;
    *) info "Start aborted"; FAIL_COUNT=$((FAIL_COUNT+1));;
  esac
fi

# Final summary and exit code
if [ $FAIL_COUNT -eq 0 ]; then
  info "All selected test steps completed successfully"
  exit 0
else
  warn "$FAIL_COUNT step(s) failed or reported issues"
  exit $FAIL_COUNT
fi
