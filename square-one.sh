#!/bin/bash
# square-one.sh — Safe, idempotent project bootstrap and system starter
# Purpose: single developer/operator entrypoint that verifies environment
#          prerequisites, offers to start the system (dev or production), and
#          can install itself to /usr/local/bin so the `square-one` command
#          becomes globally available.

set -euo pipefail

# --- Basic logging & helper functions (must be available before use) ---
timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { printf "%s [%s] %s\n" "$(timestamp)" "$1" "$2"; }
error() { log "ERROR" "$*" >&2; }
info() { log "INFO" "$*"; }
warn() { log "WARN" "$*"; }
debug() { log "DEBUG" "$*"; }

# Check whether a command exists
has_cmd() { command -v "$1" >/dev/null 2>&1; }

# Determine script path using shell parameter expansion so we don't depend on external `dirname`/`basename`
SCRIPT_SOURCE="${BASH_SOURCE[0]}"
SCRIPT_NAME="${SCRIPT_SOURCE##*/}"
SCRIPT_DIR="${SCRIPT_SOURCE%/*}"
if [ -z "$SCRIPT_DIR" ] || [ "$SCRIPT_DIR" = "$SCRIPT_SOURCE" ]; then
  SCRIPT_DIR="."
fi

# Resolve a canonical project root so the script works when installed globally.
resolve_project_root() {
  # If an operator provided JUSTNEWS_ROOT use it
  if [ -n "${JUSTNEWS_ROOT:-}" ] && [ -d "$JUSTNEWS_ROOT" ]; then
    echo "$JUSTNEWS_ROOT"; return 0
  fi
  # If script appears inside the repo tree (deploy/ exists) prefer that
  if [ -d "$SCRIPT_DIR/deploy" ]; then
    echo "$SCRIPT_DIR"; return 0
  fi
  # If installed under /usr/local/bin, prefer /opt/justnews if present
  if [ "${SCRIPT_DIR}" = "/usr/local/bin" ] || [ "${SCRIPT_DIR}" = "/usr/bin" ]; then
    if [ -d "/opt/justnews" ]; then
      echo "/opt/justnews"; return 0
    fi
  fi
  # Fall back to current directory
  echo "$SCRIPT_DIR"; return 0
}

PROJECT_ROOT="$(resolve_project_root)"
SCRIPT_DIR="$PROJECT_ROOT"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR" || true


# Default project conventions (can be overridden by env)
CONDA_ENV="${CONDA_ENV:-justnews-v2-py312}"

# Load canonical agent manifest if available (single source of truth)
if [ -f "$SCRIPT_DIR/deploy/agents_manifest.sh" ]; then
  # shellcheck disable=SC1090
  . "$SCRIPT_DIR/deploy/agents_manifest.sh"
else
  warn "Missing deploy/agents_manifest.sh — falling back to built-in agent list"
  AGENTS=(
    "mcp_bus|8000"
    "chief_editor|8001"
    "scout|8002"
    "fact_checker|8003"
    "analyst|8004"
    "synthesizer|8005"
    "critic|8006"
    "memory|8007"
    "reasoning|8008"
    "newsreader|8009"
    "db_worker|8010"
    "dashboard|8011"
    "analytics|8012"
    "balancer|8013"
    "gpu_orchestrator|8014"
    "archive_graphql|8020"
    "archive_api|8021"
  )
fi

# Compute required port list from manifest (preserve backwards compatibility)
REQUIRED_PORTS=()
AGENTS=()
if [ "${AGENTS_MANIFEST+set}" = "set" ] && [ "${#AGENTS_MANIFEST[@]}" -gt 0 ]; then
  for entry in "${AGENTS_MANIFEST[@]}"; do
    IFS='|' read -r name module port <<< "$entry"
    REQUIRED_PORTS+=("$port")
    AGENTS+=("$name|$port")
  done
fi
MODEL_STORE_ROOT_DEFAULT="${HOME}/.local/share/justnews/model_store"
MODEL_STORE_ROOT="${MODEL_STORE_ROOT:-$MODEL_STORE_ROOT_DEFAULT}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${HOME}/.local/share/justnews/agents}"

# Default DB envs (respect existing env vars)
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-justnews}"
POSTGRES_USER="${POSTGRES_USER:-justnews_user}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-password123}"

# Port check
is_port_in_use() {
  local port="$1"
  if ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN; then
    return 0
  fi
  return 1
}

# Attempt a simple psql connection using provided envs
check_postgres_conn() {
  if ! has_cmd psql; then
    warn "psql not found in PATH — cannot verify Postgres connectivity"
    return 2
  fi
  # Use PGPASSWORD to avoid interactive prompt
  PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q' >/dev/null 2>&1 && return 0 || return 1
}

# Verify conda activation availability
try_source_conda() {
  if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
    # conda executable exists
    return 0
  fi
  # Common locations
  for p in "$HOME/miniconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh" "/home/conda/etc/profile.d/conda.sh"; do
    if [ -f "$p" ]; then
      # shellcheck disable=SC1090
      . "$p"
      return 0
    fi
  done
  return 1
}

conda_env_exists() {
  if try_source_conda; then
    # Use `conda env list` and grep the env name
    if conda env list 2>/dev/null | awk '{print $1}' | grep -xq "$CONDA_ENV"; then
      return 0
    fi
  fi
  return 1
}

# Force-kill helper (tries TERM then KILL) for a given port (uses lsof/ss)
free_port_force() {
  local port="$1"
  info "Attempting to free port $port by terminating associated process(es)"
  local pids=""
  if has_cmd lsof; then
    pids=$(lsof -ti tcp:"$port" || true)
  fi
  if [ -z "$pids" ]; then
    pids=$(ss -ltnp 2>/dev/null | grep -E ":$port\b" | sed -n 's/.*pid=\([0-9]*\),.*/\1/p' | tr '\n' ' ')
  fi
  if [ -z "$pids" ]; then
    info "No PIDs found for port $port"
    return 1
  fi
  for pid in $pids; do
    info "Sending TERM to pid $pid (port $port)"
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 3
  if is_port_in_use "$port"; then
    info "Port $port still in use; sending SIGKILL to PIDs: $pids"
    for pid in $pids; do
      kill -9 "$pid" 2>/dev/null || true
    done
    sleep 1
  fi
  if is_port_in_use "$port"; then
    warn "Failed to free port $port"
    return 1
  fi
  info "Port $port freed"
  return 0
}

# Attempt graceful shutdown of all known JustNews services (but DO NOT stop Postgres)
shutdown_justnews_services() {
  info "Shutting down any running JustNews services (Postgres excluded)"

  # If systemd units are present and we are root, prefer systemctl stop per-agent units
  if has_cmd systemctl && [ "$EUID" -eq 0 ]; then
    info "Attempting systemd-based stop of justnews units (if installed)"
    for entry in "${AGENTS[@]}"; do
      name="${entry%%|*}"
      unit="justnews@${name}.service"
      if systemctl is-active --quiet "justnews@${name}" 2>/dev/null || systemctl list-unit-files | grep -q "^justnews@${name}\.service" 2>/dev/null; then
        info "Stopping systemd unit: $unit"
        systemctl stop "justnews@${name}" || true
        # Wait up to 15s for it to stop
        local deadline=$(( $(date +%s) + 15 ))
        while systemctl is-active --quiet "justnews@${name}" 2>/dev/null && [ $(date +%s) -le $deadline ]; do
          sleep 1
        done
        if systemctl is-active --quiet "justnews@${name}" 2>/dev/null; then
          warn "$unit did not stop cleanly within grace window"
        else
          info "$unit stopped"
        fi
      fi
    done
    # Short sleep to let systemd release ports
    sleep 2
  fi

  # For each expected port (excluding postgres 5432), try graceful shutdown endpoint then fall back to killing
  for port in "${REQUIRED_PORTS[@]}"; do
    if [ "$port" = "5432" ]; then
      info "Skipping Postgres port $port (leave running)"
      continue
    fi
    if is_port_in_use "$port"; then
      info "Service detected listening on port $port — attempting graceful shutdown via /shutdown"
      if has_cmd curl; then
        info "Attempting graceful shutdown on port $port via POST /shutdown"
        code=$(curl -s -o /dev/null -w "%{http_code}" -X POST --max-time 5 "http://localhost:$port/shutdown" || true)
        if [ "$code" = "200" ] || [ "$code" = "202" ] || [ "$code" = "204" ]; then
          info "Shutdown endpoint accepted POST on port $port (code $code). Waiting up to 20s for port to close..."
          deadline=$(( $(date +%s) + 20 ))
          while is_port_in_use "$port" && [ "$(date +%s)" -le "$deadline" ]; do
            sleep 1
          done
          if ! is_port_in_use "$port"; then
            info "Port $port freed by graceful shutdown"
            continue
          fi
        else
          info "Shutdown endpoint did not accept POST on port $port (code=$code)"
        fi
      else
        warn "curl not available to call shutdown endpoint on $port"
      fi
      # Fallback: try terminating the process(es) holding the port
      info "Attempting to terminate processes holding port $port"
      if free_port_force "$port"; then
        info "Successfully freed port $port"
      else
        warn "Could not free port $port"
      fi
    fi
  done
  info "Shutdown pass completed"
}

# Preflight for start: ensure ports are free and essential files/envs exist
preflight_for_start() {
  info "Running preflight checks for start"
  local failed=0
  if try_source_conda; then
    info "Conda runtime available"
    if conda_env_exists; then
      info "Conda env '$CONDA_ENV' exists"
    else
      warn "Conda env '$CONDA_ENV' not present"
      failed=1
    fi
  else
    warn "Conda not available"
    failed=1
  fi
  if check_postgres_conn; then
    info "Postgres connectivity OK"
  else
    warn "Postgres connectivity failed"
    failed=1
  fi
  for p in "${REQUIRED_PORTS[@]}"; do
    if is_port_in_use "$p"; then
      warn "Port $p still in use — cannot start until it is freed"
      failed=1
    else
      info "Port $p is free"
    fi
  done
  if [ $failed -eq 0 ]; then
    info "Preflight-for-start OK"
    return 0
  fi
  warn "Preflight-for-start detected issues"
  return 1
}

# Start services: decide systemd or dev script
start_services() {
  local mode="$1" # "auto" (default) | "systemd" | "dev"
  mode="${mode:-auto}"
  if [ "$mode" = "auto" ]; then
    # Prefer systemd path when running as root or reset script exists
    if [ "$EUID" -eq 0 ] && [ -x "$SCRIPT_DIR/deploy/systemd/reset_and_start.sh" ]; then
      mode=systemd
    elif [ -x "$SCRIPT_DIR/deploy/systemd/reset_and_start.sh" ] && [ "$FORCE_SYSTEMD" = "1" ]; then
      mode=systemd
    else
      mode=dev
    fi
  fi

  if [ "$mode" = "systemd" ]; then
    if [ -x "$SCRIPT_DIR/deploy/systemd/reset_and_start.sh" ]; then
      info "Starting production reset+start via deploy/systemd/reset_and_start.sh (requires sudo)"
      if [ "$EUID" -ne 0 ]; then
        info "Elevating to sudo to run reset_and_start.sh"
        sudo "$SCRIPT_DIR/deploy/systemd/reset_and_start.sh"
      else
        "$SCRIPT_DIR/deploy/systemd/reset_and_start.sh"
      fi
    else
      error "Systemd reset/start script missing: deploy/systemd/reset_and_start.sh"
      return 1
    fi
  else
    # dev path: start_services_daemon.sh
    if [ -x "$SCRIPT_DIR/start_services_daemon.sh" ]; then
      info "Starting development services using start_services_daemon.sh (detach mode)"
      # Prefer to run via conda run to pick correct env (the start script itself uses conda run already)
      "$SCRIPT_DIR/start_services_daemon.sh" || {
        error "start_services_daemon.sh failed — check logs in $LOG_DIR"; return 1
      }
    else
      error "Developer start script not found: start_services_daemon.sh"
      return 1
    fi
  fi
  return 0
}

# Check essential files referenced by start scripts
check_essential_files() {
  local missing=0
  local required=("start_services_daemon.sh" "deploy/systemd/reset_and_start.sh")
  for f in "${required[@]}"; do
    if [ ! -f "$SCRIPT_DIR/$f" ]; then
      warn "Required file missing: $f"
      missing=1
    fi
  done
  return $missing
}

# Preflight full checks
preflight() {
  info "Running preflight checks"
  local failed=0

  # If dry-run requested, print planned checks and exit success (non-destructive)
  if [ "${DRY_RUN:-0}" -eq 1 ]; then
    info "[dry-run] Preflight would perform the following checks:"
    info "  - Verify Conda availability and environment: $CONDA_ENV"
    info "  - Check Postgres connectivity to ${POSTGRES_USER}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
    info "  - Verify required agent ports: ${REQUIRED_PORTS[*]}"
    info "  - Confirm model store directory: $MODEL_STORE_ROOT"
    info "  - Ensure essential start scripts exist: start_services_daemon.sh and deploy/systemd/reset_and_start.sh"
    return 0
  fi

  # Conda
  if try_source_conda; then
    info "Conda runtime available"
    if conda_env_exists; then
      info "Conda env '$CONDA_ENV' exists"
    else
      warn "Conda env '$CONDA_ENV' not present. Use: conda env create -f environment.yml or conda create -n $CONDA_ENV"
      failed=1
    fi
  else
    warn "Conda appears not to be set up (conda not found). Many operations will fail without conda."
    failed=1
  fi

  # PSQL connectivity
  if check_postgres_conn; then
    info "Postgres connectivity OK (${POSTGRES_USER}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB})"
  else
    warn "Cannot connect to Postgres with provided envs (will be required for full system). Check $POSTGRES_HOST, $POSTGRES_PORT, credentials or install Postgres locally."
    failed=1
  fi

  # Ports
  local port_missing=0
  for p in "${REQUIRED_PORTS[@]}"; do
    if is_port_in_use "$p"; then
      info "Port $p already in use (expected when system already running)"
    else
      warn "Port $p not in use — service not started on that port"
      port_missing=1
    fi
  done
  [ $port_missing -eq 1 ] && failed=1

  # Models dir
  if [ -d "$MODEL_STORE_ROOT" ]; then
    info "Model store directory exists: $MODEL_STORE_ROOT"
  else
    warn "Model store directory missing: $MODEL_STORE_ROOT — models will be downloaded on first run or set STRICT_MODEL_STORE=0 to allow fallback."
    failed=1
  fi

  # Essential start files
  if check_essential_files; then
    warn "Some essential start files are missing — see warnings above"
    failed=1
  else
    info "Essential start scripts present"
  fi

  if [ $failed -eq 0 ]; then
    info "Preflight successful — environment looks ok"
    return 0
  fi
  warn "Preflight detected issues; review the warnings above"
  return 1
}

# Ensure the script is accessible globally by copying to /usr/local/bin
install_global() {
  local dest="/usr/local/bin/square-one"
  local opt_link="/opt/justnews"
  # Ensure /opt/justnews points at the project root so systemd units and installed wrapper resolve files correctly
  if [ -L "$opt_link" ] || [ -d "$opt_link" ]; then
    info "$opt_link already exists — leaving as-is"
  else
    info "Creating symlink $opt_link -> $SCRIPT_DIR (requires sudo)"
    if [ "$EUID" -ne 0 ]; then
      sudo mkdir -p /opt
      sudo ln -s "$SCRIPT_DIR" "$opt_link"
      sudo chown -h root:root "$opt_link" || true
    else
      mkdir -p /opt
      ln -s "$SCRIPT_DIR" "$opt_link"
      chown -h root:root "$opt_link" || true
    fi
  fi

  # Install a small wrapper in /usr/local/bin that delegates to /opt/justnews/square-one.sh
  if [ -f "$dest" ]; then
    info "A square-one installation already exists at $dest — backing it up to $dest.bak"
    if [ "$EUID" -ne 0 ]; then
      sudo cp "$dest" "$dest.bak"
    else
      cp "$dest" "$dest.bak"
    fi
  fi
  info "Installing wrapper $dest (requires sudo if not root)"
  wrapper_content="#!/usr/bin/env bash\nexec \"/opt/justnews/$SCRIPT_NAME\" \"\$@\"\n"
  if [ "$EUID" -ne 0 ]; then
    printf "%s" "$wrapper_content" | sudo tee "$dest" >/dev/null
    sudo chmod +x "$dest"
  else
    printf "%s" "$wrapper_content" >"$dest"
    chmod +x "$dest"
  fi
  info "Installed. You can now run 'square-one' from any directory. (/opt/justnews points to project root)"
}

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [command] [--yes] [--force-systemd] [--dry-run]
Commands:
  preflight       Run a preflight check for required envs/files/ports
  check-db        Test Postgres connection with current envs
  start [mode]    Start services; mode: auto|systemd|dev (default: auto)
  status          Show port and basic service status
  install         Install this script to /usr/local/bin/square-one (sudo may be required)
  help            Show this help
Flags:
  --yes           Non-interactive affirmative answer for install/start
  --force-systemd  Force systemd path for start when auto-detecting
  --dry-run       Print what would be done without executing actions
EOF
}

# Print a compact status
status() {
  info "Square-one status report"
  info "Model store: $MODEL_STORE_ROOT"
  info "Conda env: $CONDA_ENV"
  info "Postgres: ${POSTGRES_USER}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
  info "Ports:"
  for p in "${REQUIRED_PORTS[@]}"; do
    if is_port_in_use "$p"; then
      printf "  %-6s %s\n" "$p" "LISTEN"
    else
      printf "  %-6s %s\n" "$p" "DOWN"
    fi
  done
}

# Compute the start mode without performing actions
compute_start_mode() {
  local explicit_mode="$1"
  if [ -n "$explicit_mode" ]; then
    echo "$explicit_mode"
    return 0
  fi
  # Prefer systemd path when running as root and the reset script exists
  if [ "$EUID" -eq 0 ] && [ -x "$SCRIPT_DIR/deploy/systemd/reset_and_start.sh" ]; then
    echo "systemd"
    return 0
  fi
  # If force-systemd requested and script exists, prefer systemd
  if [ -x "$SCRIPT_DIR/deploy/systemd/reset_and_start.sh" ] && [ "$FORCE_SYSTEMD" = "1" ]; then
    echo "systemd"
    return 0
  fi
  echo "dev"
}

# Parse command args
CMD=""
ASSUME_YES=0
FORCE_SYSTEMD=0
DRY_RUN=0
START_MODE=""
if [ $# -gt 0 ]; then
  case "$1" in
    preflight|start|install|status|check-db|help)
      CMD="$1"; shift
      ;;
    *)
      usage; exit 1
      ;;
  esac
fi
if [ "$CMD" = "start" ] && [ $# -gt 0 ]; then
  case "$1" in
    auto|systemd|dev)
      START_MODE="$1"; shift
      ;;
  esac
fi
while [ $# -gt 0 ]; do
  case "$1" in
    --yes)
      ASSUME_YES=1; shift;;
    --force-systemd)
      FORCE_SYSTEMD=1; shift;;
    --dry-run)
      DRY_RUN=1; shift;;
    *)
      error "Unknown arg $1"; usage; exit 1;;
  esac
done

case "$CMD" in
  preflight)
    preflight || exit 1
    ;;
  check-db)
    if check_postgres_conn; then
      info "Postgres connection OK"
      exit 0
    else
      error "Postgres connection failed — check credentials and network"
      exit 2
    fi
    ;;
  status)
    status
    ;;
  install)
    if [ $ASSUME_YES -ne 1 ]; then
      read -r -p "Install $SCRIPT_NAME to /usr/local/bin/square-one? [y/N] " yn
      case "$yn" in
        [Yy]*) install_global ;;
        *) info "Aborted install"; exit 0 ;;
      esac
    else
      install_global
    fi
    ;;
  start)
    if [ $ASSUME_YES -ne 1 ]; then
      read -r -p "Start the full system now? [y/N] " yn
      case "$yn" in
        [Yy]*) ;;
        *) info "Aborted start"; exit 0;;
      esac
    fi

    # Before attempting to start, attempt to detect and shut down any running
    # JustNews services (excluding Postgres). This ensures a clean fresh start
    # environment. The shutdown function will try systemd stop units, call
    # /shutdown endpoints and will force-kill processes holding known ports.
    info "Initiating pre-start shutdown of any running JustNews services (Postgres will be preserved)"
    shutdown_justnews_services

    # After shutdown, validate that the environment is ready to start.
    # Use a start-focused preflight that expects ports to be free.
    if ! preflight_for_start; then
      if [ $ASSUME_YES -eq 1 ]; then
        warn "Preflight-for-start detected issues but continuing because --yes was provided"
      else
        warn "Preflight-for-start detected issues. Re-run with --yes to force start despite warnings, or fix issues and retry."
        exit 1
      fi
    fi

    # Dry-run: compute and report selected start mode without executing anything
    if [ "$DRY_RUN" -eq 1 ]; then
      if [ "${FORCE_SYSTEMD:-0}" -eq 1 ]; then
        mode_arg="systemd"
      else
        mode_arg=""
      fi
      selected_mode=$(compute_start_mode "$mode_arg")
      info "[dry-run] Would start in '$selected_mode' mode (no actions performed)"
      exit 0
    fi

    if [ -n "$START_MODE" ]; then
      start_services "$START_MODE"
    elif [ "$FORCE_SYSTEMD" = "1" ]; then
      start_services systemd
    else
      start_services
    fi
    ;;
  help|"")
    usage ;;
  *)
    usage; exit 1 ;;
esac

exit 0
