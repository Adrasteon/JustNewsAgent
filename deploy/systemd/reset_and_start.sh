#!/bin/bash
# reset_and_start.sh — Canonical reset/start script for JustNews (systemd)
#
# Performs a safe, end-to-end reset:
# - Stops and disables services
# - Frees occupied ports
# - Optionally reinstalls unit template and helper scripts
# - Optionally syncs env files to /etc/justnews
# - Optionally toggles SAFE_MODE in global.env
# - Reloads systemd and fresh-starts all services
# - Runs a health check summary at the end

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Defaults
REINSTALL=false
REINSTALL_UNITS=false
REINSTALL_SCRIPTS=false
SYNC_ENV=false
FORCE_SYNC_ENV=false
SAFE_MODE_STATE=""   # "on" | "off" | ""
CLEAN_PORTS=true
HEALTH_CHECK=true
DRY_RUN=false

PORTS=(8000 8001 8002 8003 8004 8005 8006 8007 8008 8009 8011 8012 8013 8014 8015 8016)
UNIT_TEMPLATE_SRC="$PROJECT_ROOT/deploy/systemd/units/justnews@.service"
UNIT_TEMPLATE_DST="/etc/systemd/system/justnews@.service"
START_SCRIPT_SRC="$PROJECT_ROOT/deploy/systemd/justnews-start-agent.sh"
START_SCRIPT_DST="/usr/local/bin/justnews-start-agent.sh"
WAIT_SCRIPT_SRC="$PROJECT_ROOT/deploy/systemd/scripts/wait_for_mcp.sh"
WAIT_SCRIPT_DST="/usr/local/bin/wait_for_mcp.sh"
ENV_SRC_DIR="$PROJECT_ROOT/deploy/systemd/env"
ENV_DST_DIR="/etc/justnews"
GLOBAL_ENV="$ENV_DST_DIR/global.env"
WRAPPER_MAP=(
  "/usr/local/bin/enable_all.sh|$PROJECT_ROOT/deploy/systemd/scripts/enable_all.sh"
  "/usr/local/bin/health_check.sh|$PROJECT_ROOT/deploy/systemd/scripts/health_check.sh"
  "/usr/local/bin/reset_and_start.sh|$PROJECT_ROOT/deploy/systemd/scripts/reset_and_start.sh"
  "/usr/local/bin/cold_start.sh|$PROJECT_ROOT/deploy/systemd/scripts/cold_start.sh"
)

# New: allow operator to pass an explicit python path
SET_PYTHON_BIN=""

# New CLI-configurable defaults (can be overridden)
JUSTNEWS_GROUP="justnews"
DEFAULT_SERVICE_USER="justnews"
ADMIN_GROUP="justnews-admins"
ADMIN_USERS=()

require_root() {
  if [[ $EUID -ne 0 ]]; then
    log_error "Must run as root (sudo)"
    exit 1
  fi
}

check_tools() {
  local req=(systemctl ss cp sed grep awk)
  for t in "${req[@]}"; do
    if ! command -v "$t" &>/dev/null; then
      log_error "Required tool missing: $t"; exit 1
    fi
  done
}

usage() {
  cat << EOF
Usage: $0 [options]

Options:
  --reinstall            Reinstall unit template and helper scripts
  --reinstall-units      Reinstall only systemd unit template
  --reinstall-scripts    Reinstall only helper scripts
  --sync-env             Copy env files to /etc/justnews (no overwrite)
  --force-sync-env       Copy env files and overwrite existing (with .bak)
  --safe-mode on|off     Toggle SAFE_MODE in /etc/justnews/global.env
  --no-clean-ports       Do not kill residual port listeners
  --no-health-check      Skip final health check
  --dry-run              Show actions without executing system changes
  --set-python-bin PATH  Force PYTHON_BIN to PATH when creating /etc/justnews/global.env
  --justnews-group       Set the JustNews system group (default: justnews)
  --service-user         Set the service user for JustNews (default: adra)
  --admin-user USER      Add a system user as admin (repeatable for multiple users)
  -h, --help             Show this help

Examples:
  sudo $0 --reinstall --sync-env --safe-mode on
  sudo $0 --force-sync-env --safe-mode off
  sudo $0  # default: clean ports + fresh start + health check
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --reinstall) REINSTALL=true; shift ;;
      --reinstall-units) REINSTALL_UNITS=true; shift ;;
      --reinstall-scripts) REINSTALL_SCRIPTS=true; shift ;;
      --sync-env) SYNC_ENV=true; shift ;;
      --force-sync-env) SYNC_ENV=true; FORCE_SYNC_ENV=true; shift ;;
      --safe-mode)
        SAFE_MODE_STATE="${2:-}"; shift 2 || true
        if [[ "$SAFE_MODE_STATE" != "on" && "$SAFE_MODE_STATE" != "off" ]]; then
          log_error "--safe-mode requires 'on' or 'off'"; exit 1
        fi
        ;;
      --set-python-bin)
        SET_PYTHON_BIN="${2:-}"; shift 2 || true
        if [[ -z "$SET_PYTHON_BIN" ]]; then
          log_error "--set-python-bin requires an absolute path to a Python interpreter"; exit 1
        fi
        ;;
      --justnews-group)
        JUSTNEWS_GROUP="${2:-}"; shift 2 || true
        if [[ -z "$JUSTNEWS_GROUP" ]]; then
          log_error "--justnews-group requires a group name"; exit 1
        fi
        ;;
      --admin-user)
        ADMIN_USERS+=("${2:-}"); shift 2 || true
        ;; 
      --service-user)
        DEFAULT_SERVICE_USER="${2:-}"; shift 2 || true
        if [[ -z "$DEFAULT_SERVICE_USER" ]]; then
          log_error "--service-user requires a username"; exit 1
        fi
        ;;
      --no-clean-ports) CLEAN_PORTS=false; shift ;;
      --no-health-check) HEALTH_CHECK=false; shift ;;
      --dry-run) DRY_RUN=true; shift ;;
      -h|--help) usage; exit 0 ;;
      *) log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
  done

  if [[ "$REINSTALL" == true ]]; then
    REINSTALL_UNITS=true; REINSTALL_SCRIPTS=true
  fi
}

backup_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    cp -a "$f" "${f}.bak_${ts}"
    log_info "Backed up $(basename "$f") -> ${f}.bak_${ts}"
  fi
}

stop_disable_services() {
  log_info "Stopping and disabling all JustNews services..."
  if [[ "$DRY_RUN" == true ]]; then
    echo "DRY-RUN: $SCRIPT_DIR/enable_all.sh stop/disable"; return 0
  fi
  "$SCRIPT_DIR/enable_all.sh" stop || true
  "$SCRIPT_DIR/enable_all.sh" disable || true
}

kill_ports() {
  [[ "$CLEAN_PORTS" == true ]] || { log_info "Skipping port cleanup"; return 0; }
  log_info "Cleaning residual listeners on ports: ${PORTS[*]}"
  for p in "${PORTS[@]}"; do
    local lines
    lines=$(ss -ltnp "sport = :$p" 2>/dev/null | tail -n +2 || true)
    if [[ -n "$lines" ]]; then
      log_warn "Port $p in use; attempting to kill owning processes"
      # Extract PIDs from 'users:("proc",pid=123,fd=...)'
      local pids
      pids=$(echo "$lines" | grep -oP 'pid=\K[0-9]+' | sort -u)
      if [[ -n "$pids" ]]; then
        for pid in $pids; do
          if [[ "$DRY_RUN" == true ]]; then
            echo "DRY-RUN: kill -9 $pid (port $p)"
          else
            kill -9 "$pid" 2>/dev/null || true
            log_info "Killed PID $pid (port $p)"
          fi
        done
      fi
    else
      log_success "Port $p is free"
    fi
  done
}

ensure_group_exists() {
  local g="$JUSTNEWS_GROUP"
  if getent group "$g" >/dev/null 2>&1; then
    log_info "Group $g already exists"
    return 0
  fi
  if [[ "$DRY_RUN" == true ]]; then
    echo "DRY-RUN: groupadd $g"; return 0
  fi
  groupadd "$g" || true
  log_info "Created group: $g"
}

ensure_admin_group_exists() {
  if getent group "$ADMIN_GROUP" >/dev/null 2>&1; then
    log_info "Admin group $ADMIN_GROUP already exists"
    return 0
  fi
  if [[ "$DRY_RUN" == true ]]; then
    echo "DRY-RUN: groupadd $ADMIN_GROUP"; return 0
  fi
  groupadd "$ADMIN_GROUP" || true
  log_info "Created admin group: $ADMIN_GROUP"
}

create_system_user() {
  local u="$DEFAULT_SERVICE_USER"
  if id -u "$u" >/dev/null 2>&1; then
    log_info "Service user $u already exists"
    return 0
  fi
  if [[ "$DRY_RUN" == true ]]; then
    echo "DRY-RUN: useradd --system --no-create-home --shell /usr/sbin/nologin -g $JUSTNEWS_GROUP $u"
    return 0
  fi
  useradd --system --no-create-home --shell /usr/sbin/nologin -g "$JUSTNEWS_GROUP" "$u" || true
  log_info "Created system user: $u"
}

reinstall_units_scripts() {
  # Ensure the target group exists before attempting to chown helper scripts
  ensure_group_exists

  if [[ "$REINSTALL_UNITS" == true ]]; then
    log_info "Reinstalling systemd unit template..."
    if [[ "$DRY_RUN" == true ]]; then
      echo "DRY-RUN: cp $UNIT_TEMPLATE_SRC $UNIT_TEMPLATE_DST && systemctl daemon-reload"
    else
      mkdir -p "$(dirname "$UNIT_TEMPLATE_DST")"
      backup_file "$UNIT_TEMPLATE_DST"
      cp "$UNIT_TEMPLATE_SRC" "$UNIT_TEMPLATE_DST"
      # Ensure unit template is world-readable but owned by root
      chown root:root "$UNIT_TEMPLATE_DST" || true
      chmod 0644 "$UNIT_TEMPLATE_DST" || true
      systemctl daemon-reload
      log_success "Unit template installed"
    fi
  fi

  if [[ "$REINSTALL_SCRIPTS" == true ]]; then
    log_info "Reinstalling helper scripts..."
    if [[ "$DRY_RUN" == true ]]; then
      echo "DRY-RUN: cp $START_SCRIPT_SRC $START_SCRIPT_DST; cp $WAIT_SCRIPT_SRC $WAIT_SCRIPT_DST"
    else
      # Make source helper scripts executable so systemd ExecStartPre can call the repo copy if configured
      if [[ -d "$PROJECT_ROOT/deploy/systemd/scripts" ]]; then
        chmod +x "$PROJECT_ROOT/deploy/systemd/scripts/"*.sh 2>/dev/null || true
        log_info "Ensured repo helper scripts are executable"
      fi
      backup_file "$START_SCRIPT_DST" || true
      cp "$START_SCRIPT_SRC" "$START_SCRIPT_DST"
      chmod +x "$START_SCRIPT_DST"
      # Secure helper script ownership and permissions
      chown root:"$JUSTNEWS_GROUP" "$START_SCRIPT_DST" || true
      chmod 0750 "$START_SCRIPT_DST" || true
      backup_file "$WAIT_SCRIPT_DST" || true
      cp "$WAIT_SCRIPT_SRC" "$WAIT_SCRIPT_DST"
      chmod +x "$WAIT_SCRIPT_DST"
      chown root:"$JUSTNEWS_GROUP" "$WAIT_SCRIPT_DST" || true
      chmod 0750 "$WAIT_SCRIPT_DST" || true
      log_success "Helper scripts installed"
    fi
  fi
}

ensure_path_wrappers() {
  # Always ensure operator PATH wrappers exist (idempotent)
  for pair in "${WRAPPER_MAP[@]}"; do
    IFS='|' read -r dst src <<<"$pair"
    if [[ -f "$src" ]] && [[ ! -x "$dst" ]]; then
      cp "$src" "$dst" && chmod +x "$dst"
      log_info "Installed PATH wrapper $(basename "$dst")"
      if [[ "$DRY_RUN" == true ]]; then
        echo "DRY-RUN: chown root:$JUSTNEWS_GROUP $dst; chmod 0750 $dst"
      else
        chown root:"$JUSTNEWS_GROUP" "$dst" || true
        chmod 0750 "$dst" || true
        log_info "Set owner root:$JUSTNEWS_GROUP and secure perms on $(basename "$dst")"
      fi
    fi
  done
}

sync_env_files() {
  [[ "$SYNC_ENV" == true ]] || return 0
  log_info "Syncing env files to $ENV_DST_DIR (force: $FORCE_SYNC_ENV)"
  if [[ "$DRY_RUN" == true ]]; then
    echo "DRY-RUN: mkdir -p $ENV_DST_DIR; copy env files"
    return 0
  fi
  mkdir -p "$ENV_DST_DIR"
  # Handle case where no env files exist in the source directory
  files=("$ENV_SRC_DIR"/*.env)
  if [[ ! -e "${files[0]}" ]]; then
    log_warn "No env files found in $ENV_SRC_DIR; skipping env sync"
    return 0
  fi
  for f in "$ENV_SRC_DIR"/*.env; do
    local base
    base="$(basename "$f")"
    local dst="$ENV_DST_DIR/$base"
    if [[ -f "$dst" && "$FORCE_SYNC_ENV" == true ]]; then
      backup_file "$dst"
      cp "$f" "$dst"
      log_info "Overwrote $dst (backup created)"
    elif [[ -f "$dst" && "$FORCE_SYNC_ENV" == false ]]; then
      log_warn "$dst exists; skipping (use --force-sync-env to overwrite)"
    else
      cp "$f" "$dst"; log_info "Copied $dst"
    fi
  done
}

toggle_safe_mode() {
  [[ -n "$SAFE_MODE_STATE" ]] || return 0
  if [[ ! -f "$GLOBAL_ENV" ]]; then
    log_warn "$GLOBAL_ENV not found; creating new"
    mkdir -p "$ENV_DST_DIR"
    echo "# Auto-created by reset_and_start.sh" > "$GLOBAL_ENV"
  fi
  if [[ "$DRY_RUN" == true ]]; then
    echo "DRY-RUN: Toggle SAFE_MODE=$SAFE_MODE_STATE in $GLOBAL_ENV"
    return 0
  fi
  backup_file "$GLOBAL_ENV"
  if grep -q '^SAFE_MODE=' "$GLOBAL_ENV"; then
    sed -i "s/^SAFE_MODE=.*/SAFE_MODE=$([[ "$SAFE_MODE_STATE" == on ]] && echo true || echo false)/" "$GLOBAL_ENV"
  else
    echo "SAFE_MODE=$([[ "$SAFE_MODE_STATE" == on ]] && echo true || echo false)" >> "$GLOBAL_ENV"
  fi
  log_success "SAFE_MODE set to $SAFE_MODE_STATE"
}

daemon_reload() {
  if [[ "$DRY_RUN" == true ]]; then
    echo "DRY-RUN: systemctl daemon-reload"; return 0
  fi
  systemctl daemon-reload
}

fresh_start() {
  log_info "Starting all services fresh..."
  if [[ "$DRY_RUN" == true ]]; then
    echo "DRY-RUN: $SCRIPT_DIR/enable_all.sh fresh"; return 0
  fi
  "$SCRIPT_DIR/enable_all.sh" fresh
}

health_check() {
  [[ "$HEALTH_CHECK" == true ]] || return 0
  if [[ "$DRY_RUN" == true ]]; then
    echo "DRY-RUN: $SCRIPT_DIR/health_check.sh"; return 0
  fi
  if [[ -x "$SCRIPT_DIR/health_check.sh" ]]; then
    "$SCRIPT_DIR/health_check.sh" || log_warn "Health check reported issues"
  else
    log_warn "health_check.sh not found/executable"
  fi
}

# Detect a likely Python interpreter from common conda env locations or use explicit override
detect_python_bin() {
  # Priority: explicit CLI arg, /home/*/miniconda3 env named justnews-v2-py312, /opt/conda, CONDA_PREFIX
  if [[ -n "$SET_PYTHON_BIN" ]]; then
    DETECTED_PYTHON_BIN="$SET_PYTHON_BIN"
    log_info "Using explicit Python provided via --set-python-bin: $DETECTED_PYTHON_BIN"
    return 0
  fi

  # Check for CONDA_PREFIX in environment
  if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
    DETECTED_PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    log_info "Detected Python via CONDA_PREFIX: $DETECTED_PYTHON_BIN"
    return 0
  fi

  # Common user locations (search for named env used by CI/dev workflows)
  local candidates=(/home/*/miniconda3/envs/justnews-v2-py312/bin/python /home/*/micromamba/envs/justnews-v2-py312/bin/python /opt/conda/envs/justnews-v2-py312/bin/python)
  for c in "${candidates[@]}"; do
    if [[ -x $c ]]; then
      DETECTED_PYTHON_BIN="$c"
      log_info "Auto-detected Python at: $DETECTED_PYTHON_BIN"
      return 0
    fi
  done

  # Fallback: try to locate any python in envs folder that looks like 'justnews'
  for base in /home/*/miniconda3/envs /home/*/micromamba/envs /opt/conda/envs; do
    if [[ -d $base ]]; then
      for d in "$base"/*justnews*; do
        if [[ -x "$d/bin/python" ]]; then
          DETECTED_PYTHON_BIN="$d/bin/python"
          log_info "Found candidate Python at: $DETECTED_PYTHON_BIN"
          return 0
        fi
      done
    fi
  done

  # Give up — leave empty to allow manual configuration
  log_warn "No conda/python interpreter detected automatically. You can supply one with --set-python-bin or edit $GLOBAL_ENV"
  DETECTED_PYTHON_BIN=""
}

# Inject or update PYTHON_BIN in the global env file so services run with the correct interpreter
inject_python_bin_into_global_env() {
  if [[ -z "$DETECTED_PYTHON_BIN" ]]; then
    # If forcing overwrite but nothing detected, ensure an example placeholder exists
    if [[ ! -f "$GLOBAL_ENV" ]]; then
      echo "# Auto-created by reset_and_start.sh" > "$GLOBAL_ENV"
    fi
    if ! grep -q '^PYTHON_BIN=' "$GLOBAL_ENV" 2>/dev/null; then
      echo "#PYTHON_BIN=/path/to/python (set with --set-python-bin)" >> "$GLOBAL_ENV"
      log_info "Wrote PYTHON_BIN placeholder to $GLOBAL_ENV"
    else
      log_info "PYTHON_BIN already present in $GLOBAL_ENV; no changes made"
    fi
    return 0
  fi

  # Ensure file exists
  if [[ ! -f "$GLOBAL_ENV" ]]; then
    echo "# Auto-created by reset_and_start.sh" > "$GLOBAL_ENV"
  fi

  backup_file "$GLOBAL_ENV"
  if grep -q '^PYTHON_BIN=' "$GLOBAL_ENV" 2>/dev/null; then
    sed -i "s|^PYTHON_BIN=.*|PYTHON_BIN=$DETECTED_PYTHON_BIN|" "$GLOBAL_ENV"
    log_info "Updated PYTHON_BIN in $GLOBAL_ENV -> $DETECTED_PYTHON_BIN"
  else
    echo "PYTHON_BIN=$DETECTED_PYTHON_BIN" >> "$GLOBAL_ENV"
    log_info "Inserted PYTHON_BIN into $GLOBAL_ENV -> $DETECTED_PYTHON_BIN"
  fi
}

# Create a dedicated group for service files and ensure proper permissions
ensure_justnews_group_and_perms() {
  local g="$JUSTNEWS_GROUP"
  local svc_user=""
  # Ensure group exists
  if ! getent group "$g" >/dev/null; then
    if [[ "$DRY_RUN" == true ]]; then
      echo "DRY-RUN: groupadd $g"
    else
      groupadd "$g" || true
      log_info "Created group: $g"
    fi
  else
    log_info "Group $g already exists"
  fi

  # Discover service user from installed unit template (if available)
  if [[ -f "$UNIT_TEMPLATE_DST" ]]; then
    svc_user=$(grep -E '^User=' "$UNIT_TEMPLATE_DST" | head -n1 | cut -d= -f2 || true)
  fi
  svc_user=${svc_user:-"$DEFAULT_SERVICE_USER"}

  # Ensure the system user exists and is a member of the group
  create_system_user
  if id -u "$svc_user" >/dev/null 2>&1; then
    if [[ "$DRY_RUN" == true ]]; then
      echo "DRY-RUN: usermod -aG $g $svc_user"
    else
      usermod -aG "$g" "$svc_user" || true
      log_info "Added user $svc_user to group $g"
    fi
  else
    log_warn "Service user '$svc_user' not found even after creation attempt; skipping group membership step"
  fi

  # Create and handle admin group + users
  ensure_admin_group_exists
  for au in "${ADMIN_USERS[@]}"; do
    if id -u "$au" >/dev/null 2>&1; then
      if [[ "$DRY_RUN" == true ]]; then
        echo "DRY-RUN: usermod -aG $ADMIN_GROUP $au"
        echo "DRY-RUN: usermod -aG $g $au"
      else
        usermod -aG "$ADMIN_GROUP" "$au" || true
        usermod -aG "$g" "$au" || true
        log_info "Added admin user $au to groups: $ADMIN_GROUP, $g"
      fi
    else
      log_warn "Admin user '$au' not found on system; skipping addition"
    fi
  done

  # Apply secure permissions to env dir and files
  if [[ "$DRY_RUN" == true ]]; then
    echo "DRY-RUN: chown -R root:$g $ENV_DST_DIR; chmod 0750 $ENV_DST_DIR; chmod 0640 $ENV_DST_DIR/*.env"
  else
    chown -R root:"$g" "$ENV_DST_DIR" || true
    chmod 0750 "$ENV_DST_DIR" || true
    for f in "$ENV_DST_DIR"/*.env; do
      if [[ -f "$f" ]]; then
        chmod 0640 "$f" || true
      fi
    done
    log_success "Set ownership root:$g and restrictive permissions for $ENV_DST_DIR"
  fi
}

# Defaults and constants for new functions
JUSTNEWS_GROUP="justnews"
DEFAULT_SERVICE_USER="adra"
DETECTED_PYTHON_BIN=""

main() {
  parse_args "$@"
  require_root
  check_tools

  echo "========================================"
  log_info "JustNews Canonical Reset & Start"
  echo "========================================"

  stop_disable_services
  kill_ports
  reinstall_units_scripts
  ensure_path_wrappers
  sync_env_files
  # Ensure dedicated system user and admin groups exist and are configured
  create_system_user
  ensure_admin_group_exists
  inject_python_bin_into_global_env
  ensure_justnews_group_and_perms
  toggle_safe_mode
  daemon_reload
  fresh_start
  health_check

  log_success "Reset & start sequence completed"
}

main "$@"
