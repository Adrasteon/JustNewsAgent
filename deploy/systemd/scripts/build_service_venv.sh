#!/bin/bash
# build_service_venv.sh - Deterministic venv builder for JustNews runtime
# Creates a lightweight Python virtual environment with only runtime dependencies
# for fast CI validation and production deployment.
#
# Usage:
#   ./build_service_venv.sh [target_dir] [requirements_file]
#
# Defaults:
#   target_dir: ./venv
#   requirements_file: ./requirements-runtime.txt

set -euo pipefail

# Configuration
TARGET_DIR="${1:-./venv}"
REQUIREMENTS_FILE="${2:-./requirements-runtime.txt}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if Python is available
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    log_error "Python interpreter not found: $PYTHON_BIN"
    log_error "Set PYTHON_BIN environment variable or ensure python3 is in PATH"
    exit 1
fi

# Verify Python version
PYTHON_VERSION=$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
log_info "Using Python $PYTHON_VERSION from $PYTHON_BIN"

# Check minimum version (3.10+)
if "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    log_success "Python version meets minimum requirement (3.10+)"
else
    log_error "Python version $PYTHON_VERSION is below minimum requirement (3.10+)"
    exit 1
fi

# Check if requirements file exists
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    log_error "Requirements file not found: $REQUIREMENTS_FILE"
    exit 1
fi

# Check Python command availability
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    log_error "Python command not found: $PYTHON_CMD"
    exit 1
fi

# Display Python version
PY_VERSION=$("$PYTHON_CMD" --version 2>&1)
log_info "Using Python: $PY_VERSION"

# Remove existing venv if present
if [[ -d "$VENV_PATH" ]]; then
    log_warn "Removing existing venv: $VENV_PATH"
    rm -rf "$VENV_PATH"
fi

# Create new venv
log_info "Creating virtual environment at: $VENV_PATH"
"$PYTHON_CMD" -m venv "$VENV_PATH"

# Activate venv (for verification only; actual activation happens in caller's shell)
VENV_PYTHON="$VENV_PATH/bin/python"
VENV_PIP="$VENV_PATH/bin/pip"

if [[ ! -f "$VENV_PYTHON" ]]; then
    log_error "Failed to create venv: $VENV_PYTHON not found"
    exit 1
fi

# Upgrade pip, setuptools, and wheel for deterministic installs
log_info "Upgrading pip, setuptools, and wheel..."
"$VENV_PIP" install --upgrade pip setuptools wheel

# Install requirements
log_info "Installing requirements from: $REQUIREMENTS_FILE"
"$VENV_PIP" install -r "$REQUIREMENTS_FILE"

# Verification step
if [[ "$VERIFY" == true ]]; then
    log_info "Running post-install verification..."
    
    # Check core runtime modules
    MODULES=("fastapi" "uvicorn" "pydantic" "requests")
    MISSING=()
    
    for module in "${MODULES[@]}"; do
        if ! "$VENV_PYTHON" -c "import $module" 2>/dev/null; then
            MISSING+=("$module")
        fi
    done
    
    if [[ ${#MISSING[@]} -gt 0 ]]; then
        log_error "Verification failed: missing modules: ${MISSING[*]}"
        exit 1
    fi
    
    log_info "Verification passed: all core runtime modules present"
    
    # Display installed packages summary
    log_info "Installed packages summary:"
    "$VENV_PIP" list --format=columns | head -15
fi

log_info "✓ Virtual environment built successfully: $VENV_PATH"
log_info "Activate with: source $VENV_PATH/bin/activate"
