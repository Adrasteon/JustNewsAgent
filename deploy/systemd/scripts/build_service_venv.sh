#!/bin/bash
# build_service_venv.sh - Deterministic venv builder for JustNews services
# 
# Purpose: Create a clean, reproducible Python virtual environment for production
# services or CI validation. Supports both runtime-only and full installations.
#
# Usage:
#   ./build_service_venv.sh --venv /path/to/venv --requirements path/to/requirements.txt
#   ./build_service_venv.sh --venv .venv --requirements release_beta_minimal_preview/requirements-runtime.txt
#   ./build_service_venv.sh --venv /opt/justnews/venv --requirements requirements.txt --upgrade
#
# Options:
#   --venv PATH              Path to virtual environment (required)
#   --requirements PATH      Path to requirements file (required)
#   --upgrade               Upgrade pip before installing requirements
#   --python PYTHON_CMD      Python interpreter to use (default: python3)
#   --force                 Remove existing venv and recreate
#   --help                  Show this help message
#
# Exit codes:
#   0 - Success
#   1 - Error (missing arguments, installation failure, etc.)

set -euo pipefail

# Default configuration
VENV_PATH=""
REQUIREMENTS_PATH=""
PYTHON_CMD="python3"
UPGRADE_PIP=false
FORCE_RECREATE=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Create a deterministic Python virtual environment for JustNews services.

Required Options:
  --venv PATH              Path to virtual environment
  --requirements PATH      Path to requirements file

Optional:
  --python PYTHON_CMD      Python interpreter (default: python3)
  --upgrade               Upgrade pip before installing requirements
  --force                 Remove and recreate existing venv
  --help                  Show this help message

Examples:
  # Quick runtime venv for CI:
  $(basename "$0") --venv .venv --requirements release_beta_minimal_preview/requirements-runtime.txt

  # Production venv with full dependencies:
  $(basename "$0") --venv /opt/justnews/venv --requirements requirements.txt --upgrade

  # Force recreate with specific Python:
  $(basename "$0") --venv .venv --requirements requirements.txt --python python3.11 --force

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --requirements)
            REQUIREMENTS_PATH="$2"
            shift 2
            ;;
        --python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        --upgrade)
            UPGRADE_PIP=true
            shift
            ;;
        --force)
            FORCE_RECREATE=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$VENV_PATH" ]]; then
    log_error "Missing required argument: --venv"
    show_usage
    exit 1
fi

if [[ -z "$REQUIREMENTS_PATH" ]]; then
    log_error "Missing required argument: --requirements"
    show_usage
    exit 1
fi

# Validate Python interpreter
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    log_error "Python interpreter not found: $PYTHON_CMD"
    exit 1
fi

# Validate requirements file exists
if [[ ! -f "$REQUIREMENTS_PATH" ]]; then
    log_error "Requirements file not found: $REQUIREMENTS_PATH"
    exit 1
fi

# Convert to absolute paths
VENV_PATH="$(cd "$(dirname "$VENV_PATH")" 2>/dev/null && pwd)/$(basename "$VENV_PATH")" || {
    # If parent doesn't exist, just use the path as-is
    VENV_PATH="$(realpath "$VENV_PATH" 2>/dev/null || echo "$VENV_PATH")"
}
REQUIREMENTS_PATH="$(realpath "$REQUIREMENTS_PATH")"

log_info "=========================================="
log_info "JustNews Virtual Environment Builder"
log_info "=========================================="
log_info "Python: $PYTHON_CMD ($(command -v "$PYTHON_CMD"))"
log_info "Venv path: $VENV_PATH"
log_info "Requirements: $REQUIREMENTS_PATH"
log_info ""

# Check if venv already exists
if [[ -d "$VENV_PATH" ]]; then
    if [[ "$FORCE_RECREATE" == "true" ]]; then
        log_warning "Removing existing venv: $VENV_PATH"
        rm -rf "$VENV_PATH"
    else
        log_warning "Virtual environment already exists: $VENV_PATH"
        log_warning "Use --force to recreate or install directly with pip"
        exit 1
    fi
fi

# Create virtual environment
log_info "Creating virtual environment..."
if ! "$PYTHON_CMD" -m venv "$VENV_PATH"; then
    log_error "Failed to create virtual environment"
    exit 1
fi
log_success "Virtual environment created"

# Activate venv for remaining commands
VENV_PYTHON="$VENV_PATH/bin/python"
VENV_PIP="$VENV_PATH/bin/pip"

if [[ ! -x "$VENV_PYTHON" ]]; then
    log_error "Virtual environment Python not found or not executable: $VENV_PYTHON"
    exit 1
fi

# Upgrade pip if requested
if [[ "$UPGRADE_PIP" == "true" ]]; then
    log_info "Upgrading pip..."
    if ! "$VENV_PYTHON" -m pip install --upgrade pip; then
        log_error "Failed to upgrade pip"
        exit 1
    fi
    log_success "Pip upgraded"
fi

# Install requirements
log_info "Installing requirements from: $REQUIREMENTS_PATH"
if ! "$VENV_PIP" install -r "$REQUIREMENTS_PATH"; then
    log_error "Failed to install requirements"
    exit 1
fi
log_success "Requirements installed"

# Verify installation
log_info "Verifying installation..."
VENV_PACKAGES=$("$VENV_PIP" list 2>/dev/null | wc -l)
log_success "Installed $VENV_PACKAGES packages"

# Show key packages for runtime verification
log_info ""
log_info "Key runtime packages:"
for pkg in fastapi uvicorn pydantic requests prometheus-client; do
    if "$VENV_PIP" show "$pkg" >/dev/null 2>&1; then
        version=$("$VENV_PIP" show "$pkg" 2>/dev/null | grep "^Version:" | awk '{print $2}')
        echo "  ✓ $pkg ($version)"
    fi
done

log_info ""
log_success "=========================================="
log_success "Virtual environment ready!"
log_success "=========================================="
log_info "Activate with: source $VENV_PATH/bin/activate"
log_info "Or use directly: $VENV_PYTHON"
log_info ""

exit 0
