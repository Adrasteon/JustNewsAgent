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

log_info "Building venv at: $TARGET_DIR"
log_info "Using requirements: $REQUIREMENTS_FILE"

# Remove existing venv if present
if [[ -d "$TARGET_DIR" ]]; then
    log_warning "Removing existing venv at $TARGET_DIR"
    rm -rf "$TARGET_DIR"
fi

# Create virtual environment
log_info "Creating virtual environment..."
"$PYTHON_BIN" -m venv "$TARGET_DIR"

if [[ ! -d "$TARGET_DIR" ]] || [[ ! -f "$TARGET_DIR/bin/activate" ]]; then
    log_error "Failed to create virtual environment"
    exit 1
fi

log_success "Virtual environment created"

# Upgrade pip to avoid warnings
log_info "Upgrading pip..."
"$TARGET_DIR/bin/pip" install --quiet --upgrade pip setuptools wheel

# Install requirements
log_info "Installing runtime dependencies from $REQUIREMENTS_FILE..."
"$TARGET_DIR/bin/pip" install --quiet -r "$REQUIREMENTS_FILE"

# Verify installation
log_info "Verifying installation..."
INSTALLED_COUNT=$("$TARGET_DIR/bin/pip" list --format=freeze | wc -l)
log_success "Installed $INSTALLED_COUNT packages"

# Show venv info
log_info "Virtual environment info:"
echo "  Location: $TARGET_DIR"
echo "  Python: $("$TARGET_DIR/bin/python" --version)"
echo "  Pip: $("$TARGET_DIR/bin/pip" --version | awk '{print $1, $2}')"

log_success "Virtual environment build complete"
log_info "Activate with: source $TARGET_DIR/bin/activate"

exit 0
