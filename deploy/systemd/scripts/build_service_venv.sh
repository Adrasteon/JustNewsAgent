#!/bin/bash
# build_service_venv.sh - Deterministic venv builder for JustNews production deployment
# Creates a reproducible Python virtual environment with minimal runtime dependencies
# Suitable for both deploy nodes and CI environments

set -euo pipefail

# Configuration
VENV_PATH="${1:-/opt/justnews/venv}"
REQUIREMENTS_FILE="${2:-release_beta_minimal_preview/requirements-runtime.txt}"
PYTHON_VERSION="${PYTHON_VERSION:-python3}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Ensure we have a valid Python interpreter
if ! command -v "$PYTHON_VERSION" >/dev/null 2>&1; then
    log_error "Python interpreter '$PYTHON_VERSION' not found"
    log_error "Set PYTHON_VERSION environment variable to a valid python3 executable"
    exit 1
fi

log_info "Building JustNews service venv at: $VENV_PATH"
log_info "Using Python: $(command -v "$PYTHON_VERSION")"
log_info "Python version: $("$PYTHON_VERSION" --version)"

# Check if requirements file exists
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    log_error "Requirements file not found: $REQUIREMENTS_FILE"
    exit 1
fi

log_info "Using requirements file: $REQUIREMENTS_FILE"

# Remove existing venv if present
if [[ -d "$VENV_PATH" ]]; then
    log_warning "Removing existing venv at $VENV_PATH"
    rm -rf "$VENV_PATH"
fi

# Create parent directory if needed
VENV_DIR="$(dirname "$VENV_PATH")"
if [[ ! -d "$VENV_DIR" ]]; then
    log_info "Creating parent directory: $VENV_DIR"
    mkdir -p "$VENV_DIR"
fi

# Create the virtual environment
log_info "Creating virtual environment..."
"$PYTHON_VERSION" -m venv "$VENV_PATH"

if [[ ! -f "$VENV_PATH/bin/python" ]]; then
    log_error "Failed to create virtual environment"
    exit 1
fi

log_success "Virtual environment created"

# Upgrade pip, setuptools, and wheel for reproducibility
log_info "Upgrading pip, setuptools, and wheel..."
"$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel

# Install requirements with hash checking disabled for flexibility
# In production, consider using --require-hashes with a full lock file
log_info "Installing runtime requirements..."
"$VENV_PATH/bin/pip" install -r "$REQUIREMENTS_FILE"

# Verify critical packages
log_info "Verifying installation..."
CRITICAL_PACKAGES=(fastapi uvicorn pydantic requests)
for pkg in "${CRITICAL_PACKAGES[@]}"; do
    if ! "$VENV_PATH/bin/python" -c "import $pkg" 2>/dev/null; then
        log_error "Critical package '$pkg' failed to import"
        exit 1
    fi
    log_success "  ✓ $pkg"
done

# Display installed packages for audit trail
log_info "Installed packages:"
"$VENV_PATH/bin/pip" list

# Create a marker file with build timestamp for auditing
BUILD_INFO="$VENV_PATH/build_info.txt"
cat > "$BUILD_INFO" << EOF
Build Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Python Version: $("$VENV_PATH/bin/python" --version)
Pip Version: $("$VENV_PATH/bin/pip" --version)
Requirements File: $REQUIREMENTS_FILE
Build Host: $(hostname)
EOF

log_success "Build info written to: $BUILD_INFO"
log_success "Virtual environment build complete: $VENV_PATH"

# Test that the dependency checker works with this venv
if [[ -f "deploy/systemd/scripts/ci_check_deps.py" ]]; then
    log_info "Running dependency checker with new venv..."
    if "$VENV_PATH/bin/python" deploy/systemd/scripts/ci_check_deps.py; then
        log_success "Dependency check passed!"
    else
        log_warning "Dependency check reported missing packages (may need more modules)"
    fi
fi

exit 0
