#!/bin/bash
# build_service_venv.sh - Deterministic venv builder for JustNews production deployment
# Creates a reproducible Python virtual environment with minimal runtime dependencies
# Suitable for both deploy nodes and CI environments

set -euo pipefail

# Show usage if requested
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat << 'EOF'
JustNews Service Virtual Environment Builder

USAGE:
    build_service_venv.sh [VENV_PATH] [REQUIREMENTS_FILE]

ARGUMENTS:
    VENV_PATH           Path to create the virtual environment
                        (default: /opt/justnews/venv)
    REQUIREMENTS_FILE   Requirements file to install
                        (default: release_beta_minimal_preview/requirements-runtime.txt)

ENVIRONMENT VARIABLES:
    PYTHON_VERSION      Python interpreter to use (default: python3)
    REQUIRE_HASHES      Enable hash verification (1 to enable, 0 to disable)
                        Requires hashes in requirements file for security

EXAMPLES:
    # Build default production venv
    sudo ./build_service_venv.sh

    # Build custom venv for CI
    ./build_service_venv.sh ~/test-venv release_beta_minimal_preview/requirements-runtime.txt

    # Use specific Python version
    PYTHON_VERSION=python3.12 ./build_service_venv.sh /tmp/my-venv

    # Enable hash verification for production
    REQUIRE_HASHES=1 ./build_service_venv.sh /opt/justnews/venv requirements-locked.txt

EXIT CODES:
    0 - Success
    1 - Error (missing Python, requirements file, or installation failure)
EOF
    exit 0
fi

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

# Validate Python version is 3.x
if ! PY_VERSION=$("$PYTHON_VERSION" -c "import sys; print(sys.version_info.major)" 2>/dev/null); then
    log_error "Failed to determine Python version"
    exit 1
fi
if [[ "$PY_VERSION" != "3" ]]; then
    log_error "Python 3.x is required, but found Python $PY_VERSION"
    log_error "Set PYTHON_VERSION to a Python 3.x interpreter (e.g., python3, python3.12)"
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

# Install requirements with optional hash checking for security
# Set REQUIRE_HASHES=1 to enable hash verification (requires hashes in requirements file)
log_info "Installing runtime requirements..."
if [[ "${REQUIRE_HASHES:-0}" == "1" ]]; then
    log_info "Hash verification enabled (REQUIRE_HASHES=1)"
    "$VENV_PATH/bin/pip" install --require-hashes -r "$REQUIREMENTS_FILE"
else
    log_info "Hash verification disabled (set REQUIRE_HASHES=1 to enable)"
    "$VENV_PATH/bin/pip" install -r "$REQUIREMENTS_FILE"
fi

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
        log_warning "Dependency check reported missing packages"
        log_warning "The runtime requirements may need additional modules beyond the minimal set"
        log_warning "Check the output above for specific packages to add to $REQUIREMENTS_FILE"
        if [[ -f "environment.yml" ]]; then
            log_warning "Full dependencies are defined in environment.yml for development environments"
        fi
    fi
fi

exit 0
