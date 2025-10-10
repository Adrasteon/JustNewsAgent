#!/bin/bash
# build_service_venv.sh - Build a reproducible virtualenv for JustNews services
# Creates an isolated Python virtual environment with production dependencies
#
# Usage:
#   ./build_service_venv.sh --venv /opt/justnews/venv --requirements release_beta_minimal_preview/requirements-runtime.txt
#
# Options:
#   --venv PATH           Path where the virtualenv should be created (required)
#   --requirements FILE   Path to requirements file to install (required)
#   --python PYTHON       Python interpreter to use (default: python3)
#   --force               Remove existing venv if present before creating
#   --no-upgrade-pip      Skip pip upgrade step
#   -h, --help            Show this help message

set -euo pipefail

# Default configuration
VENV_PATH=""
REQUIREMENTS_FILE=""
PYTHON_BIN="python3"
FORCE_RECREATE=false
UPGRADE_PIP=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 --venv VENV_PATH --requirements REQUIREMENTS_FILE [OPTIONS]

Build a reproducible virtualenv for JustNews services.

Required arguments:
  --venv PATH           Path where the virtualenv should be created
  --requirements FILE   Path to requirements file to install

Optional arguments:
  --python PYTHON       Python interpreter to use (default: python3)
  --force               Remove existing venv if present before creating
  --no-upgrade-pip      Skip pip upgrade step
  -h, --help            Show this help message

Examples:
  # Create a minimal runtime venv
  $0 --venv /opt/justnews/venv --requirements release_beta_minimal_preview/requirements-runtime.txt

  # Force recreate with specific Python version
  $0 --venv /tmp/test-venv --requirements requirements-runtime.txt --python python3.12 --force
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --venv)
                VENV_PATH="$2"
                shift 2
                ;;
            --requirements)
                REQUIREMENTS_FILE="$2"
                shift 2
                ;;
            --python)
                PYTHON_BIN="$2"
                shift 2
                ;;
            --force)
                FORCE_RECREATE=true
                shift
                ;;
            --no-upgrade-pip)
                UPGRADE_PIP=false
                shift
                ;;
            -h|--help)
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

    if [[ -z "$REQUIREMENTS_FILE" ]]; then
        log_error "Missing required argument: --requirements"
        show_usage
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Python interpreter
    if ! command -v "$PYTHON_BIN" &> /dev/null; then
        log_error "Python interpreter not found: $PYTHON_BIN"
        exit 1
    fi

    local python_version
    python_version=$("$PYTHON_BIN" --version 2>&1)
    log_success "✓ Found $python_version"

    # Check venv module
    if ! "$PYTHON_BIN" -m venv --help &> /dev/null; then
        log_error "Python venv module not available"
        log_error "Install it with: apt-get install python3-venv"
        exit 1
    fi
    log_success "✓ Python venv module available"

    # Check requirements file
    if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
        log_error "Requirements file not found: $REQUIREMENTS_FILE"
        exit 1
    fi
    log_success "✓ Requirements file found: $REQUIREMENTS_FILE"
}

# Create or recreate virtualenv
create_venv() {
    log_info "Creating virtualenv at $VENV_PATH..."

    # Handle existing venv
    if [[ -d "$VENV_PATH" ]]; then
        if [[ "$FORCE_RECREATE" == "true" ]]; then
            log_warning "Removing existing venv at $VENV_PATH"
            rm -rf "$VENV_PATH"
        else
            log_error "Virtualenv already exists at $VENV_PATH"
            log_error "Use --force to remove and recreate"
            exit 1
        fi
    fi

    # Create parent directory if needed
    local parent_dir
    parent_dir="$(dirname "$VENV_PATH")"
    if [[ ! -d "$parent_dir" ]]; then
        log_info "Creating parent directory: $parent_dir"
        mkdir -p "$parent_dir"
    fi

    # Create the virtualenv
    "$PYTHON_BIN" -m venv "$VENV_PATH"
    log_success "✓ Virtualenv created"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies from $REQUIREMENTS_FILE..."

    # Activate venv (in subshell)
    local pip_bin="$VENV_PATH/bin/pip"
    local python_bin="$VENV_PATH/bin/python"

    # Verify venv was created properly
    if [[ ! -x "$pip_bin" ]]; then
        log_error "Virtualenv pip not found at $pip_bin"
        exit 1
    fi

    # Upgrade pip if requested
    if [[ "$UPGRADE_PIP" == "true" ]]; then
        log_info "Upgrading pip..."
        "$python_bin" -m pip install --upgrade pip setuptools wheel
        log_success "✓ pip upgraded"
    fi

    # Install requirements
    log_info "Installing packages..."
    "$pip_bin" install -r "$REQUIREMENTS_FILE"
    log_success "✓ Dependencies installed"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    local python_bin="$VENV_PATH/bin/python"

    # Check critical packages
    local critical_packages=("uvicorn" "requests" "fastapi")
    local missing_packages=()

    for package in "${critical_packages[@]}"; do
        if "$python_bin" -c "import $package" 2>/dev/null; then
            log_success "✓ $package is importable"
        else
            log_error "✗ $package is NOT importable"
            missing_packages+=("$package")
        fi
    done

    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        log_error "Failed to verify installation. Missing packages: ${missing_packages[*]}"
        return 1
    fi

    log_success "✓ All critical packages verified"
    return 0
}

# Print summary
print_summary() {
    echo ""
    echo "========================================"
    log_success "Virtualenv build complete!"
    echo "========================================"
    echo ""
    log_info "Virtualenv location: $VENV_PATH"
    log_info "Requirements file:   $REQUIREMENTS_FILE"
    echo ""
    log_info "To activate the virtualenv, run:"
    echo "  source $VENV_PATH/bin/activate"
    echo ""
    log_info "To use with systemd services, set:"
    echo "  Environment=PYTHON_BIN=$VENV_PATH/bin/python"
    echo ""
}

# Main function
main() {
    echo "========================================"
    log_info "JustNews Service Virtualenv Builder"
    echo "========================================"
    echo ""

    parse_args "$@"
    check_prerequisites
    create_venv
    install_dependencies
    verify_installation
    print_summary
}

# Run main function
main "$@"
