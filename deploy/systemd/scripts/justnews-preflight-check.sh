#!/bin/bash
# justnews-preflight-check.sh - Pre-startup validation for JustNews services
set -euo pipefail

# Default configuration
GATE_ONLY=false
MCP_BUS_URL="${MCP_BUS_URL:-http://127.0.0.1:8000}"
GPU_ORCHESTRATOR_URL="${GPU_ORCHESTRATOR_URL:-http://127.0.0.1:8014}"
TIMEOUT=300 # 5 minutes for model preloading

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

# Parse arguments
for arg in "$@"; do
    case $arg in
        --gate-only)
        GATE_ONLY=true
        shift
        ;;
    esac
done

echo "========================================"
log_info "JustNews Preflight Check"
echo "========================================"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    log_error "This script should be run as root (sudo) for full validation"
    log_warning "Continuing with limited checks..."
fi

# Gate-only mode: for MCP Bus startup
if [ "$GATE_ONLY" = true ]; then
    log_info "Gate-only mode: ensuring gpu_orchestrator is up and models are ready"
    
    # 1. Wait for GPU Orchestrator to be healthy
    log_info "Waiting for GPU Orchestrator at $GPU_ORCHESTRATOR_URL..."
    start_time=$(date +%s)
    while true; do
        if curl -fsS "${GPU_ORCHESTRATOR_URL}/health" > /dev/null; then
            log_success "GPU Orchestrator is healthy."
            break
        fi
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        if [ $elapsed -ge $TIMEOUT ]; then
            log_error "Timeout waiting for GPU Orchestrator."
            exit 1
        fi
        sleep 5
    done

    # 2. Trigger and wait for model preloading
    log_info "Triggering model preload via GPU Orchestrator..."
    preload_response=$(curl -s -X POST "${GPU_ORCHESTRATOR_URL}/models/preload" -H "Content-Type: application/json" -d '{"refresh": false}')
    
    if [[ $(echo "$preload_response" | jq -r '.all_ready') == "true" ]]; then
        log_success "Models were already preloaded."
        exit 0
    fi

    log_info "Waiting for model preloading to complete..."
    start_time=$(date +%s)
    while true; do
        status_response=$(curl -s "${GPU_ORCHESTRATOR_URL}/models/status")
        in_progress=$(echo "$status_response" | jq -r '.in_progress')
        
        if [[ "$in_progress" == "false" ]]; then
            failed_count=$(echo "$status_response" | jq -r '.summary.failed')
            if [[ $failed_count -gt 0 ]]; then
                log_error "Model preloading failed for $failed_count models."
                log_error "Details: $(echo "$status_response" | jq -c '.errors')"
                exit 1
            else
                log_success "All models preloaded successfully."
                break
            fi
        fi

        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        if [ $elapsed -ge $TIMEOUT ]; then
            log_error "Timeout waiting for model preloading."
            exit 1
        fi
        log_info "Preloading in progress... ($(echo "$status_response" | jq -r '.summary.done')/$(echo "$status_response" | jq -r '.summary.total'))"
        sleep 10
    done
    
    log_success "Preflight check (gate-only) passed."
    exit 0
fi

# Full preflight check (for other agents)
log_info "Full preflight mode: ensuring MCP Bus is ready"
log_info "Waiting for MCP Bus at $MCP_BUS_URL..."
start_time=$(date +%s)
while true; do
    if curl -fsS "${MCP_BUS_URL}/health" > /dev/null; then
        log_success "MCP Bus is healthy."
        break
    fi
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -ge 60 ]; then
        log_error "Timeout waiting for MCP Bus."
        exit 1
    fi
    sleep 3
done

# Full preflight check (for other agents)
log_info "Full preflight mode: ensuring MCP Bus is ready"
log_info "Waiting for MCP Bus at $MCP_BUS_URL..."
start_time=$(date +%s)
while true; do
    if curl -fsS "${MCP_BUS_URL}/health" > /dev/null; then
        log_success "MCP Bus is healthy."
        break
    fi
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -ge 60 ]; then
        log_error "Timeout waiting for MCP Bus."
        exit 1
    fi
    sleep 3
done

log_success "All preflight checks passed."
exit 0
