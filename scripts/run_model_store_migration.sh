#!/usr/bin/env bash
# Run the model store population script in the justnews-v2-prod conda env
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export MODEL_STORE_ROOT="${MODEL_STORE_ROOT:-/media/adra/Data/justnews/model_store}"
export MODEL_STORE_VERSION="${MODEL_STORE_VERSION:-v1}"

echo "Populating MODEL_STORE_ROOT=$MODEL_STORE_ROOT version=$MODEL_STORE_VERSION"
conda run --name justnews-v2-prod python "$PROJECT_ROOT/scripts/populate_model_store.py"
