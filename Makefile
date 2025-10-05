.PHONY: help env-create env-install env-report test-dev test-unit test-smoke test-tensorrt test-ci check-env

# Default target
help:
	@echo "JustNewsAgent Makefile - Developer Convenience Targets"
	@echo ""
	@echo "Environment Management:"
	@echo "  make env-create     - Create conda environment from environment.yml"
	@echo "  make env-install    - Install dependencies in active environment"
	@echo "  make env-report     - Report Python and package versions"
	@echo "  make check-env      - Verify environment is activated"
	@echo ""
	@echo "Testing:"
	@echo "  make test-dev       - Run all safe tests (unit + smoke + tensorrt stub)"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-smoke     - Run smoke E2E stub test"
	@echo "  make test-tensorrt  - Run TensorRT stub build test"
	@echo "  make test-ci        - Run the canonical CI test sequence"
	@echo ""
	@echo "Note: Most targets require an activated conda environment."
	@echo "      Use: conda activate justnews-v2-py312 (or mamba)"

# Environment Management
env-create:
	@echo "Creating conda environment justnews-v2-py312 from environment.yml..."
	@if command -v mamba >/dev/null 2>&1; then \
		echo "Using mamba (faster)..."; \
		mamba env create -f environment.yml; \
	else \
		echo "Using conda..."; \
		conda env create -f environment.yml; \
	fi
	@echo ""
	@echo "Environment created. Activate with: conda activate justnews-v2-py312"

env-install:
	@echo "Installing/updating dependencies in active environment..."
	pip install -U pip setuptools wheel
	@echo "Dependencies installed."

env-report:
	@echo "=== Environment Report ==="
	@echo "Python version:"
	@python --version
	@echo ""
	@echo "Key package versions:"
	@python -c "import sys; print(f'Python: {sys.version}')" 2>/dev/null || true
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: not installed"
	@python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || echo "Transformers: not installed"
	@python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')" 2>/dev/null || echo "FastAPI: not installed"
	@python -c "import pytest; print(f'Pytest: {pytest.__version__}')" 2>/dev/null || echo "Pytest: not installed"
	@echo ""
	@echo "CUDA availability:"
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not available to check CUDA"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo ""; \
		echo "GPU info:"; \
		nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed"; \
	else \
		echo "nvidia-smi not found (no GPU or drivers not installed)"; \
	fi

check-env:
	@if [ -z "$$CONDA_DEFAULT_ENV" ]; then \
		echo "ERROR: No conda environment is active."; \
		echo "Please activate: conda activate justnews-v2-py312"; \
		exit 1; \
	else \
		echo "Active conda environment: $$CONDA_DEFAULT_ENV"; \
	fi

# Testing Targets
test-unit: check-env
	@echo "Running unit tests (excludes integration tests)..."
	pytest -v -k "not integration" --tb=short --maxfail=5

test-smoke: check-env
	@echo "Running smoke E2E stub test..."
	python tests/smoke_e2e_stub.py

test-tensorrt: check-env
	@echo "Running TensorRT stub build test..."
	pytest -v tests/test_tensorrt_stub.py

test-dev: check-env test-unit test-smoke test-tensorrt
	@echo ""
	@echo "=== All safe tests completed ==="

test-ci: check-env
	@echo "Running canonical CI test sequence..."
	@echo ""
	$(MAKE) env-report
	@echo ""
	$(MAKE) test-unit
	@echo ""
	$(MAKE) test-smoke
	@echo ""
	@echo "=== Canonical CI test sequence completed ==="
