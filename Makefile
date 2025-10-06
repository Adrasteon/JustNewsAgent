# Makefile - developer convenience targets for JustNewsAgent
# Prefers mamba when available, falls back to conda.

# Environment configuration
ENV_NAME ?= justnews-v2-py312
CONDA ?= conda

# Detect mamba binary if installed
MAMBA := $(shell command -v mamba 2>/dev/null || true)
PKG_MGR := $(if $(MAMBA),mamba,conda)

.DEFAULT_GOAL := help

.PHONY: help env-create env-install env-rapids env-report test-dev test-unit \
  test-smoke test-tensorrt test-ci test-py-override check-env check-install

help:
	@echo "JustNewsAgent Makefile - common developer tasks"
	@echo
	@echo "Targets:"
	@echo "  env-create        Create the development conda env (mamba preferred)"
	@echo "  env-install       Install utility/test deps into the env (mamba preferred)"
	@echo "  env-rapids        Install RAPIDS / GPU packages (verify CUDA compatibility)"
	@echo "  env-report        Print environment and package versions for debugging/reproducibility"
	@echo "  test-dev          Run canonical tests inside the conda env (all steps)"
	@echo "  test-unit         Run unit tests only (runner --unit)"
	@echo "  test-smoke        Run smoke tests (runner --smoke)"
	@echo "  test-tensorrt     Run TensorRT gated tests (runner --tensorrt)"
	@echo "  test-py-override  Run canonical runner with explicit PY override (useful for CI)"
	@echo "  test-ci           CI-friendly wrapper (uses PY override and runs all tests)"
	@echo "  check-env         Print detected package manager and env status"
	@echo "  check-install     Run local installation checks (unit template & env examples)"

env-create:
	@echo "Using package manager: $(PKG_MGR)"
	@if command -v mamba >/dev/null 2>&1; then \
		echo "Mamba detected — using mamba for faster environment creation"; \
		mamba create -n $(ENV_NAME) python=3.12 -c conda-forge -y; \
	else \
		echo "Mamba not detected — falling back to conda"; \
		conda create -n $(ENV_NAME) python=3.12 -c conda-forge -y; \
	fi
	@echo "Created environment '$(ENV_NAME)'. Activate with: conda activate $(ENV_NAME)"

env-install:
	@echo "Installing test/util dependencies into $(ENV_NAME)"
	@if command -v mamba >/dev/null 2>&1; then \
		mamba install -n $(ENV_NAME) -c conda-forge prometheus_client gputil pytest -y; \
	else \
		conda install -n $(ENV_NAME) -c conda-forge prometheus_client gputil pytest -y; \
	fi

env-rapids:
	@echo "Installing RAPIDS example - verify GPU & CUDA compatibility first!"
	@echo "Using channels: rapidsai, conda-forge, nvidia"
	@if command -v mamba >/dev/null 2>&1; then \
		mamba install -n $(ENV_NAME) -c rapidsai -c conda-forge -c nvidia rapids=25.04 python=3.12 cuda-version=12.4 -y; \
	else \
		conda install -n $(ENV_NAME) -c rapidsai -c conda-forge -c nvidia rapids=25.04 python=3.12 cuda-version=12.4 -y; \
	fi

env-report:
	@echo "Detected package manager: $(PKG_MGR)"
	@echo "Conda version: $(shell $(CONDA) --version 2>/dev/null || echo 'conda: not found')"
	@if command -v mamba >/dev/null 2>&1; then \
		echo "Mamba version:" $(shell mamba --version 2>/dev/null || echo 'mamba: not found'); \
	fi
	@echo "Conda environments:" \
		; $(CONDA) info --envs || true
	@if $(CONDA) env list | grep -q "$(ENV_NAME)"; then \
		echo "Environment '$(ENV_NAME)' found - conda list:"; \
		$(CONDA) list -n $(ENV_NAME) --no-builds || true; \
		echo "Probing Python and common GPU packages inside $(ENV_NAME):"; \
		conda run -n $(ENV_NAME) python - <<'PY' || true
import sys, importlib
print('python:', sys.version.replace('\n',' '))
for pkg in ('torch','cudf','cuml','cugraph','cupy'):
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, '__version__', None)
        print(f"{pkg}: {ver}")
    except Exception as e:
        print(f"{pkg}: not available ({e.__class__.__name__})")
PY
	else \
		echo "Environment '$(ENV_NAME)' not found; run 'make env-create' first"; \
	fi

test-dev:
	@echo "Running canonical tests in env $(ENV_NAME)"
	conda run -n $(ENV_NAME) ./Canonical_Test_RUNME.sh --all

test-unit:
	@echo "Running unit tests in env $(ENV_NAME)"
	conda run -n $(ENV_NAME) ./Canonical_Test_RUNME.sh --unit

test-smoke:
	@echo "Running smoke tests in env $(ENV_NAME)"
	conda run -n $(ENV_NAME) ./Canonical_Test_RUNME.sh --smoke

test-tensorrt:
	@echo "Running TensorRT gated tests in env $(ENV_NAME)"
	conda run -n $(ENV_NAME) ./Canonical_Test_RUNME.sh --tensorrt

test-py-override:
	@echo "Running canonical runner with explicit PY override using env $(ENV_NAME)"
	PY='conda run -n $(ENV_NAME) python' ./Canonical_Test_RUNME.sh --all

ci-tests:
	@echo "Run tests locally in conda env $(ENV_NAME)"
	conda run -n $(ENV_NAME) bash scripts/apply_executable_permissions.sh
	conda run -n $(ENV_NAME) pytest -q

# CI-friendly non-interactive test target. Useful for CI pipelines.
test-ci:
	@echo "CI test run (non-interactive) using env $(ENV_NAME)"
	PY='conda run -n $(ENV_NAME) python' ./Canonical_Test_RUNME.sh --all

check-env:
	@echo "Detected package manager: $(PKG_MGR)"
	@echo "Conda environments (showing only names):"
	@conda env list | awk '{print $$1}' | sed -n '1,200p' || true
	@if conda env list | grep -q "$(ENV_NAME)"; then \
		echo "Environment $(ENV_NAME) exists"; \
	else \
		echo "Environment $(ENV_NAME) not found"; \
	fi

check-install:
	@echo "Running local installation checks (unit template & env examples)"
	@bash scripts/ci/check_unit_and_env.sh
