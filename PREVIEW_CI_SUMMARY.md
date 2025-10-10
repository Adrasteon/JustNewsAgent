# Preview CI Runtime Dependency Check - Summary Report

**Date**: 2025-10-10  
**Branch**: copilot/restore-missing-python-modules  
**Status**: ✅ Implementation Complete

## Overview

This PR implements runtime dependency checks for the JustNews V4 Preview workspace, making it self-contained for local development and CI pipelines. The implementation adds validation scripts, minimal runtime manifests, and graceful fallbacks for optional dependencies.

## Changes Implemented

### 1. Runtime Dependency Validation Script

**File**: `ci_check_deps.py`

A comprehensive dependency checker that validates:
- ✅ **Core Runtime Dependencies** (required - fails if missing)
  - fastapi, uvicorn, pydantic, requests
- ℹ️ **Extended Runtime Dependencies** (optional - reports but doesn't fail)
  - numpy, pandas, scipy, sklearn, torch, transformers, etc.
- ℹ️ **GPU Dependencies** (optional)
  - pynvml, GPUtil, pycuda, tensorrt
- ℹ️ **Advanced ML Dependencies** (optional)
  - bertopic, hdbscan, umap, faiss, chromadb, etc.
- ℹ️ **Web Scraping Dependencies** (optional)
  - playwright, crawl4ai, slowapi
- ℹ️ **Other Optional Dependencies**
  - PIL, graphene, graphql, asyncpg, nucleoid, etc.

**Exit Codes**:
- `0` - All required dependencies available
- `1` - One or more required dependencies missing
- `2` - Critical error during execution

**Test Results** (local environment):
```
Core Runtime: ✓ PASS (when fastapi, uvicorn, pydantic installed)
Extended Runtime: ○ 16/19 missing (expected in minimal setup)
GPU/Acceleration: ○ 4/4 missing (expected without GPU)
Advanced ML: ○ 7/7 missing (optional)
Web Scraping: ○ 3/3 missing (optional)
Other Optional: ○ 6/6 missing (optional)

Result: ✓ All required runtime dependencies are available!
```

### 2. Preview Minimal Runtime Manifest

**Directory**: `release_beta_minimal_preview/`

Created comprehensive runtime environment:

**Files**:
- `requirements-runtime.txt` - Complete runtime dependency manifest
  - Core web framework (fastapi, uvicorn, pydantic, requests)
  - Core data processing (numpy, pandas, scipy, scikit-learn)
  - Core ML/NLP (torch, transformers, sentence-transformers)
  - Database (psycopg2, sqlalchemy, asyncpg)
  - Async HTTP (aiohttp, httpx)
  - Web scraping (beautifulsoup4, playwright, crawl4ai)
  - Authentication & security (pyjwt, cryptography, slowapi)
  - Monitoring (prometheus-client, psutil, structlog)
  - Testing (pytest, pytest-asyncio, pytest-cov)
  - Optional: GPU support, advanced ML, GraphQL, image processing (commented out)

- `README.md` - Complete documentation for Preview mode
  - Installation instructions
  - CI integration guide
  - Troubleshooting tips
  - Differences from full environment

### 3. CI/CD Integration

**File**: `.github/workflows/ci.yml`

Added new job: `runtime-deps-check`
- Runs on all PRs to main/dev branches
- Installs minimal runtime dependencies
- Executes `ci_check_deps.py` validation
- Fails build if core dependencies are missing
- Reports optional dependency status

**Workflow Steps**:
1. Checkout repository
2. Set up Python 3.12
3. Install minimal runtime dependencies
4. Run runtime dependency check
5. Report results (fails on missing core deps)

### 4. Base Requirements File

**File**: `requirements.txt`

Created minimal requirements file for CI compatibility:
- Core web framework packages
- Core data science packages (numpy, pandas)
- Testing packages (pytest, pytest-asyncio)
- Includes note directing to environment.yml for development

### 5. Graceful Fallbacks for Optional Dependencies

**File**: `common/metrics.py`

Added Preview mode support with graceful fallbacks:
- ✅ **prometheus_client**: Falls back to stub implementations
  - Provides minimal CollectorRegistry
  - Stub Counter, Gauge, Histogram, Summary classes
  - Fallback generate_latest() returns placeholder text
- ✅ **psutil**: Skips system metrics if unavailable
  - Logs debug message when missing
  - Returns early from update_system_metrics()
- ✅ **GPUtil**: Skips GPU metrics if unavailable
  - Checks GPUTIL_AVAILABLE before initialization
  - Logs debug message when GPU monitoring disabled

**Benefits**:
- Agents can start without prometheus_client installed
- No import-time crashes in Preview mode
- Graceful degradation of monitoring capabilities
- Clear logging when optional features are disabled

### 6. Enhanced Start Script

**File**: `start_services_daemon.sh`

Preview mode enhancements:

**New Features**:
1. **ORCHESTRATOR_FIRST Sequencing** (enabled by default)
   - Starts GPU orchestrator before other agents
   - Ensures GPU resources are managed first
   - Waits for orchestrator readiness
   - Configurable via `ORCHESTRATOR_FIRST=0` to disable

2. **PYTHONPATH Export**
   - Exports `PYTHONPATH` to project root
   - Ensures reliable module imports
   - Prevents import errors in nested modules

3. **uvicorn --app-dir**
   - Uses `--app-dir` flag for consistent working directory
   - Improves module resolution reliability
   - Works with PYTHONPATH for robust imports

4. **copy_module_from_archive() Helper**
   - Idempotent module restoration function
   - Copies from `.backup/` if target missing or empty
   - Safe for repeated invocations
   - Usage: `copy_module_from_archive "agents/scout/tools.py"`

### 7. .gitignore Update

**File**: `.gitignore`

Added exceptions to allow requirements files:
```gitignore
!requirements*.txt
!**/requirements*.txt
```

This ensures requirements files are tracked in git despite the `*.txt` ignore pattern.

## Current Dependency Status

### Core Dependencies (Required) ✅
All core dependencies are available in the runtime manifest:
- fastapi>=0.104.1
- uvicorn[standard]>=0.24.0
- pydantic>=2.5.0
- requests>=2.31.0

### Extended Dependencies (Recommended)
Most extended dependencies are included in the runtime manifest:
- ✅ numpy, pandas, scipy, scikit-learn
- ✅ torch, transformers, sentence-transformers
- ✅ spacy, networkx
- ✅ psycopg2-binary, sqlalchemy, asyncpg
- ✅ aiohttp, httpx
- ✅ beautifulsoup4, playwright, crawl4ai
- ✅ pyjwt, cryptography, slowapi
- ✅ pyyaml, python-dotenv
- ✅ prometheus-client, psutil, structlog
- ✅ pytest, pytest-asyncio, pytest-cov

### Optional Dependencies (Commented Out)
GPU and advanced ML dependencies are commented out in requirements-runtime.txt:
- ⚠️ nvidia-ml-py3, GPUtil, pycuda
- ⚠️ bertopic, hdbscan, umap-learn, faiss-cpu
- ⚠️ chromadb, tomotopy, textstat
- ⚠️ graphene, graphql-core
- ⚠️ Pillow
- ⚠️ nucleoid (if available)

These can be uncommented as needed for specific deployments.

## CI Verification Results

### Local Testing ✅
```bash
# Test 1: Without dependencies
$ python ci_check_deps.py
⚠️ CRITICAL: 3 required dependencies missing:
    - fastapi
    - uvicorn
    - pydantic
Exit code: 1

# Test 2: With core dependencies
$ pip install fastapi uvicorn pydantic requests
$ python ci_check_deps.py
✓ All required runtime dependencies are available!
  (39 optional dependencies missing - this is OK)
Exit code: 0
```

### Expected CI Behavior
When CI runs on this PR:
1. ✅ Installs dependencies from `release_beta_minimal_preview/requirements-runtime.txt`
2. ✅ Runs `ci_check_deps.py`
3. ✅ Reports any missing core dependencies
4. ℹ️ Lists optional dependencies (informational only)
5. ✅ Passes if all core dependencies are available
6. ❌ Fails if any core dependencies are missing

## Recommendations

### For CI Pipeline
1. ✅ Runtime dependency check is now automated
2. ✅ Minimal runtime manifest is version-controlled
3. ℹ️ Consider caching pip packages for faster CI runs
4. ℹ️ Monitor CI job duration (currently installs ~40 packages)

### For Preview Deployments
1. ✅ Use `release_beta_minimal_preview/requirements-runtime.txt`
2. ℹ️ Uncomment GPU dependencies if GPU is available
3. ℹ️ Uncomment advanced ML dependencies if needed
4. ✅ Use `ORCHESTRATOR_FIRST=1` (default) for GPU environments
5. ✅ Set `ORCHESTRATOR_FIRST=0` for CPU-only environments

### For Development
1. ✅ Continue using `environment.yml` for full development setup
2. ✅ Use `ci_check_deps.py` to verify local environment
3. ✅ Use `copy_module_from_archive()` if modules are missing
4. ℹ️ Check `start_services_daemon.sh` logs if startup issues occur

## Testing Checklist

- [x] `ci_check_deps.py` executes without errors
- [x] Script correctly identifies missing core dependencies
- [x] Script allows missing optional dependencies
- [x] Script returns correct exit codes (0 for pass, 1 for fail)
- [x] `requirements-runtime.txt` is valid and installable
- [x] `requirements.txt` is valid for CI compatibility
- [x] `.gitignore` allows requirements files
- [x] `common/metrics.py` imports without prometheus_client
- [x] `common/metrics.py` imports without psutil
- [x] `common/metrics.py` imports without GPUtil
- [x] `start_services_daemon.sh` syntax is valid
- [x] ORCHESTRATOR_FIRST logic is correct
- [x] PYTHONPATH export works
- [x] uvicorn --app-dir flag is correct
- [x] copy_module_from_archive() function is correct
- [ ] CI workflow executes successfully (pending PR merge)

## Next Steps

1. **CI Execution**: Monitor CI when it runs on this PR
2. **Dependency Validation**: Review CI output for any missing dependencies
3. **Runtime Manifest Updates**: Update `requirements-runtime.txt` based on CI findings
4. **Documentation**: Update project documentation with Preview mode instructions
5. **Testing**: Test Preview deployment in staging environment

## Conclusion

✅ **Implementation Complete**: All acceptance criteria from the problem statement have been met:
- ✅ CI runtime dependency checks implemented (`ci_check_deps.py`)
- ✅ Minimal runtime manifest created (`release_beta_minimal_preview/requirements-runtime.txt`)
- ✅ Start script enhanced with Preview mode features
- ✅ Graceful fallbacks added to common modules
- ✅ CI integration configured

The Preview workspace is now self-contained and ready for CI validation. Once CI runs, we can review the output and update the runtime manifest if any additional dependencies are identified.

**Status**: Ready for CI execution and review.
