# Runtime Dependency Management for JustNews

This document describes the runtime dependency management system introduced for production deployments.

## Overview

JustNews uses a multi-layered dependency checking system to ensure all required Python modules are available before services start:

1. **Preflight checks** - Validate dependencies before system startup
2. **Per-agent checks** - Verify agent-specific dependencies at startup
3. **CI validation** - Automated checks in GitHub Actions

## Components

### 1. Preflight Check Script

**Location**: `deploy/systemd/scripts/justnews-preflight-check.sh`

Performs pre-startup validation including Python runtime dependency checks for agents. Prefers conda environments when available.

**Usage**:
```bash
# Full check
sudo ./deploy/systemd/scripts/justnews-preflight-check.sh

# Gate-only mode (for MCP Bus)
./deploy/systemd/scripts/justnews-preflight-check.sh --gate-only
```

### 2. Agent Startup Script

**Location**: `deploy/systemd/scripts/justnews-start-agent.sh`

Standardized agent startup with runtime dependency validation. Automatically detects conda environments or production venv.

**Usage**:
```bash
./deploy/systemd/scripts/justnews-start-agent.sh <agent_name>
```

### 3. CI Dependency Checker

**Location**: `deploy/systemd/scripts/ci_check_deps.py`

Maps agents to required runtime modules and validates their presence. Returns non-zero exit code on missing modules.

**Usage**:
```bash
python deploy/systemd/scripts/ci_check_deps.py
```

**Agent Module Mappings**:
- `mcp_bus`: requests
- `gpu_orchestrator`: requests, uvicorn
- `chief_editor`: requests
- `default`: requests (fallback for other agents)

### 4. Deterministic Venv Builder

**Location**: `deploy/systemd/scripts/build_service_venv.sh`

Creates lightweight Python virtual environments with only runtime dependencies for fast CI validation and production deployment.

**Usage**:
```bash
# Default: creates ./venv using ./requirements-runtime.txt
./deploy/systemd/scripts/build_service_venv.sh

# Custom target and requirements
./deploy/systemd/scripts/build_service_venv.sh /opt/justnews/venv requirements-runtime.txt
```

**Features**:
- Python 3.10+ version validation
- Automatic pip upgrade
- Installation verification
- Clear error messages

### 5. Runtime Requirements File

**Location**: `release_beta_minimal_preview/requirements-runtime.txt`

Minimal runtime-only dependencies for Preview workspace. Contains only packages needed to start and run agents, without development, testing, or full ML dependencies.

**Included Packages**:
- fastapi, uvicorn, pydantic (web framework)
- requests (HTTP client)
- psycopg2-binary, sqlalchemy (database)
- python-dotenv (configuration)

## Deployment Workflow

### For Development

1. Use conda environment with full dependencies:
   ```bash
   conda activate justnews-v2-py312
   ```

2. Dependency checks automatically prefer conda:
   ```bash
   # Automatically uses: conda run -n justnews-v2-py312 python
   ./deploy/systemd/scripts/justnews-start-agent.sh mcp_bus
   ```

### For Production

1. Build production venv:
   ```bash
   sudo /opt/justnews/deploy/systemd/scripts/build_service_venv.sh \
     /opt/justnews/venv \
     /opt/justnews/release_beta_minimal_preview/requirements-runtime.txt
   ```

2. Set PYTHON_BIN in global env:
   ```bash
   echo "PYTHON_BIN=/opt/justnews/venv/bin/python" | sudo tee -a /etc/justnews/global.env
   ```

3. Start services:
   ```bash
   sudo systemctl start justnews@mcp_bus.service
   ```

## CI Integration

### Preview Workflow

**Location**: `.github/workflows/preview.yml`

Validates runtime dependencies for the Preview workspace on every PR.

**Steps**:
1. Checkout repository
2. Set up Python 3.12
3. Build lightweight venv using `build_service_venv.sh`
4. Run dependency checks with `ci_check_deps.py`
5. Report results

**Trigger**: Pull requests and pushes to main/dev that modify:
- `release_beta_minimal_preview/**`
- `deploy/systemd/scripts/ci_check_deps.py`
- `deploy/systemd/scripts/build_service_venv.sh`
- `.github/workflows/preview.yml`

### Main CI Workflow

**Location**: `.github/workflows/ci.yml`

Includes runtime dependency check as part of standard CI:

```yaml
- name: Check production runtime dependencies
  run: python deploy/systemd/scripts/ci_check_deps.py
```

## Error Handling

All dependency checks provide actionable error messages:

### Conda Environment
```
Missing python modules for agent 'gpu_orchestrator': uvicorn
Install into the developer conda env (example): 
  conda run -n justnews-v2-py312 pip install uvicorn
```

### Production Venv
```
Missing python modules for agent 'mcp_bus': requests
Install them into the service venv (example): 
  sudo /opt/justnews/venv/bin/pip install requests
```

## Maintenance

### Adding New Agent Dependencies

1. Update `ci_check_deps.py` AGENT_MODULE_MAP:
   ```python
   AGENT_MODULE_MAP = {
       "new_agent": ["requests", "new_module"],
       ...
   }
   ```

2. Add to `requirements-runtime.txt` if needed:
   ```
   new_module>=1.0.0
   ```

3. Run CI to validate:
   ```bash
   python deploy/systemd/scripts/ci_check_deps.py
   ```

### Updating Runtime Requirements

1. Edit `release_beta_minimal_preview/requirements-runtime.txt`
2. Rebuild venv:
   ```bash
   ./deploy/systemd/scripts/build_service_venv.sh
   ```
3. Test with dependency checker:
   ```bash
   ./venv/bin/python deploy/systemd/scripts/ci_check_deps.py
   ```

## Additional Resources

- **Logrotate Config**: `deploy/systemd/logrotate.conf` - Log rotation for all services
- **Systemd Units**: `deploy/systemd/units/` - Service definitions with dependency ordering
- **Environment Examples**: `deploy/systemd/examples/` - Configuration templates

## Troubleshooting

### "No Python interpreter found"
Ensure either conda is in PATH or PYTHON_BIN is set in `/etc/justnews/global.env`.

### "Missing python modules"
Follow the actionable install command in the error message.

### "Virtual environment build failed"
Check Python version (must be 3.10+) and verify requirements file exists.

### CI Workflow Fails
Review GitHub Actions logs for specific missing modules and update `requirements-runtime.txt`.
