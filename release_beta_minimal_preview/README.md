# JustNews V4 - Preview Minimal Runtime

This directory contains the minimal runtime configuration for deploying JustNews V4 in Preview mode.

## What is Preview Mode?

Preview mode is a lightweight deployment configuration designed for:
- Development environments
- CI/CD pipelines
- Testing and validation
- Resource-constrained deployments

## Files

- **requirements-runtime.txt**: Minimal set of Python dependencies needed to run the agent system
  - Core dependencies are uncommented and required
  - Optional dependencies are commented out and can be enabled as needed
  - GPU dependencies are optional with graceful fallback

## Installation

### Quick Start (Minimal)

```bash
# Install only core dependencies
pip install -r requirements-runtime.txt
```

### With Optional Features

Edit `requirements-runtime.txt` to uncomment the features you need:
- GPU Support: Uncomment `nvidia-ml-py3`, `GPUtil`, `pycuda`
- Advanced ML: Uncomment `bertopic`, `hdbscan`, `umap-learn`, `faiss-cpu`, etc.
- Image Processing: Uncomment `Pillow`
- GraphQL API: Uncomment `graphene`, `graphql-core`

Then install:
```bash
pip install -r requirements-runtime.txt
```

## CI Integration

The `ci_check_deps.py` script (located in the repository root) validates that all required dependencies are available:

```bash
python ci_check_deps.py
```

This script:
- Checks core runtime dependencies (fails if missing)
- Checks extended dependencies (reports but doesn't fail)
- Reports GPU and advanced ML dependencies as optional
- Provides actionable error messages

## Starting Services

Once dependencies are installed, start the agent system:

```bash
./start_services_daemon.sh
```

This will start all agents in daemon mode with health checks.

## Differences from Full Environment

The minimal runtime differs from the full development environment (`environment.yml`) in:

1. **No Conda**: Uses pip only (simpler for CI/containers)
2. **Optional GPU**: GPU dependencies are optional with runtime fallback
3. **Minimal ML**: Advanced ML packages are optional
4. **No Development Tools**: Excludes heavy development tools
5. **Flexible Versions**: Uses minimum versions with `>=` for compatibility

## Runtime Dependency Check

The CI runs `ci_check_deps.py` to validate dependencies. This ensures:
- All required packages are installable
- Import statements work correctly
- No missing dependencies at runtime

## Troubleshooting

### Missing Dependencies

If `ci_check_deps.py` reports missing dependencies:

1. Check if it's a **core dependency** (required):
   ```bash
   pip install <package-name>
   ```

2. Check if it's **optional** (can be skipped):
   - Read the error message to determine impact
   - Uncomment in `requirements-runtime.txt` if needed

### Import Errors

If agents fail to start with import errors:

1. Run the dependency check:
   ```bash
   python ci_check_deps.py
   ```

2. Install missing packages:
   ```bash
   pip install -r requirements-runtime.txt
   ```

3. Check agent-specific requirements in `agents/*/requirements.txt` (if present)

### GPU Issues

If GPU dependencies are missing but you don't need GPU:
- This is expected behavior
- Agents will fall back to CPU mode automatically
- No action required

If you need GPU support:
- Uncomment GPU dependencies in `requirements-runtime.txt`
- Ensure CUDA is installed on the system
- Verify GPU is accessible: `nvidia-smi`

## See Also

- `/start_services_daemon.sh` - Main startup script
- `/ci_check_deps.py` - Dependency validation script
- `/environment.yml` - Full conda environment (for development)
- `.github/workflows/ci.yml` - CI configuration with dependency checks
