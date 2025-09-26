# Version Management Standards

## Overview

This document outlines the standards and processes for maintaining version consistency across the JustNewsAgent codebase. Proper version management ensures reliable deployments, clear communication, and maintainable code.

## Core Principles

### 1. Single Source of Truth
- **Central Version**: All version references must originate from `justnews/__init__.py`
- **No Hardcoded Versions**: Never hardcode version numbers in application code
- **Import Pattern**: Always import version from the central location

```python
# ✅ Correct
from justnews import __version__

# ❌ Incorrect - Never do this
VERSION = "0.8.0"
```

### 2. Version Format Standards
- **Semantic Versioning**: Follow [SemVer](https://semver.org/) format: `MAJOR.MINOR.PATCH`
- **Pre-release**: Use `-beta`, `-alpha`, `-rc` suffixes for pre-releases
- **Examples**:
  - `0.8.0` - Production release
  - `0.9.0-beta` - Beta release
  - `1.0.0-rc.1` - Release candidate

### 3. Version Reference Categories

#### Application Versions (Must Match Central Version)
- API response versions: `"version": "0.8.0"`
- Health check responses: `"agent_version": "0.8.0"`
- README badges and version text
- Documentation examples and schemas

#### External Tool Versions (May Differ)
- Docker Compose versions: `version: '3.8'`
- Kubernetes API versions: `apiVersion: v1`
- HTTP protocol versions: `proxy_http_version 1.1`
- Database versions: PostgreSQL 16, CUDA 12.4

## File-Specific Standards

### Python Code Files
```python
# justnews/__init__.py - Central version definition
__version__ = "0.8.0"
VERSION_INFO = {
    "version": __version__,
    "status": "beta",
    "release_date": "2025-09-25",
    "description": "Beta release candidate with unified startup system"
}

# All other Python files
from justnews import __version__, VERSION_INFO
```

### API Endpoints
```python
# Health check endpoint
@app.get("/health")
def health_check():
    from justnews import __version__, VERSION_INFO
    return {
        "status": "healthy",
        "version": __version__,
        "version_info": VERSION_INFO
    }
```

### Documentation Files

#### README.md
```markdown
[![Version](https://img.shields.io/badge/version-0.8.0--beta-orange.svg)]()

**Version:** 0.8.0 (Beta)
```

#### API Documentation
```json
{
  "version": "0.8.0",
  "agent_version": "0.8.0",
  "api_version": "v4"
}
```

#### Agent Documentation
All JSON examples in agent documentation must use the current version:
```json
{
  "version": "0.8.0",
  "status": "success"
}
```

## Automated Tools

### Version Validation
Run version compliance checks before commits:
```bash
# Comprehensive validation
python scripts/validate_version_compliance.py

# Quick version check
python scripts/check_version.py
```

### Version Bumping
Use the automated version bump script for releases:
```bash
# Dry run first
python scripts/bump_version.py 0.9.0 --dry-run --verbose

# Perform the bump
python scripts/bump_version.py 0.9.0
```

## Development Workflow

### 1. Pre-Commit Checks
The pre-commit hook automatically validates version consistency:
```bash
git add .
git commit -m "Your commit message"
# Pre-commit hook runs version validation automatically
```

### 2. CI/CD Validation
GitHub Actions automatically validate version consistency on:
- Pull requests to main/dev branches
- Pushes to main/dev branches
- Manual workflow dispatch

### 3. Release Process
Follow the release process in `docs/RELEASE_PROCESS.md`:
1. Update CHANGELOG.md
2. Run version bump script
3. Validate all references
4. Create release commit and tag

## Common Pitfalls

### ❌ Incorrect Patterns
```python
# Hardcoded version
def get_version():
    return "0.8.0"  # Never do this

# Inconsistent version references
# README shows 0.8.0 but code shows 0.7.0

# Missing version imports
# API returns hardcoded version instead of importing
```

### ✅ Correct Patterns
```python
# Always import from central location
from justnews import __version__

def get_version():
    return __version__

# API responses
{
    "version": __version__,
    "status": "operational"
}
```

## Troubleshooting

### Version Mismatch Errors
If you encounter version validation failures:

1. **Check central version**: Verify `justnews/__init__.py` has the correct version
2. **Run validation**: `python scripts/validate_version_compliance.py --verbose`
3. **Fix inconsistencies**: Update any hardcoded or outdated references
4. **Re-validate**: Run validation again to confirm fixes

### Pre-commit Hook Failures
If commits are blocked by version validation:

1. **Check what changed**: Review your modifications for version references
2. **Run manual validation**: `python scripts/validate_version_compliance.py`
3. **Fix issues**: Address any version inconsistencies
4. **Bypass if needed**: Use `git commit --no-verify` only for false positives

### CI/CD Failures
If CI/CD pipelines fail version checks:

1. **Review CI logs**: Check which specific validation failed
2. **Fix locally**: Address issues on your branch
3. **Re-push**: Push fixes to trigger CI again

## Maintenance

### Regular Tasks
- **Weekly**: Run version validation on main branch
- **Pre-release**: Comprehensive version audit
- **Post-release**: Validate all references updated correctly

### Tool Updates
- Keep validation scripts updated with new file patterns
- Update CI/CD workflows as needed
- Review and update documentation standards annually

## Contact

For questions about version management:
- Check this document first
- Review existing issues/PRs for similar problems
- Create an issue with the "version-management" label

---

**Last Updated**: September 25, 2025
**Version**: 0.8.0