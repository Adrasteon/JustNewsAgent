#!/usr/bin/env python3
"""
CI Runtime Dependency Checker for JustNews V4 Preview

This script validates that all required runtime dependencies are importable
in the current environment. It is designed to run in CI to catch missing
dependencies early.

Exit codes:
  0 - All required dependencies are available
  1 - One or more required dependencies are missing
  2 - Critical error during execution
"""
from __future__ import annotations

import sys
from typing import List, Dict, Tuple
from pathlib import Path


# Core runtime dependencies required for all agents to start
# These are checked before attempting to import any agent code
CORE_RUNTIME_DEPS = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "requests",
]

# Extended runtime dependencies used by various agents
# These are nice-to-have but not all agents need all of them
EXTENDED_RUNTIME_DEPS = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",  # scikit-learn
    "torch",
    "transformers",
    "sentence_transformers",
    "spacy",
    "networkx",
    "psycopg2",
    "aiohttp",
    "httpx",
    "bs4",  # beautifulsoup4
    "yaml",  # pyyaml
    "jwt",  # pyjwt
    "cryptography",
    "prometheus_client",
    "psutil",
    "structlog",
]

# GPU/Acceleration dependencies (optional - agents fall back gracefully)
GPU_DEPS = [
    "pynvml",  # nvidia-ml-py3
    "GPUtil",
    "pycuda",
    "tensorrt",
]

# Advanced ML dependencies (optional - used by specific agents only)
ADVANCED_ML_DEPS = [
    "bertopic",
    "hdbscan",
    "umap",
    "faiss",
    "chromadb",
    "tomotopy",
    "textstat",
]

# Web scraping dependencies (used by scout/crawler)
SCRAPING_DEPS = [
    "playwright",
    "crawl4ai",
    "slowapi",
]

# Other optional dependencies
OPTIONAL_DEPS = [
    "PIL",  # Pillow
    "graphene",
    "graphql",
    "asyncpg",
    "huggingface_hub",
    "nucleoid",
]


def check_import(module_name: str) -> Tuple[bool, str]:
    """
    Attempt to import a module and return success status and error message.
    
    Args:
        module_name: Name of the module to import
        
    Returns:
        Tuple of (success: bool, error_message: str)
    """
    try:
        __import__(module_name)
        return True, ""
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"


def check_dependencies(
    deps: List[str], 
    category: str, 
    required: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Check a list of dependencies and report results.
    
    Args:
        deps: List of module names to check
        category: Human-readable category name for reporting
        required: Whether these dependencies are required or optional
        
    Returns:
        Tuple of (available, missing) module lists
    """
    print(f"\n{'='*60}")
    print(f"Checking {category} ({'Required' if required else 'Optional'})")
    print(f"{'='*60}")
    
    available = []
    missing = []
    
    for dep in deps:
        success, error = check_import(dep)
        if success:
            print(f"  ✓ {dep}")
            available.append(dep)
        else:
            marker = "✗" if required else "○"
            print(f"  {marker} {dep} - NOT AVAILABLE")
            if error:
                print(f"    Error: {error}")
            missing.append(dep)
    
    return available, missing


def print_summary(results: Dict[str, Tuple[List[str], List[str]]]) -> None:
    """Print a summary of all dependency checks."""
    print(f"\n{'='*60}")
    print("DEPENDENCY CHECK SUMMARY")
    print(f"{'='*60}")
    
    total_checked = 0
    total_available = 0
    total_missing = 0
    critical_missing = []
    
    for category, (available, missing) in results.items():
        checked = len(available) + len(missing)
        total_checked += checked
        total_available += len(available)
        total_missing += len(missing)
        
        status = "✓ PASS" if not missing else f"○ {len(missing)}/{checked} missing"
        if category == "Core Runtime" and missing:
            status = "✗ FAIL"
            critical_missing.extend(missing)
        
        print(f"  {category:30s}: {status}")
        if missing and category == "Core Runtime":
            for dep in missing:
                print(f"    - {dep} (CRITICAL)")
    
    print(f"\n{'='*60}")
    print(f"Total: {total_available}/{total_checked} available, {total_missing} missing")
    
    if critical_missing:
        print(f"\n⚠️  CRITICAL: {len(critical_missing)} required dependencies missing:")
        for dep in critical_missing:
            print(f"    - {dep}")
        print("\nInstall with: pip install " + " ".join(critical_missing))
        return False
    
    print("\n✓ All required runtime dependencies are available!")
    if total_missing > 0:
        print(f"  ({total_missing} optional dependencies missing - this is OK)")
    
    return True


def main() -> int:
    """Main entry point."""
    print("JustNews V4 - Runtime Dependency Check")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Check each category of dependencies
    results = {}
    
    results["Core Runtime"] = check_dependencies(
        CORE_RUNTIME_DEPS, "Core Runtime Dependencies", required=True
    )
    
    results["Extended Runtime"] = check_dependencies(
        EXTENDED_RUNTIME_DEPS, "Extended Runtime Dependencies", required=False
    )
    
    results["GPU/Acceleration"] = check_dependencies(
        GPU_DEPS, "GPU/Acceleration Dependencies", required=False
    )
    
    results["Advanced ML"] = check_dependencies(
        ADVANCED_ML_DEPS, "Advanced ML Dependencies", required=False
    )
    
    results["Web Scraping"] = check_dependencies(
        SCRAPING_DEPS, "Web Scraping Dependencies", required=False
    )
    
    results["Other Optional"] = check_dependencies(
        OPTIONAL_DEPS, "Other Optional Dependencies", required=False
    )
    
    # Print summary and determine exit code
    success = print_summary(results)
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nCRITICAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)
