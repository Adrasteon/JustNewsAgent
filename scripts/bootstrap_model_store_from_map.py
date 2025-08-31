#!/usr/bin/env python3
"""Bootstrap a MODEL_STORE_ROOT from markdown_docs/agent_documentation/AGENT_MODEL_MAP.json

Creates MODEL_STORE_ROOT/<agent>/<version>/ with model files downloaded via
huggingface_hub.snapshot_download when available, and falls back to using
transformers.from_pretrained to populate the cache directory.

The script is idempotent: if a version directory already contains files it will
skip downloads and ensure the `current` symlink points to the version.

Usage:
  MODEL_STORE_ROOT=/media/adra/Data/justnews/model_store python3 scripts/bootstrap_model_store_from_map.py

Optional environment variables:
  MODEL_STORE_ROOT - root path of model store (default: /media/adra/Data/justnews/model_store)
  MODEL_STORE_VERSION - version directory name (default: v1)
  MAP_FILE - path to the AGENT_MODEL_MAP.json (default: markdown_docs/agent_documentation/AGENT_MODEL_MAP.json)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
MAP_FILE = Path(os.environ.get('MAP_FILE', str(ROOT / 'markdown_docs' / 'agent_documentation' / 'AGENT_MODEL_MAP.json')))
MODEL_STORE_ROOT = Path(os.environ.get('MODEL_STORE_ROOT', '/media/adra/Data/justnews/model_store'))
VERSION_NAME = os.environ.get('MODEL_STORE_VERSION', 'v1')


def load_map(path: Path) -> Dict[str, List[str]]:
    with path.open() as f:
        return json.load(f)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def snapshot_download(repo_id: str, target: Path) -> bool:
    """Try to download a model snapshot to target. Return True if successful."""
    ensure_dir(target)
    # Try huggingface_hub.snapshot_download first
    try:
        from huggingface_hub import snapshot_download
        print(f"snapshot_download: {repo_id} -> {target}")
        snapshot_download(repo_id=repo_id, cache_dir=str(target), local_dir_use_symlinks=False)
        return True
    except Exception as e:
        print(f"snapshot_download not available or failed for {repo_id}: {e}")

    # Fallback: try to use transformers to populate cache
    try:
        from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
        print(f"Attempting transformers.from_pretrained for {repo_id}")
        AutoTokenizer.from_pretrained(repo_id, cache_dir=str(target))
        try:
            AutoModel.from_pretrained(repo_id, cache_dir=str(target))
        except Exception:
            try:
                AutoModelForCausalLM.from_pretrained(repo_id, cache_dir=str(target))
            except Exception as e:
                print(f"transformers fallback failed for model class: {e}")
        return True
    except Exception as e:
        print(f"transformers fallback not available for {repo_id}: {e}")

    return False


def populate_one_agent(agent: str, models: List[str]):
    print(f"\n--- Processing agent: {agent}")
    agent_dir = MODEL_STORE_ROOT / agent
    version_dir = agent_dir / VERSION_NAME
    ensure_dir(version_dir)

    # If the version directory already contains files, assume it's populated and skip downloads
    if any(version_dir.iterdir()):
        print(f"Version directory {version_dir} already populated; skipping downloads for {agent}")
        # Ensure current symlink points to this version
        current = agent_dir / 'current'
        try:
            if current.exists() or current.is_symlink():
                current.unlink()
            current.symlink_to(version_dir)
            print(f"Ensured current symlink for {agent}: {current} -> {version_dir}")
        except Exception as e:
            print(f"Failed to ensure symlink for {agent}: {e}")
        return

    # Create a temporary staging directory for downloads to keep atomicity
    with tempfile.TemporaryDirectory(dir=str(MODEL_STORE_ROOT)) as td:
        staging = Path(td) / VERSION_NAME
        staging.mkdir(parents=True, exist_ok=True)

        for m in models:
            try:
                ok = snapshot_download(m, staging)
                print(f"Downloaded {m}: {ok}")
            except Exception as e:
                print(f"Failed to download {m}: {e}")

        # Move staging to final version_dir atomically
        try:
            if version_dir.exists():
                shutil.rmtree(version_dir)
            shutil.move(str(staging), str(version_dir))
        except Exception as e:
            print(f"Failed to move staging to version dir for {agent}: {e}")

        # Create/ensure current symlink
        current = agent_dir / 'current'
        try:
            if current.exists() or current.is_symlink():
                current.unlink()
            current.symlink_to(version_dir)
            print(f"Created current symlink for {agent}: {current} -> {version_dir}")
        except Exception as e:
            print(f"Failed to create current symlink for {agent}: {e}")


def populate_all():
    print(f"MODEL_STORE_ROOT = {MODEL_STORE_ROOT}")
    ensure_dir(MODEL_STORE_ROOT)
    try:
        agent_map = load_map(MAP_FILE)
    except Exception as e:
        print(f"Failed to read agent map {MAP_FILE}: {e}")
        sys.exit(1)

    for agent, models in agent_map.items():
        populate_one_agent(agent, models)


if __name__ == '__main__':
    populate_all()
