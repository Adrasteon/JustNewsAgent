#!/usr/bin/env python3
"""Populate the central MODEL_STORE_ROOT with canonical per-agent model folders.

This script will:
- Read markdown_docs/agent_documentation/AGENT_MODEL_MAP.json for agent->model list
- For each agent, create MODEL_STORE_ROOT/<agent>/v1 and download each model there
- Create an atomic symlink MODEL_STORE_ROOT/<agent>/current -> MODEL_STORE_ROOT/<agent>/v1
- Remove workspace-local fallback model directories (./models and agents/*/models)

WARNING: This script will DELETE workspace-local model folders as requested. Do not run unless you want to proceed.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_FILE = ROOT / "markdown_docs" / "agent_documentation" / "AGENT_MODEL_MAP.json"
MODEL_STORE_ROOT = Path(
    os.environ.get("MODEL_STORE_ROOT", "/media/adra/Data/justnews/model_store")
)
VERSION_NAME = os.environ.get("MODEL_STORE_VERSION", "v1")


def load_map() -> dict[str, list[str]]:
    with MAP_FILE.open() as f:
        return json.load(f)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def snapshot_download(repo_id: str, target: Path):
    """Use huggingface_hub snapshot_download when available, otherwise try transformers loader to warm cache."""
    try:
        from huggingface_hub import snapshot_download

        print(f"snapshot_download: {repo_id} -> {target}")
        snapshot_download(
            repo_id=repo_id, cache_dir=str(target), local_dir_use_symlinks=False
        )
        return True
    except Exception as e:
        print(f"snapshot_download not available or failed for {repo_id}: {e}")
    # Fallback: try to import with transformers to force cache
    try:
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        print(f"Attempting transformers.from_pretrained for {repo_id}")
        # Try tokenizer + model
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


def download_spacy_model(model: str, target: Path) -> bool:
    """Download and extract a spacy model to a target directory."""
    import subprocess
    import sys

    try:
        print(f"Downloading spacy model: {model}")
        # It's easier to download and then move.
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model])

        import spacy

        nlp = spacy.load(model)
        model_path = nlp.path

        model_target_path = target / model
        if model_target_path.exists():
            shutil.rmtree(model_target_path)

        shutil.copytree(model_path, model_target_path)
        print(f"Copied spacy model {model} from {model_path} to {model_target_path}")
        return True
    except Exception as e:
        print(f"Failed to download or place spacy model {model}: {e}")
        return False


def populate():
    agent_map = load_map()
    ensure_dir(MODEL_STORE_ROOT)
    for agent, models in agent_map.items():
        print(f"\n--- Processing agent: {agent}")
        agent_dir = MODEL_STORE_ROOT / agent
        version_dir = agent_dir / VERSION_NAME
        ensure_dir(version_dir)
        # If the version directory already contains files, assume it's populated and skip downloads
        if any(version_dir.iterdir()):
            print(
                f"Version directory {version_dir} already populated; skipping downloads for {agent}"
            )
            # Ensure current symlink points to this version
            current = agent_dir / "current"
            if current.exists() or current.is_symlink():
                current.unlink()
            current.symlink_to(version_dir)
            print(f"Ensured current symlink for {agent}: {current} -> {version_dir}")
            continue
        for m in models:
            # Normalize HF ids (sentence-transformers may include path)
            try:
                if m.startswith("en_"):  # Simple check for spacy model
                    ok = download_spacy_model(m, version_dir)
                else:
                    ok = snapshot_download(m, version_dir)
                print(f"Downloaded {m}: {ok}")
            except Exception as e:
                print(f"Failed to download {m}: {e}")
        # Create atomic symlink
        current = agent_dir / "current"
        if current.exists() or current.is_symlink():
            current.unlink()
        current.symlink_to(version_dir)
        print(f"Created current symlink for {agent}: {current} -> {version_dir}")

    # Remove workspace-local model folders (destructive by user request)
    print("\nRemoving workspace-local './models' folder if present...")
    local_models = ROOT / "models"
    if local_models.exists():
        if local_models.is_dir():
            shutil.rmtree(local_models)
            print("Removed ./models")
        else:
            local_models.unlink()

    # Remove agents/*/models
    print("Removing agents/*/models folders if present...")
    agents_dir = ROOT / "agents"
    for sub in agents_dir.iterdir():
        models_dir = sub / "models"
        if models_dir.exists() or models_dir.is_symlink():
            # If it's a symlink, unlink it. If it's a directory, remove it.
            try:
                if models_dir.is_symlink():
                    models_dir.unlink()
                    print(f"Unlinked symlink {models_dir}")
                elif models_dir.is_dir():
                    shutil.rmtree(models_dir)
                    print(f"Removed {models_dir}")
                else:
                    models_dir.unlink()
                    print(f"Removed file {models_dir}")
            except Exception as e:
                print(f"Failed to remove {models_dir}: {e}")


if __name__ == "__main__":
    print(f"MODEL_STORE_ROOT = {MODEL_STORE_ROOT}")
    populate()
