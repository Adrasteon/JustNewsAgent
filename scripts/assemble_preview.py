#!/usr/bin/env python3
"""Assemble a standalone preview tree from a pruned candidate report.

Usage:
  python3 scripts/assemble_preview.py --report reports/beta_candidate_pruned9.json --target /path/to/preview

The script copies the selected candidate files and a small set of companion
deployment files into the target directory, preserving relative paths.
It initializes a minimal git repository at the target and writes a basic
.gitignore to keep large model caches out.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path


DEFAULT_ADDITIONAL_FILES = [
    "environment.yml",
    "start_services_daemon.sh",
    "stop_services.sh",
    "agents/dashboard/dashboard_config.json",
    "scripts/db_operations.py",
    "scripts/migrate_performance_indexes.py",
    "release_beta_minimal_preview/README.md",
    "release_beta_minimal_preview/requirements.txt",
    "release_beta_minimal_preview/run_db_migrations.sh",
    "release_beta_minimal_preview/bootstrap_models_from_store.sh",
    "release_beta_minimal_preview/create_justnews_user.sh",
    "deploy/logrotate/justnews",
    "deploy/security/README_SECRETS.md",
    "deploy/systemd/units/overrides/justnews-defaults.conf",
    "deploy/systemd/examples/justnews.env.example",
    "agents/memory/db_migrations/001_create_articles_table.sql",
    "agents/memory/db_migrations/002_create_training_examples_table.sql",
    "agents/memory/db_migrations/003_create_article_vectors_table.sql",
    "agents/memory/db_migrations/004_add_performance_indexes.sql",
    "reports/beta_candidate_pruned9.json",
    "reports/beta_candidate_pruned9.md",
]


def copy_file_preserve(src: Path, dst_root: Path, repo_root: Path) -> None:
    rel = src.resolve().relative_to(repo_root.resolve())
    dest = dst_root.joinpath(rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def copy_dir_tree(src: Path, dst_root: Path, repo_root: Path) -> None:
    # copytree-like behavior but place under dst_root preserving path
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        for f in files:
            p = root_path / f
            copy_file_preserve(p, dst_root, repo_root)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--repo-root", default=os.getcwd())
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    target = Path(args.target).resolve()
    if target.exists():
        print(f"Target exists: {target} — removing to create clean preview")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    with open(args.report, "r", encoding="utf-8") as f:
        payload = json.load(f)

    candidates = payload.get("candidates", {})
    print(f"Copying {len(candidates)} candidate files to {target}")
    for abspath in candidates.keys():
        src = Path(abspath)
        if not src.exists():
            print(f"Warning: candidate not found: {src}")
            continue
        copy_file_preserve(src, target, repo_root)

    # Copy additional deployment files / dirs
    for rel in DEFAULT_ADDITIONAL_FILES:
        src = repo_root.joinpath(rel)
        if src.exists():
            if src.is_dir():
                copy_dir_tree(src, target, repo_root)
            else:
                copy_file_preserve(src, target, repo_root)
        else:
            print(f"Note: missing optional companion file: {rel}")

    # write a minimal .gitignore
    gi = target.joinpath(".gitignore")
    gi.write_text("""
# Model caches and large runtime artifacts
model_store/
agents/*/models/
*.db
logs/
.venv/
__pycache__/
""")

    # initialize git repo
    # create a tiny HEAD commit by writing .git dir using git init externally
    print("Preview tree assembled at", target)
    print("Next: run 'git init' and commit inside the preview directory to create standalone repo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
