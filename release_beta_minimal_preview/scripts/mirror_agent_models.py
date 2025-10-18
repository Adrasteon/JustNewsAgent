#!/usr/bin/env python3
"""
Lightweight helper to create symlinks from a central Model Store into agent model directories.

This script intentionally creates symlinks rather than copying large model files.
Operators should ensure the central Model Store is mounted or accessible.
"""

from pathlib import Path
import argparse
import logging
import shutil
import sys

logger = logging.getLogger("mirror_agent_models")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def mirror(agent: str, model_store: Path, dest_root: Path, dry_run: bool = False):
    """Create symlink for models used by an agent."""
    agent_models_dir = dest_root / agent / "models"
    agent_models_dir.mkdir(parents=True, exist_ok=True)

    # For simplicity, link any file under model_store/<agent> to dest
    source_dir = model_store / agent
    if not source_dir.exists():
        logger.error(f"Model store path does not exist for agent '{agent}': {source_dir}")
        return 1

    for item in source_dir.iterdir():
        dest = agent_models_dir / item.name
        logger.info(f"Linking {item} -> {dest}")
        if dry_run:
            continue

        if dest.exists():
            if dest.is_symlink() and dest.resolve() == item.resolve():
                logger.debug(f"Symlink already exists: {dest}")
                continue
            else:
                logger.warning(f"Destination exists and will be replaced: {dest}")
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()

        dest.symlink_to(item)

    logger.info(f"Finished linking models for agent: {agent}")
    return 0

def parse_args():
    p = argparse.ArgumentParser(description="Mirror Model Store into agent directories via symlinks")
    p.add_argument("--model-store", type=Path, required=True, help="Root path where a Model Store is mounted")
    p.add_argument("--dest-root", type=Path, default=Path("/opt/justnews/agents"), help="Destination root for agent data")
    p.add_argument("--agent", type=str, required=False, help="If provided, limit to a single agent")
    p.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    return p.parse_args()

def main():
    args = parse_args()
    model_store = args.model_store

    if args.agent:
        return mirror(args.agent, model_store, args.dest_root, dry_run=args.dry_run)

    # Mirror for all agents (each subdir in model_store)
    for agent_dir in model_store.iterdir():
        if agent_dir.is_dir():
            rc = mirror(agent_dir.name, model_store, args.dest_root, dry_run=args.dry_run)
            if rc != 0:
                return rc

    return 0

if __name__ == "__main__":
    sys.exit(main())
