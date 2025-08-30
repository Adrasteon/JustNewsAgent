#!/usr/bin/env python3
"""Migrate per-agent `agents/<agent>/models` directories to a large-volume target and create symlinks.

This script is conservative by default (requires --yes to perform actions) and supports a dry-run mode.

Behavior:
- For each subdirectory under `agents/` (skips hidden names), if `agents/<agent>/models` exists:
  - Create `TARGET/<agent>/models.tmp.<pid>` and copy the source contents into it.
  - Atomically rename the tmp folder to `TARGET/<agent>/models` where possible.
  - Remove the original `agents/<agent>/models` and create a symlink `agents/<agent>/models -> TARGET/<agent>/models`.
- If `agents/<agent>/models` doesn't exist, no action is taken by default.

Notes:
- For true atomic renames the TARGET should be on the same filesystem. If not, the script will attempt a safe move/copy fallback.
- Use --dry-run to preview actions.

Usage examples:
    # Canonical target layout (recommended): /media/adra/Data/justnews/agents
    python3 scripts/mirror_agent_models.py --target /media/adra/Data/justnews/agents --yes
    python3 scripts/mirror_agent_models.py --target /media/adra/Data/justnews/agents --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable


def iter_agents(agents_root: Path) -> Iterable[Path]:
    if not agents_root.is_dir():
        return []
    for entry in sorted(agents_root.iterdir()):
        if entry.name.startswith('.'):
            continue
        if entry.is_dir():
            yield entry


def copy_tree_safe(src: Path, dst: Path) -> None:
    """Copy src tree to dst. dst must not exist."""
    shutil.copytree(src, dst, symlinks=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def perform_migration(agent_dir: Path, target_base: Path, dry_run: bool = True) -> None:
    src_models = agent_dir / 'models'
    if not src_models.exists():
        print(f"[SKIP] {agent_dir.name}: no models/ directory")
        return

    target_agent_models = target_base / agent_dir.name / 'models'
    tmp_suffix = f'.tmp.{os.getpid()}'
    tmp_target = target_agent_models.with_name(target_agent_models.name + tmp_suffix)

    print(f"\n[PLAN] Agent: {agent_dir.name}")
    print(f"  source: {src_models}")
    print(f"  target: {target_agent_models}")
    print(f"  staging: {tmp_target}")

    if dry_run:
        return

    # Ensure target parent exists
    ensure_parent(target_agent_models)

    if tmp_target.exists():
        print(f"  [WARN] Removing stale staging folder {tmp_target}")
        shutil.rmtree(tmp_target)

    # Copy tree into staging
    print(f"  [COPY] copying {src_models} -> {tmp_target}")
    copy_tree_safe(src_models, tmp_target)

    # Attempt atomic replace
    try:
        # Ensure final parent exists
        (target_agent_models.parent).mkdir(parents=True, exist_ok=True)
        if target_agent_models.exists():
            # if target exists, remove it to allow replace
            print(f"  [INFO] target exists, removing {target_agent_models}")
            shutil.rmtree(target_agent_models)
        print(f"  [RENAME] {tmp_target} -> {target_agent_models}")
        os.replace(str(tmp_target), str(target_agent_models))
    except Exception as e:
        print(f"  [WARN] atomic rename failed: {e}; attempting fallback move")
        try:
            shutil.move(str(tmp_target), str(target_agent_models))
        except Exception as e2:
            print(f"  [ERROR] fallback move failed: {e2}")
            # Cleanup tmp_target if exists
            if tmp_target.exists():
                shutil.rmtree(tmp_target)
            raise

    # Now create symlink from agent_dir/models -> target_agent_models
    if src_models.exists():
        print(f"  [REMOVE] removing original {src_models}")
        shutil.rmtree(src_models)

    print(f"  [LINK] {agent_dir / 'models'} -> {target_agent_models}")
    os.symlink(str(target_agent_models), str(agent_dir / 'models'))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Mirror per-agent models to a large-volume target and symlink them back.")
    parser.add_argument('--agents', default='agents', help='Path to agents directory (default: agents)')
    parser.add_argument('--target', required=False, help='Target base directory on the large volume (overrides DATA_DRIVE_TARGET env var).')
    parser.add_argument('--dry-run', action='store_true', help='Show actions without performing them')
    parser.add_argument('--yes', action='store_true', help='Perform actions (required to actually migrate)')
    parser.add_argument('--agent', default=None, help='Only migrate a single agent by name')
    args = parser.parse_args(argv)

    agents_root = Path(args.agents).resolve()

    # Determine target: CLI --target wins, then DATA_DRIVE_TARGET env var, then default to
    # the canonical data path which includes the `agents/` prefix so final model folders
    # will be: /media/adra/Data/justnews/agents/<agent>/models
    env_target = os.environ.get('DATA_DRIVE_TARGET')
    if args.target:
        target_base = Path(args.target).resolve()
    elif env_target:
        target_base = Path(env_target).resolve()
    else:
        # Default canonical data drive path (includes agents/ prefix)
        target_base = Path('/media/adra/Data/justnews/agents').resolve()

    if not args.dry_run and not args.yes:
        print("Refusing to run: pass --yes to perform actions (or use --dry-run to preview).")
        return 2

    agents = list(iter_agents(agents_root))
    if args.agent:
        agents = [a for a in agents if a.name == args.agent]
        if not agents:
            print(f"Agent {args.agent} not found under {agents_root}")
            return 3

    print(f"Found {len(agents)} agents under {agents_root}")

    for agent in agents:
        try:
            perform_migration(agent, target_base, dry_run=args.dry_run)
        except Exception as e:
            print(f"[ERROR] migration failed for {agent.name}: {e}")

    print("\nDone.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
