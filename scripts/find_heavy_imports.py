#!/usr/bin/env python3
"""Find heavy ML/GPU imports in the repository.

Searches for top-level occurrences of common heavy libraries that should be
imported lazily (e.g., torch, transformers, playwright, tensorrt, pycuda,
bitsandbytes). Prints file path, line number, and the matching line.

Usage:
    python scripts/find_heavy_imports.py [<path>]

If no path is provided, uses the repository root (script location's parent).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

HEAVY_IMPORT_PATTERNS = [
    r"^\s*import\s+torch\b",
    r"^\s*from\s+torch\b",
    r"^\s*from\s+transformers\b",
    r"^\s*import\s+transformers\b",
    r"^\s*from\s+playwright\b",
    r"^\s*import\s+playwright\b",
    r"^\s*import\s+tensorrt\b",
    r"^\s*from\s+tensorrt\b",
    r"^\s*import\s+pycuda\b",
    r"^\s*from\s+pycuda\b",
    r"^\s*from\s+bitsandbytes\b",
    r"^\s*import\s+bitsandbytes\b",
    r"^\s*from\s+transformers\s+import\s+BitsAndBytesConfig\b",
]

PATTERN = re.compile("(?:" + ")|(?:".join(p.strip() for p in HEAVY_IMPORT_PATTERNS) + ")")

EXCLUDE_DIRS = {"venv", ".venv", "__pycache__", ".git", ".idea", ".vscode", "build", "dist", "tmp", "quality_backups"}


def search_path(root: Path) -> list[tuple[Path, int, str]]:
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        # filter out unwanted dirs in-place for speed
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = Path(dirpath) / fname
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    for i, line in enumerate(fh, start=1):
                        if PATTERN.search(line):
                            results.append((fpath, i, line.rstrip()))
            except Exception:
                # skip unreadable files
                continue
    return results


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    print(f"Scanning for heavy imports under: {root}")

    matches = search_path(root)
    if not matches:
        print("No heavy imports found.")
        return 0

    # Group by file for nicer output
    from collections import defaultdict
    grouped = defaultdict(list)
    for fpath, lineno, line in matches:
        grouped[fpath].append((lineno, line))

    for fpath in sorted(grouped.keys()):
        print("\n" + str(fpath))
        for lineno, line in grouped[fpath]:
            print(f"  {lineno:4d}: {line}")

    print(f"\nTotal files with matches: {len(grouped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

