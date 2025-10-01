#!/usr/bin/env python3
"""Simple engine verifier: checks engine file and metadata JSON exist and basic fields match."""

import json
import sys
from pathlib import Path


def verify(engine_path: Path):
    meta_path = engine_path.with_suffix(".json")
    if not engine_path.exists():
        print("Engine file missing:", engine_path)
        return False
    if not meta_path.exists():
        print("Metadata missing:", meta_path)
        return False
    try:
        meta = json.loads(meta_path.read_text())
        required = ["task", "precision", "created_at"]
        for r in required:
            if r not in meta:
                print("Missing metadata field:", r)
                return False
    except Exception as e:
        print("Failed to read metadata:", e)
        return False
    print("Engine and metadata OK:", engine_path, meta.get("task"))
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: verify_engine.py <engine_path>")
        sys.exit(2)
    p = Path(sys.argv[1])
    ok = verify(p)
    sys.exit(0 if ok else 1)
