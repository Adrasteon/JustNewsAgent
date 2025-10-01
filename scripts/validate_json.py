#!/usr/bin/env python3
"""Validate all .json files in the repository by attempting to parse them with the stdlib json module.
Print any file path and exception encountered for quick triage.
"""
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
failed = []
for path in sorted(repo_root.rglob("*.json")):
    try:
        # read in text mode with utf-8
        text = path.read_text(encoding="utf-8")
        json.loads(text)
    except Exception as e:
        failed.append((str(path), str(e)))

if not failed:
    print("OK: all JSON files parsed successfully")
else:
    for p, err in failed:
        print(f"{p}: {err}")
    raise SystemExit(2)
