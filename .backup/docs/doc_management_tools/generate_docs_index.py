#!/usr/bin/env python3
"""
Generate docs_index.json from markdown_docs/ structure and frontmatter.

- Groups by top-level markdown_docs categories (agent_documentation, production_status, development_reports, optimization_reports, etc.)
- Extracts frontmatter fields for title/description/last_updated/status
- Writes docs_index.json at repository root

Usage:
  python docs/doc_management_tools/generate_docs_index.py --write
  python docs/doc_management_tools/generate_docs_index.py --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

FM_START = re.compile(r"^---\s*$")
FM_END = re.compile(r"^---\s*$")


def parse_frontmatter(text: str) -> Dict[str, str]:
    lines = text.splitlines()
    fm: Dict[str, str] = {}
    if not lines or not FM_START.match(lines[0]):
        return fm
    for i in range(1, min(len(lines), 200)):
        if FM_END.match(lines[i]):
            for ln in lines[1:i]:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    fm[k.strip()] = v.strip().strip("'\"")
            break
    return fm


def scan_docs(root: Path) -> List[Dict]:
    md_root = root / "markdown_docs"
    entries: Dict[str, List[Dict]] = {}
    for md in md_root.rglob("*.md"):
        rel = md.relative_to(root)
        parts = rel.parts
        # category is markdown_docs/<category>/...
        if len(parts) < 3:
            # skip top-level README in markdown_docs
            continue
        category = parts[1]
        text = md.read_text(encoding="utf-8", errors="replace")
        fm = parse_frontmatter(text)
        entry = {
            "path": str(md),
            "title": fm.get("title") or md.stem.replace("_", " "),
            "description": fm.get("description") or "",
            "last_updated": fm.get("last_updated") or "",
            "status": fm.get("status") or "",
        }
        entries.setdefault(category, []).append(entry)
    # Build output structure (list of categories)
    out = []
    for cat, files in sorted(entries.items()):
        out.append({"category": cat.replace("_", " ").title(), "files": sorted(files, key=lambda f: f["title"].lower())})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    data = scan_docs(root)

    if args.dry_run:
        print(json.dumps(data[:2], indent=2))
        return 0

    out_path = root / "docs_index.json"
    if args.write:
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
