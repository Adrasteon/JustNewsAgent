#!/usr/bin/env python3
"""CI enforcement for documentation location and legacy link policy.

Checks:
- No Markdown files under deploy/systemd/ or deploy/monitoring/
- No links in docs/ or markdown_docs/ pointing to deploy/systemd or deploy/monitoring

Exit non-zero on violations with clear, actionable messages.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]


def find_legacy_md_files() -> List[Path]:
    """Return list of .md files under legacy locations that must be empty.

    We enforce that no Markdown files exist under:
    - deploy/systemd/**
    - deploy/monitoring/**
    """
    legacy_dirs = [REPO_ROOT / "deploy/systemd", REPO_ROOT / "deploy/monitoring"]
    bad: List[Path] = []
    for base in legacy_dirs:
        if not base.exists():
            continue
        for p in base.rglob("*.md"):
            bad.append(p)
    return bad


def scan_docs_for_legacy_links() -> List[Tuple[Path, int, str]]:
    """Scan docs/ and markdown_docs/ Markdown files for legacy links.

    Returns tuples of (file, line_no, line_text) for offending lines.
    """
    link_re = re.compile(r"\]\((deploy/(systemd|monitoring)/[^)]+)\)")
    offenders: List[Tuple[Path, int, str]] = []
    for docs_root in (REPO_ROOT / "docs", REPO_ROOT / "markdown_docs"):
        if not docs_root.exists():
            continue
        for md in docs_root.rglob("*.md"):
            try:
                with md.open("r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, start=1):
                        if link_re.search(line):
                            offenders.append((md.relative_to(REPO_ROOT), i, line.rstrip()))
            except Exception:  # pragma: no cover - robust in CI
                continue
    return offenders


def main() -> int:
    violations = {
        "legacy_markdown_files": [str(p.relative_to(REPO_ROOT)) for p in find_legacy_md_files()],
        "legacy_links": [
            {
                "file": str(p),
                "line": line_no,
                "text": text,
            }
            for (p, line_no, text) in scan_docs_for_legacy_links()
        ],
    }

    has_violations = bool(violations["legacy_markdown_files"] or violations["legacy_links"])
    if has_violations:
        print("::error::Documentation policy violations detected")
        print(json.dumps(violations, indent=2))
        print(
            "\nResolution:\n"
            "- Move any Markdown under deploy/systemd or deploy/monitoring into markdown_docs/ per policy.\n"
            "- Update links in docs/ and markdown_docs/ to the new markdown_docs/ paths.\n"
        )
        return 1

    print("Documentation policy check passed: no legacy files or links found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
