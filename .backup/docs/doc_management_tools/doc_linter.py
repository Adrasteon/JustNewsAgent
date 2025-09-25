#!/usr/bin/env python3
"""
Docs Linter and (optional) Quick Fixer

Validates markdown docs for:
- Location rule: all .md must be under markdown_docs/ or docs/ (except root README.md and CHANGELOG.md)
- Frontmatter presence with required fields: title, description, tags, status, last_updated
- Suspicious code fences: accidental "````markdown" fences
- Optional: presence of "## See also" section (warn only)

Optionally, with --fix:
- Injects basic frontmatter if missing (title from H1 or filename, last_updated = today)
- Converts leading "````markdown"/trailing "````" fences to standard triple backticks

Usage:
  python docs/doc_management_tools/doc_linter.py --report
  python docs/doc_management_tools/doc_linter.py --report --fix

Exit codes:
  0 on success (no hard violations or fixed)
  1 if violations found and --fix not provided
  2 on unexpected error
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REQUIRED_KEYS = ["title", "description", "tags", "status", "last_updated"]
ALLOWED_ROOT_MD = {"README.md", "CHANGELOG.md"}

# Directories to ignore entirely (vendor caches, third-party subrepos, ephemeral outputs)
IGNORED_DIR_PARTS = {
    ".git",
    ".github",
    "venv",
    "node_modules",
    "archive_obsolete_files",
    "model_cache",
    ".pytest_cache",
    "large_scale_crawl_results",
}

# Paths that are considered third-party content to skip (do not lint/move)
THIRD_PARTY_HINTS = [
    "agents/reasoning/nucleoid_repo",
]

FM_START = re.compile(r"^---\s*$")
FM_END = re.compile(r"^---\s*$")
H1_RE = re.compile(r"^#\s+(.+?)\s*$")
QUAD_FENCE_START = re.compile(r"^````(\w+)?\s*$")
QUAD_FENCE_END = re.compile(r"^````\s*$")

@dataclass
class DocIssue:
    path: str
    kind: str
    message: str

@dataclass
class Frontmatter:
    raw: str
    data: Dict[str, object]


def load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def save_text(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")


def parse_frontmatter(text: str) -> Tuple[Optional[Frontmatter], str]:
    lines = text.splitlines()
    if not lines or not FM_START.match(lines[0]):
        return None, text
    # find end
    for i in range(1, min(len(lines), 200)):
        if FM_END.match(lines[i]):
            fm_block = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1 :])
            try:
                # naive YAML: use json-like fallback if braces; else parse key: value lines
                data: Dict[str, object] = {}
                for ln in fm_block.splitlines():
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    if ":" in ln:
                        k, v = ln.split(":", 1)
                        key = k.strip()
                        val = v.strip().strip("\"'")
                        # split tags by comma if looks like list string
                        if key == "tags" and ("," in val or val.startswith("[") or val.endswith("]")):
                            val = [t.strip().strip("\"'") for t in re.split(r"[,\[\]]", val) if t.strip() and t.strip() not in {"[", "]"}]
                        data[key] = val
                return Frontmatter(raw=fm_block, data=data), body
            except Exception:
                return None, text
    return None, text


def ensure_frontmatter(body: str, filename: str, existing_fm: Optional[Frontmatter]) -> Tuple[str, bool]:
    changed = False
    fm_data = existing_fm.data.copy() if existing_fm else {}
    # Title
    if "title" not in fm_data:
        # try h1 from body
        title = None
        for ln in body.splitlines()[:10]:
            m = H1_RE.match(ln)
            if m:
                title = m.group(1).strip()
                break
        if not title:
            title = Path(filename).stem.replace("_", " ")
        fm_data["title"] = title
        changed = True
    # Description placeholder if missing
    if "description" not in fm_data:
        fm_data["description"] = f"Auto-generated description for {fm_data['title']}"
        changed = True
    # Tags minimal default
    if "tags" not in fm_data:
        fm_data["tags"] = ["documentation"]
        changed = True
    # Status default
    if "status" not in fm_data:
        fm_data["status"] = "current"
        changed = True
    # last_updated set to today if missing
    if "last_updated" not in fm_data:
        fm_data["last_updated"] = str(date.today())
        changed = True

    # Recompose frontmatter
    fm_lines = ["---"]
    fm_lines.append(f"title: {fm_data['title']}")
    fm_lines.append(f"description: {fm_data['description']}")
    tags_val = fm_data["tags"]
    if isinstance(tags_val, list):
        tags_serialized = "[" + ", ".join(tags_val) + "]"
    else:
        tags_serialized = str(tags_val)
    fm_lines.append(f"tags: {tags_serialized}")
    fm_lines.append(f"status: {fm_data['status']}")
    fm_lines.append(f"last_updated: {fm_data['last_updated']}")
    fm_lines.append("---")

    fm_text = "\n".join(fm_lines) + "\n\n"
    if existing_fm is None:
        return fm_text + body.lstrip(), True
    else:
        # Body came without original FM; if we had existing, changed only if we added fields
        if changed:
            return fm_text + body.lstrip(), True
        return "---\n" + existing_fm.raw + "\n---\n\n" + body, False


def fix_quad_fences(text: str) -> Tuple[str, bool]:
    changed = False
    new_lines: List[str] = []
    for ln in text.splitlines():
        if QUAD_FENCE_START.match(ln):
            new_lines.append("```")
            changed = True
        elif QUAD_FENCE_END.match(ln):
            new_lines.append("```")
            changed = True
        else:
            new_lines.append(ln)
    return "\n".join(new_lines), changed


def is_location_violation(root: Path, p: Path) -> bool:
    rel = p.relative_to(root)
    if len(rel.parts) == 1:  # in repo root
        return rel.name not in ALLOWED_ROOT_MD
    # ok if under markdown_docs or docs
    return not (str(rel).startswith("markdown_docs/") or str(rel).startswith("docs/"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="Apply safe fixes (frontmatter injection, quad-fence normalization)")
    ap.add_argument("--report", action="store_true", help="Print JSON report of issues")
    ap.add_argument("--strict-location", action="store_true", help="Treat out-of-place markdown as hard violations (default: warn only)")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]), help="Workspace root (auto-detected)")
    ap.add_argument("--add-seealso", action="store_true", help="When combined with --fix, inject a default 'See also' section if missing")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    issues: List[DocIssue] = []
    fixed: List[str] = []

    md_files: List[Path] = []
    for p in root.rglob("*.md"):
        # skip noisy/vendor dirs
        if any(part in IGNORED_DIR_PARTS for part in p.parts):
            continue
        # skip well-known third-party trees
        rel_str = str(p.relative_to(root))
        if any(hint in rel_str for hint in THIRD_PARTY_HINTS):
            continue
        # skip vendor-like model README trees under agents/**/models/**
        parts = p.parts
        if "agents" in parts and "models" in parts:
            continue
        md_files.append(p)

    for p in md_files:
        try:
            text = load_text(p)
            if is_location_violation(root, p):
                kind = "location" if args.strict_location else "location-warn"
                issues.append(DocIssue(str(p), kind, "Markdown file is outside markdown_docs/ or docs/"))

            fm, body = parse_frontmatter(text)
            if fm is None:
                issues.append(DocIssue(str(p), "frontmatter", "Missing frontmatter"))
                if args.fix:
                    new_text, _ = ensure_frontmatter(text, p.name, fm)
                    text = new_text
                    fixed.append(str(p))
                    fm, body = parse_frontmatter(text)
            else:
                # Check required keys
                for k in REQUIRED_KEYS:
                    if k not in fm.data:
                        issues.append(DocIssue(str(p), "frontmatter", f"Missing key: {k}"))

            if "````markdown" in text:
                issues.append(DocIssue(str(p), "fence", "Contains quad-backtick markdown fence"))
                if args.fix:
                    text2, changed = fix_quad_fences(text)
                    if changed:
                        text = text2
                        fixed.append(str(p))

            # Warn if missing See also, optionally insert a default section
            if "## See also" not in text:
                issues.append(DocIssue(str(p), "seealso", "Missing 'See also' section (warning)"))
                if args.fix and args.add_seealso:
                    default_block = (
                        "\n\n## See also\n\n"
                        "- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md\n"
                        "- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md\n"
                    )
                    text = text.rstrip() + default_block + "\n"
                    fixed.append(str(p))

            if args.fix and text != load_text(p):
                save_text(p, text)
        except Exception as e:
            issues.append(DocIssue(str(p), "error", f"Exception: {e}"))

    if args.report:
        print(json.dumps({
            "total_files": len(md_files),
            "issues": [issue.__dict__ for issue in issues],
            "fixed": fixed,
        }, indent=2))

    # Exit status
    hard_violations = [i for i in issues if i.kind in {"location", "frontmatter", "fence", "error"}]
    if hard_violations and not args.fix:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(2)
