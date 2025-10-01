#!/usr/bin/env python3
"""
Safe codemod to find and optionally replace insecure MD5 usages across the repo.

Rules applied:
- Replace occurrences of `hashlib.md5(` with `hashlib.sha256(` in Python code.
- Replace `hashlib.md5(...).hexdigest()` with `hashlib.sha256(...).hexdigest()`.
- Replace SQL examples using `md5('...')` or `md5("...")` with
  `encode(digest('...','sha256'),'hex')` to preserve hex output while using SHA-256 via pgcrypto.

Usage:
  - Preview only (no writes): python scripts/replace_md5_with_sha256.py
  - Apply changes in-place:    python scripts/replace_md5_with_sha256.py --apply

The script writes a report to `./.md5_rewrite_report.json` when run.
"""

from __future__ import annotations

import argparse
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Patterns and replacements
PY_HASHLIB_MD5 = re.compile(r"hashlib\.md5\(")
PY_HASHLIB_MD5_HEX = re.compile(r"hashlib\.md5\((.*?)\)\.hexdigest\(\)")
# matches md5('literal') or md5("literal") and captures the literal
SQL_MD5_LITERAL = re.compile(r"md5\((['\"]).*?\1\)")
SQL_MD5_CAPTURE = re.compile(r"md5\((['\"])(?P<inner>.*?)\1\)")

# File extensions to process
TEXT_EXTS = (
    ".py",
    ".md",
    ".markdown",
    ".json",
    ".sql",
    ".txt",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
)

# Paths to skip (common large or binary dirs)
SKIP_DIRS = {".git", "node_modules", "__pycache__"}


def is_text_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in TEXT_EXTS


def replace_in_text(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (new_text, list_of_replacements) for a single file's text."""
    replacements: list[tuple[str, str]] = []
    new_text = text

    # 1) hashlib.md5(...) -> hashlib.sha256(...)
    if PY_HASHLIB_MD5.search(new_text):
        new_text = PY_HASHLIB_MD5.sub("hashlib.sha256(", new_text)
        replacements.append(("hashlib.md5(", "hashlib.sha256("))

    # 2) hashlib.md5(...).hexdigest() -> hashlib.sha256(...).hexdigest()
    # Use the HEURISTIC: simply replace .md5(...).hexdigest() to .sha256(...).hexdigest()
    if PY_HASHLIB_MD5_HEX.search(new_text):
        new_text = PY_HASHLIB_MD5_HEX.sub(r"hashlib.sha256(\1).hexdigest()", new_text)
        replacements.append(
            ("hashlib.md5(...).hexdigest()", "hashlib.sha256(...).hexdigest()")
        )

    # 3) SQL md5('...') literal -> encode(digest('...','sha256'),'hex')
    # We perform a safe capture-and-replace to preserve the inner quoted literal.
    def _sql_md5_repl(m: re.Match) -> str:
        inner = m.group("inner")
        # encode(digest('text','sha256'),'hex') preserves the hex string returned by md5()
        return f"encode(digest('{inner}','sha256'),'hex')"

    if SQL_MD5_CAPTURE.search(new_text):
        new_text = SQL_MD5_CAPTURE.sub(_sql_md5_repl, new_text)
        replacements.append(("md5('..')", "encode(digest('..','sha256'),'hex')"))

    return new_text, replacements


def find_files(root: str) -> list[str]:
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # filter out skip dirs in-place to prune traversal
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            if is_text_file(path):
                files.append(path)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find & replace MD5 usage with SHA-256 variants."
    )
    parser.add_argument("--apply", action="store_true", help="Apply changes in-place")
    parser.add_argument(
        "--root",
        default=ROOT,
        help="Repository root path (default: parent of this script)",
    )
    args = parser.parse_args(argv)

    files = find_files(args.root)

    report = {
        "root": args.root,
        "applied": bool(args.apply),
        "files_scanned": len(files),
        "changes": [],
    }

    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, PermissionError):
            # skip binary or unreadable files
            continue

        new_text, replacements = replace_in_text(text)
        if replacements and new_text != text:
            change_entry = {
                "file": os.path.relpath(path, args.root),
                "replacements": [r[0] + " -> " + r[1] for r in replacements],
            }
            report["changes"].append(change_entry)
            if args.apply:
                # write a backup and then the new file
                backup_path = path + ".bak"
                if not os.path.exists(backup_path):
                    with open(backup_path, "w", encoding="utf-8") as bf:
                        bf.write(text)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_text)

    # write report
    report_path = os.path.join(args.root, ".md5_rewrite_report.json")
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2)

    print(f"Scanned {report['files_scanned']} files.")
    print(f"Found {len(report['changes'])} files with MD5 usage.")
    if report["changes"]:
        print("Sample changes (first 20):")
        for c in report["changes"][:20]:
            print("-", c["file"], c["replacements"])
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
