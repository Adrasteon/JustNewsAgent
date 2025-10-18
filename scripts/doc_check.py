#!/usr/bin/env python3
"""Documentation checker

Checks markdown files for:
- broken internal links (file existence)
- missing anchors (links to headings)
- missing image files
- stub/placeholder files (small or containing TODO/TBD)

Usage:
  python3 scripts/doc_check.py --paths markdown_docs docs

This script avoids network checks by default. Use --check-external to validate
http/https links (may be slow).
"""
from __future__ import annotations
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


LINK_RE = re.compile(r"!\[.*?\]\((.*?)\)|\[.*?\]\((.*?)\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
EXTERNAL_SCHEMES = ("http://", "https://", "mailto:")


def slugify(text: str) -> str:
    """Approximate GitHub-style anchor slugification."""
    s = text.strip().lower()
    # remove anything that's not alnum, space, or hyphen
    s = re.sub(r"[^a-z0-9 \-]", "", s)
    s = s.replace(" ", "-")
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def extract_headings(path: Path) -> Set[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return set()
    headings = set()
    for m in HEADING_RE.finditer(text):
        anchor = slugify(m.group(1))
        if anchor:
            headings.add(anchor)
    return headings


def resolve_target(source: Path, link: str) -> Tuple[str, str]:
    """Resolve a link into (type, normalized_target).
    type: 'external', 'anchor', 'file'
    normalized_target: for file -> absolute Path str, for anchor -> (file, anchor)
    """
    if any(link.startswith(s) for s in EXTERNAL_SCHEMES):
        return "external", link
    # possible anchor-only link: '#section'
    if link.startswith("#"):
        return "anchor", (str(source), link.lstrip("#"))
    # split anchor from file: file.md#section
    if "#" in link:
        filepart, anchor = link.split("#", 1)
    else:
        filepart, anchor = link, None
    # resolve relative paths
    target_path = (source.parent / filepart).resolve()
    return "file", (str(target_path), anchor)  # anchor may be None


def is_probable_stub(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    if len(text.strip()) < 120:
        return True
    if re.search(r"\b(TODO|TBD|PLACEHOLDER|stub|to be written)\b", text, re.I):
        return True
    return False


def check_paths(paths: List[Path], check_external: bool = False) -> int:
    md_files: List[Path] = []
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".md":
            md_files.append(p)
        elif p.is_dir():
            for root, _, files in os.walk(p):
                for f in files:
                    if f.lower().endswith(".md"):
                        md_files.append(Path(root) / f)

    print(f"Found {len(md_files)} markdown files to check")

    # cache headings for files we encounter
    headings_cache: Dict[str, Set[str]] = {}
    broken_links: List[Tuple[Path, str]] = []
    missing_anchors: List[Tuple[Path, str, str]] = []
    missing_images: List[Tuple[Path, str]] = []
    stubs: List[Path] = []

    for md in md_files:
        # stub detection
        if is_probable_stub(md):
            stubs.append(md)

        try:
            text = md.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading {md}: {e}")
            continue

        for m in LINK_RE.finditer(text):
            link = m.group(1) or m.group(2)
            if not link:
                continue
            typ, target = resolve_target(md, link)
            if typ == "external":
                if check_external:
                    # simple network check
                    try:
                        import requests

                        r = requests.head(target, allow_redirects=True, timeout=6)
                        if r.status_code >= 400:
                            broken_links.append((md, link))
                    except Exception:
                        broken_links.append((md, link))
                continue
            # file or anchor
            tgt_path_str, anchor = target
            tgt_path = Path(tgt_path_str)
            # try common variants: with .md appended if missing
            exists = False
            candidates = [tgt_path]
            if tgt_path.suffix == "":
                candidates.append(tgt_path.with_suffix('.md'))
                candidates.append(tgt_path / 'README.md')
            for c in candidates:
                if c.exists():
                    exists = True
                    tgt_path = c
                    break

            if not exists:
                # treat links to anchors within same file (e.g., README.md#foo) when file omitted
                broken_links.append((md, link))
                continue

            if anchor:
                key = str(tgt_path)
                if key not in headings_cache:
                    headings_cache[key] = extract_headings(tgt_path)
                slug = slugify(anchor)
                if slug not in headings_cache[key]:
                    missing_anchors.append((md, str(tgt_path), anchor))

            # detect images
            if m.group(1):
                # an image
                if not tgt_path.exists():
                    missing_images.append((md, link))

    # Report
    def report_list(title: str, items: List, formatter):
        if not items:
            print(f"OK: {title}: 0 issues")
            return
        print(f"WARNING: {title}: {len(items)}")
        for it in items:
            print(formatter(it))

    report_list("Broken internal links", broken_links, lambda x: f"{x[0]} -> {x[1]}")
    report_list("Missing anchors", missing_anchors, lambda x: f"{x[0]} -> {x[1]}#{x[2]}")
    report_list("Missing images", missing_images, lambda x: f"{x[0]} -> {x[1]}")
    if stubs:
        print(f"POTENTIAL STUBS: {len(stubs)} files")
        for s in stubs:
            print(f"- {s}")
    else:
        print("No obvious stubs detected")

    # exit code non-zero if problems found
    problems = len(broken_links) + len(missing_anchors) + len(missing_images)
    return 1 if problems else 0


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", nargs="+", required=True, help="Paths to check (directories or files)")
    p.add_argument("--check-external", action="store_true", help="Check external http/https links (slow)")
    args = p.parse_args(argv)

    paths = [Path(p) for p in args.paths]
    return check_paths(paths, check_external=args.check_external)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
