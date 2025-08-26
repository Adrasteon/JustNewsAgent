#!/usr/bin/env python3
"""Scan required Python files for referenced assets and identify latest docs.

Outputs:
 - /tmp/required_assets.txt
 - /tmp/include_manifest.tsv
 - /tmp/latest_docs.txt
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional


REPO = Path.cwd()
REQUIRED = Path('/tmp/required_files.txt')
OUT_ASSETS = Path('/tmp/required_assets.txt')
OUT_MANIFEST = Path('/tmp/include_manifest.tsv')
OUT_DOCS = Path('/tmp/latest_docs.txt')

KEEP_DIRS = {'models', 'common', 'training_system'}

# string literal matcher for likely asset references
# match single-line string literals only (no newlines), limit length to avoid pathological matches
PAT = re.compile(r'["\']([^"\'\n]{1,500})["\']', re.I)
KEYWORD_RE = re.compile(r'(models|common|docs|markdown_docs|\\.json|\\.md|\\.txt|\\.pt|\\.onnx)', re.I)


def gather_required_files() -> List[Path]:
    if not REQUIRED.exists():
        print('Missing /tmp/required_files.txt')
        return []
    return [REPO / p.strip() for p in REQUIRED.read_text().splitlines() if p.strip()]


def resolve_candidate(s: str) -> Optional[Path]:
    # avoid extremely long strings which may be code blobs or accidental captures
    if len(s) > 260:
        return None
    p = Path(s)
    try:
        if p.exists():
            return p
    except OSError:
        return None
    rp = REPO / s
    if rp.exists():
        return rp
    if s.startswith('./'):
        rp = REPO / s[2:]
        if rp.exists():
            return rp
    return None


def scan() -> None:
    required_files = gather_required_files()
    assets: set[Path] = set()
    manifest: dict[str, str] = {}

    for f in required_files:
        try:
            rel = str(f.relative_to(REPO))
        except Exception:
            rel = str(f)
        manifest[rel] = 'seed'
        try:
            txt = f.read_text(errors='ignore')
        except Exception:
            continue
        for m in PAT.finditer(txt):
            cand = m.group(1)
            # only consider candidates that contain our target keywords
            if not KEYWORD_RE.search(cand):
                continue
            resolved = resolve_candidate(cand)
            if resolved:
                assets.add(resolved.relative_to(REPO))
                manifest[str(resolved.relative_to(REPO))] = 'referenced'

    # include keep dirs
    for d in KEEP_DIRS:
        for p in REPO.glob(f'{d}/**'):
            if p.is_file():
                assets.add(p.relative_to(REPO))
                manifest[str(p.relative_to(REPO))] = 'keeproot'

    docs = list(REPO.glob('markdown_docs/**/*.md')) + list(REPO.glob('docs/**/*.md'))
    docs_sorted = sorted(docs, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    latest = [p.relative_to(REPO) for p in docs_sorted[:20]]

    OUT_ASSETS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_ASSETS.open('w') as fo:
        for p in sorted(assets):
            fo.write(str(p) + '\n')

    with OUT_MANIFEST.open('w') as fo:
        fo.write('path\treason\n')
        for k, v in sorted(manifest.items()):
            fo.write(f"{k}\t{v}\n")

    with OUT_DOCS.open('w') as fo:
        for p in latest:
            fo.write(str(p) + '\n')

    print('Wrote assets:', OUT_ASSETS, 'manifest:', OUT_MANIFEST, 'latest docs:', OUT_DOCS)


if __name__ == '__main__':
    scan()
