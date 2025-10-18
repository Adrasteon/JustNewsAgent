#!/usr/bin/env python3
"""
Auto-fix deterministic links broken by moving tracked files into `archive/`.

This script scans markdown files under `markdown_docs` and `docs`, looks for
relative links that reference files we moved into `archive/` and updates the
path to point to the new location under `archive/`.

It only performs deterministic replacements where the target filename (basename)
uniquely matches a file under `archive/`. It writes files in-place and prints a
summary of changes. Use in the `maint/cleanup` branch.
"""
import re
import sys
from pathlib import Path
from collections import defaultdict


ROOT = Path(__file__).resolve().parent.parent
SEARCH_PATHS = [ROOT / 'markdown_docs', ROOT / 'docs']
ARCHIVE_DIR = ROOT / 'archive'


def build_archive_index():
    """Return mapping basename -> relative path in archive for MD files."""
    mapping = defaultdict(list)
    if not ARCHIVE_DIR.exists():
        return mapping
    for p in ARCHIVE_DIR.rglob('*.md'):
        mapping[p.name].append(p.relative_to(ROOT))
    return mapping


LINK_RE = re.compile(r"(?P<prefix>\[.*?\]\()(?P<path>[^)]+)(?P<suffix>\))")


def process_file(path: Path, archive_index):
    text = path.read_text(encoding='utf-8')
    changed = False
    replacements = []

    def repl(m):
        nonlocal changed
        p = m.group('path')
        # only handle simple relative links (no http(s)://, mailto:, #anchors only)
        if p.startswith('http://') or p.startswith('https://') or p.startswith('mailto:'):
            return m.group(0)
        if p.startswith('#'):
            return m.group(0)
        # strip any anchor
        anchor = ''
        if '#' in p:
            target_part, anchor = p.split('#', 1)
            anchor = '#'+anchor
        else:
            target_part = p
        basename = Path(target_part).name
        if basename in archive_index and len(archive_index[basename]) == 1:
            new_rel = archive_index[basename][0].as_posix()
            # preserve any anchor
            new_path = new_rel + (anchor or '')
            changed = True
            replacements.append((p, new_path))
            return m.group('prefix') + new_path + m.group('suffix')
        return m.group(0)

    new_text = LINK_RE.sub(repl, text)
    if changed:
        path.write_text(new_text, encoding='utf-8')
    return changed, replacements


def main():
    archive_index = build_archive_index()
    if not archive_index:
        print('No archive files found; nothing to do.')
        return 0

    total_changed = 0
    per_file_replacements = {}

    for base in SEARCH_PATHS:
        if not base.exists():
            continue
        for md in base.rglob('*.md'):
            changed, repls = process_file(md, archive_index)
            if changed:
                total_changed += 1
                per_file_replacements[str(md.relative_to(ROOT))] = repls

    print(f'Processed paths: {[str(p) for p in SEARCH_PATHS]}')
    print(f'Archive index entries: {sum(len(v) for v in archive_index.values())} files')
    print(f'Files changed: {total_changed}')
    for f, reps in per_file_replacements.items():
        print(f'-- {f}')
        for old, new in reps:
            print(f'   {old} -> {new}')

    if total_changed:
        print('\nNext: run `git add -A && git commit -m "chore(docs): auto-fix archive links"` to commit changes')
    return 0


if __name__ == '__main__':
    sys.exit(main())
