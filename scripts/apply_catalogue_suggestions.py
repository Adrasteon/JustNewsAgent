#!/usr/bin/env python3
"""
Apply description and tag suggestions from the description suggestions report
to the docs catalogue JSON (`docs/docs_catalogue_v2.json`).

This script is conservative: it only updates catalogue entries that match the
paths in the report and where the report explicitly lists issues such as
`description_too_short` or `no_tags`.

It writes a short report of which entries were updated and leaves a backup of
the original catalogue at `docs/docs_catalogue_v2.json.bak`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_json(p: Path) -> Dict[str, Any]:
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(p: Path, data: Dict[str, Any]) -> None:
    with p.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--report', required=True, help='Path to description suggestions report (JSON)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    ws = Path(args.workspace)
    report_path = Path(args.report)
    catalogue_path = ws / 'docs' / 'docs_catalogue_v2.json'

    if not report_path.exists():
        raise SystemExit(f"Report not found: {report_path}")
    if not catalogue_path.exists():
        raise SystemExit(f"Catalogue not found: {catalogue_path}")

    report = load_json(report_path)
    suggestions = report.get('suggestions', {})

    catalogue = load_json(catalogue_path)

    updates = []

    # Build a mapping from path -> (category_idx, doc_idx)
    path_map = {}
    for c_idx, cat in enumerate(catalogue.get('categories', [])):
        for d_idx, doc in enumerate(cat.get('documents', [])):
            path_map[doc.get('path')] = (c_idx, d_idx)

    for path, info in suggestions.items():
        if path not in path_map:
            # skip suggestions for files not present in the catalogue
            continue

        c_idx, d_idx = path_map[path]
        doc = catalogue['categories'][c_idx]['documents'][d_idx]

        changed = {}
        issues = info.get('issues', [])
        if 'description_too_short' in issues:
            new_desc = info.get('suggested_description')
            if new_desc and new_desc != doc.get('description', ''):
                changed['description'] = {'from': doc.get('description'), 'to': new_desc}
                doc['description'] = new_desc

        if 'no_tags' in issues:
            new_tags = info.get('suggested_tags', [])
            if new_tags and set(new_tags) != set(doc.get('tags', [])):
                changed['tags'] = {'from': doc.get('tags', []), 'to': new_tags}
                doc['tags'] = new_tags

        if changed:
            updates.append({'path': path, 'changes': changed})

    # Backup original
    backup_path = catalogue_path.with_suffix('.json.bak')
    if not args.dry_run:
        if not backup_path.exists():
            write_json(backup_path, catalogue)

    # If not dry-run, write updated catalogue
    if not args.dry_run:
        write_json(catalogue_path, catalogue)

    # Write a small updates report
    out_report = Path('reports/catalogue_update_report.json')
    out_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_report, {'updated_count': len(updates), 'updates': updates})

    print(f"Applied {len(updates)} updates to catalogue; report: {out_report}")


if __name__ == '__main__':
    main()
