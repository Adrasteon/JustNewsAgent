#!/usr/bin/env python3
"""
Generate one-line description and tag suggestions for catalogue quality issues.

This script scans the existing docs catalogue and for any document with a
short description (under 50 chars) or missing tags it will produce a
suggested one-line description and a small set of candidate tags. The output
is written to a JSON report for maintainers to review. By default it does
not modify any files; an --apply flag can be used to write suggestions into
markdown frontmatter (opt-in).

Usage:
    python3 scripts/generate_description_suggestions.py --workspace /home/adra/JustNewsAgent --report reports/description_suggestions.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


def load_catalogue(catalogue_path: Path) -> Dict:
    with catalogue_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def read_markdown_first_paragraph(md_path: Path) -> str:
    if not md_path.exists():
        return ""

    text = md_path.read_text(encoding='utf-8')

    # Strip YAML frontmatter
    if text.startswith('---'):
        parts = text.split('---', 2)
        if len(parts) == 3:
            text = parts[2]

    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.S)

    # Find first non-empty paragraph
    for paragraph in re.split(r'\n\s*\n', text):
        cleaned = paragraph.strip()
        # Skip headings and short lines
        if not cleaned:
            continue
        if cleaned.startswith('#'):
            continue
        # Remove markdown links inline text
        cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)
        # Collapse whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        if len(cleaned) >= 30:
            return cleaned

    return ''


STOPWORDS = {
    'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'using',
    'are', 'was', 'were', 'have', 'has', 'but', 'not', 'our', 'its', 'can'
}


def suggest_tags(title: str, paragraph: str, max_tags: int = 3) -> List[str]:
    # Simple heuristic: take words from title + paragraph, filter stopwords
    tokens = re.findall(r"[A-Za-z]{4,}", (title + ' ' + paragraph).lower())
    freq: Dict[str, int] = {}
    for t in tokens:
        if t in STOPWORDS:
            continue
        freq[t] = freq.get(t, 0) + 1

    # Sort by frequency then alphabetically
    candidates = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in candidates[:max_tags]]


def build_suggestion(title: str, paragraph: str) -> str:
    if paragraph:
        # Return a condensed first-sentence style suggestion
        # Split into sentences
        sentences = re.split(r'[\.\!?]\s+', paragraph)
        first = sentences[0].strip()
        if len(first) > 140:
            first = first[:137].rstrip() + '...'
        return first

    # Fallback
    if title:
        return f"Overview of {title}. Expand with key details and examples."
    return "Short overview — expand with context and examples."


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate one-line description and tag suggestions')
    parser.add_argument('--workspace', required=True, help='Workspace root')
    parser.add_argument('--report', default='reports/description_suggestions.json', help='Report file path')
    parser.add_argument('--apply', action='store_true', help='Apply suggestions directly to files (opt-in)')
    args = parser.parse_args()

    ws = Path(args.workspace)
    catalogue_path = ws / 'docs' / 'docs_catalogue_v2.json'
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    catalogue = load_catalogue(catalogue_path)

    suggestions: Dict[str, Dict] = {}

    for category in catalogue.get('categories', []):
        for doc in category.get('documents', []):
            path = doc.get('path')
            title = doc.get('title', '')
            description = doc.get('description', '') or ''
            tags = doc.get('tags') or []

            issues = []
            if len(description) < 50:
                issues.append('description_too_short')
            if not tags:
                issues.append('no_tags')

            if not issues:
                continue

            md_path = ws / path
            paragraph = read_markdown_first_paragraph(md_path)
            suggested_description = build_suggestion(title, paragraph)
            suggested_tags = suggest_tags(title, paragraph)

            suggestions[path] = {
                'title': title,
                'issues': issues,
                'current_description': description,
                'suggested_description': suggested_description,
                'current_tags': tags,
                'suggested_tags': suggested_tags
            }

            if args.apply:
                # Opt-in: write suggestion into frontmatter if frontmatter exists
                if md_path.exists():
                    text = md_path.read_text(encoding='utf-8')
                    # Ensure we have a frontmatter block
                    if text.startswith('---'):
                        parts = text.split('---', 2)
                        if len(parts) == 3:
                            fm_raw = parts[1]
                            body = parts[2]
                            # Replace or insert description and tags
                            fm = fm_raw
                            if 'description:' in fm:
                                fm = re.sub(r"description:.*", f"description: '{suggested_description}'", fm)
                            else:
                                fm = fm + f"\ndescription: '{suggested_description}'\n"

                            if 'tags:' in fm:
                                # naive replace
                                fm = re.sub(r"tags:.*", f"tags: {json.dumps(suggested_tags)}", fm)
                            else:
                                fm = fm + f"\ntags: {json.dumps(suggested_tags)}\n"

                            new_text = '---' + fm + '---' + body
                            md_path.write_text(new_text, encoding='utf-8')
                    else:
                        # No frontmatter: create one
                        fm = '---\n'
                        fm += f"title: '{title}'\n"
                        fm += f"description: '{suggested_description}'\n"
                        fm += f"tags: {json.dumps(suggested_tags)}\n"
                        fm += '---\n\n'
                        new_text = fm + text
                        md_path.write_text(new_text, encoding='utf-8')

    # Write report
    with report_path.open('w', encoding='utf-8') as rf:
        json.dump({'generated_at': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
                   'workspace': str(ws),
                   'suggestions': suggestions}, rf, indent=2, ensure_ascii=False)

    print(f"Wrote suggestions for {len(suggestions)} documents to {report_path}")


if __name__ == '__main__':
    main()
