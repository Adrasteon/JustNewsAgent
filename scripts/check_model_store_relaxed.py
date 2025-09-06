#!/usr/bin/env python3
"""Relaxed audit: show whether required models are actually present using common HF directory patterns.

Outputs JSON with per-agent lists showing how each required model was found:
 - found_by: one of [exact, tail, pattern, substring, none]
 - disk_entry: the disk dir that matched (if any)

Also lists agents present on disk, unexpected model dirs, and summary counts.
"""
import json
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = REPO_ROOT / 'markdown_docs' / 'agent_documentation' / 'AGENT_MODEL_MAP.json'
MODEL_STORE_ROOT = Path('/media/adra/Data/justnews/model_store')


def read_map():
    return json.loads(MAP_PATH.read_text(encoding='utf-8'))


def list_disk():
    present = {}
    if not MODEL_STORE_ROOT.exists():
        return present
    for agent_dir in sorted(MODEL_STORE_ROOT.iterdir()):
        if not agent_dir.is_dir():
            continue
        v1 = agent_dir / 'v1'
        children = []
        if v1.exists() and v1.is_dir():
            children = [p.name for p in sorted(v1.iterdir()) if p.is_dir()]
        else:
            children = [p.name for p in sorted(agent_dir.iterdir()) if p.is_dir() and p.name != 'current']
        present[agent_dir.name] = children
    return present


def match_model_on_disk(model_id, disk_entries):
    # Attempt several matching heuristics
    tail = model_id.split('/')[-1]
    org = model_id.split('/')[0] if '/' in model_id else None

    for d in disk_entries:
        # exact match
        if d == model_id:
            return ('exact', d)
    for d in disk_entries:
        if d == tail:
            return ('tail', d)
    # huggingface_hub snapshot pattern: models--org--name or models--org--name--revision
    pattern = None
    if org:
        # escape org and tail for regex
        pattern = re.compile(rf"models--{re.escape(org)}--{re.escape(tail)}($|--)" )
        for d in disk_entries:
            if pattern.search(d):
                return ('pattern', d)
    # substring anywhere
    for d in disk_entries:
        if tail in d or model_id.replace('/', '--') in d:
            return ('substring', d)
    return ('none', None)


def main():
    mapping = read_map()
    present = list_disk()
    report = {
        'checked_at': None,
        'summary': {},
        'per_agent': {},
        'agents_on_disk': sorted(list(present.keys()))
    }

    total_required = 0
    total_found = 0

    for agent, required_models in mapping.items():
        disk_entries = present.get(agent, [])
        agent_report = []
        found_count = 0
        for m in required_models:
            total_required += 1
            found_by, disk = match_model_on_disk(m, disk_entries)
            if found_by != 'none':
                found_count += 1
                total_found += 1
            agent_report.append({'model': m, 'found_by': found_by, 'disk_entry': disk})
        report['per_agent'][agent] = {
            'required_count': len(required_models),
            'found_count': found_count,
            'details': agent_report,
            'disk_entries': disk_entries
        }

    report['summary'] = {'total_required': total_required, 'total_found': total_found}
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
