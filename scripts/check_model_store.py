#!/usr/bin/env python3
"""Audit the MODEL_STORE_ROOT against markdown_docs/agent_documentation/AGENT_MODEL_MAP.json

Outputs a JSON report to stdout with keys:
 - required: mapping agent -> list of required model ids
 - present: mapping agent -> list of model directories found under v1
 - missing: mapping agent -> list of required models not present
 - extra_agents: agents present on disk but not in the map
 - extra_models: mapping agent -> list of model directories present but not in the map

This script does not modify the filesystem.
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = REPO_ROOT / 'markdown_docs' / 'agent_documentation' / 'AGENT_MODEL_MAP.json'
MODEL_STORE_ROOT = Path('/media/adra/Data/justnews/model_store')


def read_map():
    with open(MAP_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_models_on_disk():
    out = {}
    if not MODEL_STORE_ROOT.exists():
        return out
    for agent_dir in MODEL_STORE_ROOT.iterdir():
        if not agent_dir.is_dir():
            continue
        # Look for v1 under agent
        v1 = agent_dir / 'v1'
        if v1.exists() and v1.is_dir():
            # list immediate children in v1
            children = [p.name for p in v1.iterdir() if p.is_dir()]
            out[agent_dir.name] = sorted(children)
        else:
            # Maybe the agent folder itself contains model dirs (older layout)
            children = [p.name for p in agent_dir.iterdir() if p.is_dir() and p.name != 'current']
            out[agent_dir.name] = sorted(children)
    return out


def audit():
    required = read_map()
    present = list_models_on_disk()

    report = {
        'required': required,
        'present': present,
        'missing': {},
        'extra_agents': [],
        'extra_models': {},
    }

    # Check each required agent
    for agent, models in required.items():
        disk_models = present.get(agent, [])
        # For matching, allow model ids like 'google/bert...' -> directory last segment sometimes
        # We'll try two checks: exact model id match, and last-path-segment match
        def model_matches_on_disk(model_id, disk_list):
            if model_id in disk_list:
                return True
            tail = model_id.split('/')[-1]
            return tail in disk_list

        missing = []
        for m in models:
            if not model_matches_on_disk(m, disk_models):
                missing.append(m)
        report['missing'][agent] = missing

        # Extra models on disk for this agent
        extras = []
        allowed = set([m for m in models] + [m.split('/')[-1] for m in models])
        for dm in disk_models:
            if dm not in allowed:
                extras.append(dm)
        if extras:
            report['extra_models'][agent] = extras

    # Agents on disk but not in required map
    for agent in present.keys():
        if agent not in required:
            report['extra_agents'].append(agent)

    return report


if __name__ == '__main__':
    r = audit()
    json.dump(r, sys.stdout, indent=2)
    sys.stdout.write('\n')
