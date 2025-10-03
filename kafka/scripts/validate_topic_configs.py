#!/usr/bin/env python3
"""Validate topic config YAML files for naming and schema existence.

Usage:
  python kafka/scripts/validate_topic_configs.py kafka/config/topics/*.yaml

Checks:
 - Topic name valid: only lowercase letters, numbers, dots, hyphens and underscores
 - Partition and replication reasonable numbers
 - Referenced schema files exist under kafka/config/schemas (matching base name)
"""
from __future__ import annotations

import sys
import re
import os
import glob
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCHEMAS_DIR = os.path.join(ROOT, 'config', 'schemas')

TOPIC_NAME_RE = re.compile(r'^[a-z0-9_.-]+$')


def load_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def schema_exists(schema_base: str) -> bool:
    # Accept files with suffix like _v1.avsc or _v2.json etc.
    for f in os.listdir(SCHEMAS_DIR):
        if f.startswith(schema_base) and (f.endswith('.avsc') or f.endswith('.json')):
            return True
    return False


def validate_topic_config(path: str) -> int:
    data = load_yaml(path)
    failures = 0
    for t in data.get('topics', []):
        name = t.get('name')
        if not name or not TOPIC_NAME_RE.match(name):
            print(f"INVALID TOPIC NAME: {name} in {path}")
            failures += 1
        partitions = t.get('partitions', 1)
        replication = t.get('replication', 1)
        if not isinstance(partitions, int) or partitions < 1 or partitions > 1000:
            print(f"INVALID PARTITIONS for {name}: {partitions}")
            failures += 1
        if not isinstance(replication, int) or replication < 1 or replication > 5:
            print(f"INVALID REPLICATION for {name}: {replication}")
            failures += 1
        schema = t.get('schema')
        if schema and not schema_exists(schema):
            print(f"MISSING SCHEMA: {schema} referenced by topic {name} (checked under {SCHEMAS_DIR})")
            failures += 1
    return failures


if __name__ == '__main__':
    paths = sys.argv[1:] or glob.glob(os.path.join(ROOT, 'config', 'topics', '*.yaml'))
    total_fail = 0
    for p in paths:
        print(f"Validating topic config: {p}")
        total_fail += validate_topic_config(p)
    if total_fail:
        print(f"Validation failed: {total_fail} issues found")
        sys.exit(2)
    print('All topic configs validated successfully')
    sys.exit(0)
