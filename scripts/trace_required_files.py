#!/usr/bin/env python3
"""Static import tracer: starting from seed files, parse imports to collect a set of required files.
It performs static parsing of import statements and attempts to map them to local files under the repo.
It is conservative: if an import cannot be resolved to a local file, it's ignored (assumed external).

Output: /tmp/required_files.txt (one path per line, repository-relative)
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path.cwd()
OUT = Path('/tmp/required_files.txt')

seed_patterns = [
    'agents/*/main.py',
    'mcp_bus/**',
    'tests/**',
]


def find_seed_files() -> list[Path]:
    files = []
    for pat in seed_patterns:
        for p in REPO_ROOT.glob(pat):
            if p.is_file():
                files.append(p.relative_to(REPO_ROOT))
    return sorted(files)


def parse_imports(path: Path) -> set[str]:
    code = path.read_text(errors='ignore')
    tree = ast.parse(code, filename=str(path))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def module_name_to_paths(modname: str) -> list[Path]:
    # map module names like agents.balancer.tools to possible file paths
    parts = modname.split('.')
    candidates = []
    # try package module path
    file1 = REPO_ROOT.joinpath(*parts).with_suffix('.py')
    if file1.exists():
        candidates.append(file1.relative_to(REPO_ROOT))
    # try package __init__.py
    file2 = REPO_ROOT.joinpath(*parts, '__init__.py')
    if file2.exists():
        candidates.append(file2.relative_to(REPO_ROOT))
    return candidates


def trace():
    seeds = find_seed_files()
    print('Seeds:', len(seeds))
    required: set[Path] = {Path(s) for s in seeds}
    queue = list(required)
    seen_modules = set()

    while queue:
        p = queue.pop()
        try:
            imports = parse_imports(p)
        except Exception as e:
            print('Parse error', p, e)
            continue
        for mod in imports:
            if mod in seen_modules:
                continue
            seen_modules.add(mod)
            for cand in module_name_to_paths(mod):
                candp = Path(cand)
                if candp not in required:
                    required.add(candp)
                    queue.append(candp)
    # write output
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w') as fo:
        for p in sorted(required):
            fo.write(str(p) + '\n')
    print('Wrote', OUT, 'entries:', len(required))


if __name__ == '__main__':
    trace()
