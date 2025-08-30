#!/usr/bin/env python3
"""Export members of a cluster to markdown files.
Usage: python3 scripts/export_cluster_members.py <cluster_id> [count]

This script reads logs/crawl/dedup_review.csv to locate the cluster and member ids.
It imports the fetch_article/to_markdown logic from export_article_md.py by executing it as a module.
"""
import csv, sys, subprocess
from pathlib import Path

CSV_PATH = Path('logs/crawl/dedup_review.csv')
OUT_DIR = Path('logs/crawl')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: export_cluster_members.py <cluster_id> [count]', file=sys.stderr)
        sys.exit(2)
    cid = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    if not CSV_PATH.exists():
        print('Review CSV not found:', CSV_PATH, file=sys.stderr)
        sys.exit(1)

    with open(CSV_PATH, encoding='utf-8') as f:
        r = csv.DictReader(f)
        cluster_row = None
        for row in r:
            if row['cluster_id'] == str(cid):
                cluster_row = row
                break

    if cluster_row is None:
        print('Cluster', cid, 'not found in', CSV_PATH, file=sys.stderr)
        sys.exit(1)

    member_ids = cluster_row['member_ids'].split(',')
    member_ids = [int(x) for x in member_ids if x.strip()]
    to_export = member_ids[:count]

    created = []
    for aid in to_export:
        outpath = OUT_DIR / f'cluster1_member_{aid}.md'
        # call the existing exporter script to keep logic consistent
        res = subprocess.run(['python3','scripts/export_article_md.py',str(aid),str(outpath)], capture_output=True, text=True)
        if res.returncode != 0:
            print(f'Failed to export {aid}:', res.stderr, file=sys.stderr)
        else:
            created.append(res.stdout.strip())

    for p in created:
        print(p)
