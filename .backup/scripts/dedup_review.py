#!/usr/bin/env python3
"""Produce a human-review CSV from dedup clusters and article titles in DB.

Reads:
 - logs/crawl/dedup_clusters.tsv
Queries Postgres for up to 5 titles per cluster member and writes CSV:
 - logs/crawl/dedup_review.csv

Columns: cluster_id,size,representative_id,representative_title,member_ids,sample_member_titles
"""
from __future__ import annotations

import csv
import os
import sys

import psycopg2

CLUSTERS = 'logs/crawl/dedup_clusters.tsv'
OUT = 'logs/crawl/dedup_review.csv'

DB_PARAMS = {
    'host': 'localhost',
    'dbname': 'justnews',
    'user': 'justnews_user',
    'password': 'password123'
}

if not os.path.exists(CLUSTERS):
    print('clusters file not found:', CLUSTERS, file=sys.stderr)
    sys.exit(1)

# load clusters
clusters = []
with open(CLUSTERS, encoding='utf-8') as f:
    header = f.readline()
    for line in f:
        line = line.rstrip('\n')
        parts = line.split('\t')
        if len(parts) < 5:
            continue
        cluster_id = int(parts[0])
        # convert member ids to integers for DB queries
        ids = [int(x) for x in parts[1].split(',')] if parts[1] else []
        size = int(parts[2])
        rep_id = int(parts[3])
        rep_title = parts[4]
        clusters.append((cluster_id, size, rep_id, rep_title, ids))

# open DB
conn = psycopg2.connect(host=DB_PARAMS['host'], dbname=DB_PARAMS['dbname'], user=DB_PARAMS['user'], password=DB_PARAMS['password'])
cur = conn.cursor()

with open(OUT, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['cluster_id','size','representative_id','representative_title','member_ids','sample_member_titles'])
    for cluster_id, size, rep_id, rep_title, ids in clusters:
        sample_titles: list[str] = []
        # sample up to 5 member titles
        sample_ids = ids[:5]
        if sample_ids:
            # avoid array binding issues by building an IN-list of placeholders
            placeholders = ','.join(['%s'] * len(sample_ids))
            q = f"SELECT id, COALESCE(metadata->>'title', metadata->>'headline', split_part(content, '\n',1)) AS title FROM articles WHERE id IN ({placeholders})"
            cur.execute(q, tuple(sample_ids))
            rows = cur.fetchall()
            # preserve order of sample_ids
            title_map = {r[0]: (r[1] or '') for r in rows}
            for sid in sample_ids:
                sample_titles.append(title_map.get(sid, ''))
        writer.writerow([cluster_id, size, rep_id, rep_title, ','.join(map(str, ids)), ' | '.join(sample_titles)])

cur.close()
conn.close()
print('wrote', OUT, file=sys.stderr)
