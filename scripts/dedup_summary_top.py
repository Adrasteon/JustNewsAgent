#!/usr/bin/env python3
import csv
import sys

path = "logs/crawl/dedup_review.csv"
try:
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
except Exception as e:
    print("ERROR reading CSV:", e, file=sys.stderr)
    sys.exit(1)

rows.sort(key=lambda x: int(x["size"]), reverse=True)

def preview(s: str) -> str:
    s = s.strip().replace("\n", " ").replace('"', "'")
    return s if len(s) <= 120 else s[:117] + "..."

print("cluster_id,size,rep_id,rep_title_preview")
for row in rows[:20]:
    print(f'{row["cluster_id"]},{row["size"]},{row["representative_id"]},"{preview(row["representative_title"]) }"')
