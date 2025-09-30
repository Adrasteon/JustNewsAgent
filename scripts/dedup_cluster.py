#!/usr/bin/env python3
"""Collapse fuzzy pair edges into connected components (union-find).

Input: logs/crawl/dedup_fuzzy.txt (tab-separated: id1\tid2\tscore\ttitle1\ttitle2)
Outputs:
 - logs/crawl/dedup_clusters.tsv: cluster_id\tids(comma)\tsize\trepresentative_id\trepresentative_title
 - logs/crawl/dedup_summary.txt: overall stats
 - logs/crawl/dedup_sample.txt: sample of top clusters

This script is lightweight, pure-Python and works on large pair lists.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict

IN = "logs/crawl/dedup_fuzzy.txt"
OUT_CLUSTERS = "logs/crawl/dedup_clusters.tsv"
OUT_SUMMARY = "logs/crawl/dedup_summary.txt"
OUT_SAMPLE = "logs/crawl/dedup_sample.txt"

# Simple union-find
class UnionFind:
    def __init__(self):
        self.parent: dict[int,int] = {}
        self.size: dict[int,int] = {}
    def find(self, a: int) -> int:
        if a not in self.parent:
            self.parent[a] = a
            self.size[a] = 1
            return a
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a
    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        # attach smaller to larger
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] = self.size.get(ra,1) + self.size.get(rb,1)


def main():
    uf = UnionFind()
    titles: dict[int,str] = {}
    total_pairs = 0
    seen_ids = set()

    print("reading edges from", IN, file=sys.stderr)
    with open(IN, encoding="utf-8") as f:
        header = f.readline()
        if not header:
            print("empty file", file=sys.stderr)
            return
        # header line may start with 'FUZZY' - skip until data
        # we'll assume data lines start with digits; process lines with at least two tabs
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                id1 = int(parts[0])
                id2 = int(parts[1])
            except ValueError:
                continue
            total_pairs += 1
            seen_ids.add(id1)
            seen_ids.add(id2)
            uf.union(id1, id2)
            # store titles if present
            if len(parts) >= 4:
                # title columns might be long; keep first appearance
                if id1 not in titles and parts[3].strip():
                    titles[id1] = parts[3].strip()
                if id2 not in titles and len(parts) >= 5 and parts[4].strip():
                    titles[id2] = parts[4].strip()

    print(f"read {total_pairs} pairs, {len(seen_ids)} unique ids", file=sys.stderr)

    # group by root
    groups: dict[int, list[int]] = defaultdict(list)
    for id_ in seen_ids:
        root = uf.find(id_)
        groups[root].append(id_)

    # create canonical cluster ids (sorted by size desc)
    clusters = sorted(groups.items(), key=lambda kv: -len(kv[1]))

    with open(OUT_CLUSTERS, "w", encoding="utf-8") as out:
        out.write("cluster_id\tids\tsize\trepresentative_id\trepresentative_title\n")
        for idx, (root, members) in enumerate(clusters, start=1):
            members_sorted = sorted(members)
            size = len(members_sorted)
            rep = members_sorted[0]
            rep_title = titles.get(rep, "")
            out.write(f"{idx}\t{','.join(map(str,members_sorted))}\t{size}\t{rep}\t{rep_title}\n")

    # summary
    sizes = [len(m) for _,m in clusters]
    num_clusters = len(sizes)
    multi = sum(1 for s in sizes if s>1)
    max_size = max(sizes) if sizes else 0
    top_counts = Counter(sizes).most_common(10)

    with open(OUT_SUMMARY, "w", encoding="utf-8") as s:
        s.write("dedup clustering summary\n")
        s.write(f"pairs_read\t{total_pairs}\n")
        s.write(f"unique_ids_in_pairs\t{len(seen_ids)}\n")
        s.write(f"num_clusters\t{num_clusters}\n")
        s.write(f"multi_item_clusters\t{multi}\n")
        s.write(f"max_cluster_size\t{max_size}\n")
        s.write("top size counts (size:count)\n")
        for size,count in top_counts:
            s.write(f"{size}:{count}\n")

    # sample top clusters
    with open(OUT_SAMPLE, "w", encoding="utf-8") as samp:
        samp.write("# top 50 clusters (cluster_id\tsize\trepresentative_id\trepresentative_title\tmembers...)\n")
        for idx, (root,members) in enumerate(clusters[:50], start=1):
            members_sorted = sorted(members)
            rep = members_sorted[0]
            rep_title = titles.get(rep, "")
            samp.write(f"{idx}\t{len(members_sorted)}\t{rep}\t{rep_title}\t{','.join(map(str,members_sorted[:30]))}\n")

    print("wrote", OUT_CLUSTERS, OUT_SUMMARY, OUT_SAMPLE, file=sys.stderr)

if __name__ == '__main__':
    main()
