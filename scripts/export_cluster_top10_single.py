#!/usr/bin/env python3
"""Export first N members of a cluster into a single markdown file with separators.
Usage: python3 scripts/export_cluster_top10_single.py <cluster_id> [count] [outpath]
"""

import csv
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

DB = dict(
    dbname="justnews", user="justnews_user", password="password123", host="localhost"
)
CSV_PATH = Path("logs/crawl/dedup_review.csv")


def derive_title_from_meta(article: dict) -> str:
    meta = article.get("metadata") or {}
    for key in ("title", "headline", "headline_text", "name", "byline"):
        if isinstance(meta, dict) and key in meta and meta[key]:
            return str(meta[key])
    content = (article.get("content") or "").strip()
    if not content:
        return f"Article {article['id']}"
    first_line = content.splitlines()[0].strip()
    if first_line and len(first_line) >= 8:
        return first_line if len(first_line) <= 140 else first_line[:137] + "..."
    return content[:140] + ("..." if len(content) > 140 else "")


def fetch_article(article_id: int):
    q = "SELECT id, content, metadata, created_at FROM articles WHERE id = %s"
    conn = psycopg2.connect(**DB)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, (article_id,))
            return cur.fetchone()
    finally:
        conn.close()


def article_to_markdown(article: dict) -> str:
    title = derive_title_from_meta(article)
    parts = []
    parts.append(f"# {title}")
    meta = article.get("metadata") or {}
    if isinstance(meta, dict) and meta:
        if meta.get("byline"):
            parts.append(f"**By:** {meta.get('byline')}")
        if meta.get("authors"):
            parts.append(f"**Authors:** {meta.get('authors')}")
        if meta.get("url"):
            parts.append(f"**Source:** {meta.get('url')}")
    if article.get("created_at"):
        parts.append(f"**Stored at:** {article.get('created_at')}")
    parts.append("---")
    content = article.get("content") or ""
    parts.append(content)
    return "\n\n".join(parts)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: export_cluster_top10_single.py <cluster_id> [count] [outpath]",
            file=sys.stderr,
        )
        sys.exit(2)
    cluster_id = str(sys.argv[1])
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    outpath = (
        Path(sys.argv[3]) if len(sys.argv) > 3 else Path("logs/crawl/cluster1_top10.md")
    )

    if not CSV_PATH.exists():
        print("Review CSV not found:", CSV_PATH, file=sys.stderr)
        sys.exit(1)

    with open(CSV_PATH, encoding="utf-8") as f:
        r = csv.DictReader(f)
        cluster_row = None
        for row in r:
            if row["cluster_id"] == cluster_id:
                cluster_row = row
                break

    if cluster_row is None:
        print("Cluster", cluster_id, "not found in", CSV_PATH, file=sys.stderr)
        sys.exit(1)

    member_ids = [int(x) for x in cluster_row["member_ids"].split(",") if x.strip()]
    selected = member_ids[:count]

    sections = []
    for aid in selected:
        art = fetch_article(aid)
        if not art:
            sections.append(f"# Article {aid}\n\n*ERROR: article not found in DB*")
            continue
        sections.append(article_to_markdown(art))

    # join with two blank lines
    content = "\n\n\n".join(sections)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(content)
    print(outpath)
