#!/usr/bin/env python3
"""Export a single article by id to a markdown file.
Usage: python3 scripts/export_article_md.py <article_id> [output_path]
"""

import sys

import psycopg2
from psycopg2.extras import RealDictCursor

DB = dict(
    dbname="justnews", user="justnews_user", password="password123", host="localhost"
)


def fetch_article(article_id: int):
    # current articles table columns: id, content (text), metadata (jsonb), created_at, embedding
    q = "SELECT id, content, metadata, created_at FROM articles WHERE id = %s"
    conn = psycopg2.connect(**DB)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, (article_id,))
            r = cur.fetchone()
            return r
    finally:
        conn.close()


def derive_title(article: dict) -> str:
    # Try common metadata fields first, then fallback to the first non-empty line of content
    meta = article.get("metadata") or {}
    for key in ("title", "headline", "headline_text", "name", "byline"):
        if isinstance(meta, dict) and key in meta and meta[key]:
            return str(meta[key])

    content = (article.get("content") or "").strip()
    if not content:
        return f"Article {article['id']}"
    # Use first line or first 120 chars as title fallback
    first_line = content.splitlines()[0].strip()
    if first_line and len(first_line) >= 8:
        return first_line if len(first_line) <= 140 else first_line[:137] + "..."
    return content[:140] + ("..." if len(content) > 140 else "")


def to_markdown(article: dict) -> str:
    title = derive_title(article)
    md = f"# {title}\n\n"
    meta = article.get("metadata") or {}
    if isinstance(meta, dict) and meta:
        # include a few helpful metadata fields if present
        if meta.get("byline"):
            md += f"**By:** {meta.get('byline')}\n\n"
        if meta.get("authors"):
            md += f"**Authors:** {meta.get('authors')}\n\n"
        if meta.get("url"):
            md += f"**Source:** {meta.get('url')}\n\n"
    if article.get("created_at"):
        md += f"**Stored at:** {article.get('created_at')}\n\n"
    md += "---\n\n"
    content = article.get("content") or ""
    md += content
    return md


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: export_article_md.py <article_id> [output_path]", file=sys.stderr)
        sys.exit(2)
    aid = int(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else f"logs/crawl/cluster1_rep_{aid}.md"

    article = fetch_article(aid)
    if not article:
        print(f"Article {aid} not found", file=sys.stderr)
        sys.exit(1)

    md = to_markdown(article)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(out)
