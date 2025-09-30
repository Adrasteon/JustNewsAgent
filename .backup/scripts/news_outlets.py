"""Import news outlet list from markdown into justnews Postgres DB.

Usage:
  Set environment variables: JUSTNEWS_DB_HOST, JUSTNEWS_DB_PORT, JUSTNEWS_DB_NAME,
  JUSTNEWS_DB_USER, JUSTNEWS_DB_PASSWORD (or provide a DATABASE_URL).

  python scripts/news_outlets.py --file markdown_docs/agent_documentation/potential_news_sources.md
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterable

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import psycopg2.extras

from scripts.db_config import get_db_conn


def parse_markdown_table_rows(md: str) -> Iterable[tuple[str, str, str]]:
    """Yield (url, name, description) tuples by scanning markdown tables in the file.

    The file contains many tables with header: | URL | Name | Description |
    This function finds those tables and yields each data row.
    """
    lines = md.splitlines()
    i = 0
    header_re = re.compile(r"\|\s*URL\s*\|\s*Name\s*\|\s*Description\s*\|", re.I)
    while i < len(lines):
        line = lines[i]
        if header_re.search(line):
            # Next line should be the separator (| :--- | :--- | :--- |)
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = lines[i].strip()
                # Skip separator-like lines
                if re.match(r"^\|\s*:?-+", row):
                    i += 1
                    continue
                # Parse columns
                cols = [c.strip() for c in row.split("|")]
                # row.split yields leading/trailing empty strings, so filter
                cols = [c for c in cols if c != ""]
                if len(cols) >= 3:
                    url = cols[0]
                    name = cols[1]
                    desc = cols[2]
                    # cleanup backticks
                    url = url.strip(' `')
                    name = name.strip(' `')
                    desc = desc.strip(' `')
                    yield (url, name, desc)
                i += 1
        else:
            i += 1


def domain_from_url(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url)
    return m.group(1).lower() if m else url


UPSERT_SQL = """
WITH updated AS (
    UPDATE public.sources
    SET domain = %(domain)s,
        name = %(name)s,
        description = %(description)s,
        metadata = public.sources.metadata || %(metadata)s::jsonb,
        last_verified = now(),
        updated_at = now()
    WHERE lower(url) = lower(%(url)s)
    RETURNING id
),
inserted AS (
    INSERT INTO public.sources (url, domain, name, description, last_verified, metadata, updated_at)
    SELECT %(url)s, %(domain)s, %(name)s, %(description)s, now(), %(metadata)s::jsonb, now()
    WHERE NOT EXISTS (SELECT 1 FROM updated)
    RETURNING id
)
SELECT id FROM updated
UNION ALL
SELECT id FROM inserted;
"""


def upsert_outlets(rows: Iterable[tuple[str, str, str]], conn) -> list[int]:
    ids = []
    with conn.cursor() as cur:
        for url, name, desc in rows:
            domain = domain_from_url(url)
            metadata = {"source": "potential_news_sources.md"}
            cur.execute(UPSERT_SQL, {"url": url, "domain": domain, "name": name, "description": desc, "metadata": psycopg2.extras.Json(metadata)})
            try:
                row = cur.fetchone()
                if row:
                    ids.append(row[0])
            except psycopg2.ProgrammingError:
                # no row returned
                pass
        conn.commit()
    return ids


def create_provenance_mappings(conn, source_rows: list[tuple[str, str, str]]):
    """Attempt to map existing articles to sources by domain and insert into article_source_map.

    This is best-effort and should be run after articles table and sources table exist.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        # Build domain -> source_id map
        cur.execute("SELECT id, domain FROM public.sources")
        sources = cur.fetchall()
        domain_map = {s['domain'].lower(): s['id'] for s in sources if s['domain']}

        # Find articles with URLs matching known domains
        cur.execute("SELECT id, metadata FROM public.articles WHERE metadata->>'url' IS NOT NULL")
        articles = cur.fetchall()
        insert_sql = "INSERT INTO public.article_source_map (article_id, source_id, confidence, detected_at, metadata) VALUES (%s, %s, %s, now(), %s) ON CONFLICT DO NOTHING"
        for a in articles:
            meta = a['metadata'] or {}
            url = meta.get('url')
            if not url:
                continue
            domain = domain_from_url(url)
            sid = domain_map.get(domain)
            if sid:
                cur.execute(insert_sql, (a['id'], sid, 0.95, psycopg2.extras.Json({'matched_by': 'domain_match'})))
        conn.commit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", required=True, help="Path to potential_news_sources.md")
    parser.add_argument("--map-articles", action="store_true", help="Attempt to map existing articles to sources by domain and insert into article_source_map and update articles.source_id")
    parser.add_argument("--dry-run", action="store_true", help="Parse and show rows but do not write to DB")
    args = parser.parse_args(argv)

    path = args.file
    if not os.path.exists(path):
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    with open(path, encoding="utf-8") as fh:
        md = fh.read()

    rows = list(parse_markdown_table_rows(md))
    if not rows:
        print("No rows found in the markdown file.")
        return 1

    if args.dry_run:
        print(f"Dry run: found {len(rows)} rows")
        for r in rows[:20]:
            print(r)
        return 0

    conn = get_db_conn()
    try:
        ids = upsert_outlets(rows, conn)
        print(f"Upserted {len(ids)} sources")
        if args.map_articles:
            print("Creating provenance mappings by domain... this may take a while")
            create_provenance_mappings(conn, rows)
            print("Provenance mapping completed")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
