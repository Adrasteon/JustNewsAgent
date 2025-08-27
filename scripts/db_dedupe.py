"""Small helper for DB-side dedupe using crawled_urls table.
Provides: ensure_table(conn) and register_url(conn, url, url_hash=None)
"""
from typing import Optional
import hashlib


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode('utf-8')).hexdigest()


def ensure_table(conn) -> None:
    """Ensure the crawled_urls table exists in the database."""
    with conn.cursor() as cur:
        cur.execute(open('scripts/migrations/0001_create_crawled_urls.sql', 'r').read())
    conn.commit()


def register_url(conn, url: str, url_hash: Optional[str] = None) -> bool:
    """Register URL in crawled_urls table. Returns True if URL was newly inserted, False if it already existed.

    This uses INSERT ... ON CONFLICT DO NOTHING and checks the rowcount.
    """
    if url_hash is None:
        url_hash = _hash_url(url)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO crawled_urls (url, url_hash, first_seen, last_seen)
            VALUES (%s, %s, now(), now())
            ON CONFLICT (url) DO UPDATE SET last_seen = EXCLUDED.last_seen
            RETURNING (xmax = 0) as inserted
            """,
            (url, url_hash),
        )
        row = cur.fetchone()
        conn.commit()
        # If RETURNING inserted is present, use it; otherwise conservatively return False
        if row is None:
            return False
        try:
            return bool(row[0])
        except Exception:
            return False
