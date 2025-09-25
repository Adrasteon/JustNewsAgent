"""Backfill and maintenance scripts for sources and article mappings.

1. Computes and stores url_hash on sources (SHA256) for fast lookups.
2. Creates functional indexes for domains and lower(domain).
3. Backfills articles.source_id from the highest-confidence mapping in article_source_map.
"""
from __future__ import annotations

import hashlib

from scripts.db_config import get_db_conn


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ensure_columns_and_indexes(conn):
    with conn.cursor() as cur:
        # Add url_hash column if missing
        cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='sources' AND column_name='url_hash') THEN
                ALTER TABLE public.sources ADD COLUMN url_hash TEXT;
            END IF;
        END$$;
        """)

        # Create index on domain lower(domain)
        cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'sources_domain_lower_idx') THEN
                CREATE INDEX sources_domain_lower_idx ON public.sources (lower(domain));
            END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'sources_url_hash_idx') THEN
                CREATE INDEX sources_url_hash_idx ON public.sources (url_hash);
            END IF;
        END$$;
        """)
        conn.commit()


def populate_url_hash(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id, url FROM public.sources WHERE url_hash IS NULL OR url_hash = ''")
        rows = cur.fetchall()
        for id_, url in rows:
            cur.execute("UPDATE public.sources SET url_hash = %s WHERE id = %s", (sha256_hex(url), id_))
        conn.commit()


def backfill_articles_source_id(conn):
    with conn.cursor() as cur:
        # For each article, pick the highest-confidence mapping
        cur.execute("""
        WITH ranked AS (
          SELECT article_id, source_id,
            ROW_NUMBER() OVER (PARTITION BY article_id ORDER BY confidence DESC, detected_at DESC) AS rn
          FROM public.article_source_map
        )
        UPDATE public.articles a
        SET source_id = r.source_id
        FROM ranked r
        WHERE r.rn = 1 AND a.id = r.article_id AND (a.source_id IS DISTINCT FROM r.source_id);
        """)
        conn.commit()


def main():
    conn = get_db_conn()
    try:
        ensure_columns_and_indexes(conn)
        populate_url_hash(conn)
        backfill_articles_source_id(conn)
        print("Backfill completed")
    finally:
        conn.close()


if __name__ == '__main__':
    main()
