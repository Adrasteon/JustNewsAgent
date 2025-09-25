-- Migration: create crawled_urls table for idempotent crawl inserts
CREATE TABLE IF NOT EXISTS crawled_urls (
    url TEXT PRIMARY KEY,
    url_hash TEXT,
    first_seen TIMESTAMPTZ DEFAULT now(),
    last_seen TIMESTAMPTZ DEFAULT now()
);

-- Optional index on url_hash for faster lookups if used
CREATE INDEX IF NOT EXISTS idx_crawled_urls_url_hash ON crawled_urls(url_hash);
