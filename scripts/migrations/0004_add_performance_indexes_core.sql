-- Add performance indexes for core DB tables (crawled_urls, sources, source_scores)
-- Guard existence to ensure idempotence

DO $$
BEGIN
  IF EXISTS (SELECT FROM information_schema.tables WHERE table_schema='public' AND table_name='crawled_urls') THEN
    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_crawled_urls_url') THEN
      EXECUTE 'CREATE INDEX idx_crawled_urls_url ON crawled_urls (url)';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_crawled_urls_timestamp') THEN
      EXECUTE 'CREATE INDEX idx_crawled_urls_timestamp ON crawled_urls (created_at)';
    END IF;
  END IF;

  IF EXISTS (SELECT FROM information_schema.tables WHERE table_schema='public' AND table_name='sources') THEN
    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_sources_name') THEN
      EXECUTE 'CREATE INDEX idx_sources_name ON sources (name)';
    END IF;
  END IF;

  IF EXISTS (SELECT FROM information_schema.tables WHERE table_schema='public' AND table_name='source_scores') THEN
    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_source_scores_source_id') THEN
      EXECUTE 'CREATE INDEX idx_source_scores_source_id ON source_scores (source_id)';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_source_scores_score') THEN
      EXECUTE 'CREATE INDEX idx_source_scores_score ON source_scores (score DESC)';
    END IF;
  END IF;
END
$$;
