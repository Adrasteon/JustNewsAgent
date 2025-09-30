-- Migration: create article_source_map and add nullable source_id to articles

CREATE TABLE IF NOT EXISTS public.article_source_map (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT NOT NULL,
    source_id BIGINT NOT NULL REFERENCES public.sources(id) ON DELETE CASCADE,
    confidence NUMERIC DEFAULT 1.0,
    detected_at TIMESTAMPTZ DEFAULT now(),
    metadata JSONB DEFAULT '{}'::jsonb
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'article_source_map_article_idx'
    ) THEN
        CREATE INDEX article_source_map_article_idx ON public.article_source_map (article_id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'article_source_map_source_idx'
    ) THEN
        CREATE INDEX article_source_map_source_idx ON public.article_source_map (source_id);
    END IF;
END$$;

-- Add nullable source_id to articles (if articles table exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'articles') THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'articles' AND column_name = 'source_id'
        ) THEN
            ALTER TABLE public.articles ADD COLUMN source_id BIGINT REFERENCES public.sources(id);
        END IF;
    END IF;
END$$;

COMMENT ON TABLE public.article_source_map IS 'Mapping table to record provenance between articles and canonical sources. Supports multiple candidate sources per article.';
