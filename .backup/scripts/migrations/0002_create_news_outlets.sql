-- Migration: create sources and source_scores
-- Add a sources table to hold canonical source metadata and a separate source_scores table
-- Designed to be idempotent when applied to the main justnews PostgreSQL database

CREATE TABLE IF NOT EXISTS public.sources (
    id BIGSERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    domain TEXT,
    name TEXT,
    description TEXT,
    country TEXT,
    language TEXT,
    paywall BOOLEAN DEFAULT FALSE,
    paywall_type TEXT,
    last_verified TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Ensure URL uniqueness for idempotent upserts
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'sources_url_idx'
    ) THEN
        CREATE UNIQUE INDEX sources_url_idx ON public.sources ((lower(url)));
    END IF;
END$$;

-- Source evaluation scores (time-series, multiple evaluators/score types allowed)
CREATE TABLE IF NOT EXISTS public.source_scores (
    id BIGSERIAL PRIMARY KEY,
    source_id BIGINT NOT NULL REFERENCES public.sources(id) ON DELETE CASCADE,
    evaluator TEXT, -- e.g., 'human_review', 'automated_bias_detector'
    score NUMERIC, -- numeric score, interpretation depends on score_type
    score_type TEXT, -- e.g., 'bias', 'trust', 'paywall_score', 'credibility'
    details JSONB DEFAULT '{}'::jsonb, -- free-form details about how the score was computed
    created_at TIMESTAMPTZ DEFAULT now()
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'source_scores_source_id_idx'
    ) THEN
        CREATE INDEX source_scores_source_id_idx ON public.source_scores (source_id);
    END IF;
END$$;

-- Optional mapping table if articles are stored separately and need to link to sources
-- (See scripts/migrations/0003_create_article_source_map.sql for provenance mapping and optional articles.source_id column.)

COMMENT ON TABLE public.sources IS 'Canonical list of news sources used by JustNews; populated from potential_news_sources.md and enrichment pipelines.';
COMMENT ON TABLE public.source_scores IS 'Time-series of evaluation/bias/paywall and other scores for each source.';
