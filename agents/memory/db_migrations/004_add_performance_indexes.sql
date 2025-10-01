-- Vector search performance improvements (memory DB scoped)
-- Create vector extension if not exists (for pgvector if available)
CREATE EXTENSION IF NOT EXISTS vector;

-- Add GIN index on metadata for faster JSON queries (if articles table exists)
DO $$
BEGIN
  IF EXISTS (SELECT FROM information_schema.tables WHERE table_schema='public' AND table_name='articles') THEN
    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_articles_metadata') THEN
      EXECUTE 'CREATE INDEX idx_articles_metadata ON articles USING GIN (metadata)';
    END IF;
  END IF;
END
$$;

-- Add index on embedding for faster vector operations (if embedding column exists)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='articles' AND column_name='embedding') THEN
    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_articles_embedding') THEN
      EXECUTE 'CREATE INDEX idx_articles_embedding ON articles USING GIN (embedding)';
    END IF;
  END IF;
END
$$;

-- Add index on content for text search operations (if content column exists)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='articles' AND column_name='content') THEN
    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_articles_content') THEN
      EXECUTE 'CREATE INDEX idx_articles_content ON articles USING GIN (to_tsvector(''english'', content))';
    END IF;
  END IF;
END
$$;

-- Add indexes for training_examples table (guard columns first)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name='training_examples') THEN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='training_examples' AND column_name='agent_name') THEN
      IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_training_examples_agent_task') THEN
        EXECUTE 'CREATE INDEX idx_training_examples_agent_task ON training_examples (agent_name, task_type)';
      END IF;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='training_examples' AND column_name='timestamp') THEN
      IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_training_examples_timestamp') THEN
        EXECUTE 'CREATE INDEX idx_training_examples_timestamp ON training_examples (timestamp)';
      END IF;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='training_examples' AND column_name='importance_score') THEN
      IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_training_examples_importance') THEN
        EXECUTE 'CREATE INDEX idx_training_examples_importance ON training_examples (importance_score DESC)';
      END IF;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='training_examples' AND column_name='correction_priority') THEN
      IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_training_examples_priority') THEN
        EXECUTE 'CREATE INDEX idx_training_examples_priority ON training_examples (correction_priority DESC)';
      END IF;
    END IF;
  END IF;
END
$$;

-- Composite index and partial index creation (guarded)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='articles' AND column_name='id') THEN
    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_articles_content_metadata') THEN
      BEGIN
        EXECUTE 'CREATE INDEX idx_articles_content_metadata ON articles (id, content, metadata)';
      EXCEPTION WHEN others THEN
        -- Ignore errors for complex expressions on some PG versions
        RAISE NOTICE 'Skipping idx_articles_content_metadata creation due to: %', SQLERRM;
      END;
    END IF;
  END IF;
END
$$;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='articles' AND column_name='embedding') THEN
    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='idx_articles_with_embedding') THEN
      EXECUTE 'CREATE INDEX idx_articles_with_embedding ON articles (id, embedding) WHERE embedding IS NOT NULL';
    END IF;
  END IF;
END
$$;