-- Non-destructive migration to add a pgvector column to articles for future migration to VECTOR type
-- This adds a new column `embedding_vector` of type vector(768) if pgvector is installed and the column is missing.
-- It does NOT remove or alter existing `embedding` or `article_vectors` structures.

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector') THEN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='articles' AND column_name='embedding_vector') THEN
      EXECUTE 'ALTER TABLE articles ADD COLUMN embedding_vector VECTOR(768)';
    END IF;
  END IF;
END
$$;

-- Note to maintainers: If you want to convert existing NUMERIC[] embeddings to VECTOR, add a conversion script
-- that reads numeric[] values and writes them into the new embedding_vector column after appropriate scaling/format.
-- Conversion is a data migration and should be performed with backups and verification steps.