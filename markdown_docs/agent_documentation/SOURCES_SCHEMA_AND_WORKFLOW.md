# Sources Schema and Workflow

This document specifies the `sources` schema, provenance mapping (`article_source_map`), ingestion workflows, canonicalization rules, and usage examples for the JustNews project.

## Goals

- Maintain a canonical, auditable list of news sources (homepages and canonical publisher URLs).
- Record provenance for every stored article via `article_source_map`.
- Provide a fast primary lookup (`articles.source_id`) for common joins/analytics while keeping provenance in `article_source_map`.
- Allow time-series scores (bias/trust/paywall) to be stored in `source_scores`.

## Schema (high-level)

1. `public.sources` (canonical source metadata)
   - id: BIGSERIAL PK
   - url: TEXT NOT NULL (canonical homepage or publisher URL)
   - domain: TEXT (host portion of the URL)
   - name: TEXT
   - description: TEXT
   - country: TEXT
   - language: TEXT
   - paywall: BOOLEAN DEFAULT FALSE
   - paywall_type: TEXT
   - last_verified: TIMESTAMPTZ
   - url_hash: TEXT (sha256 of url for fast dedupe)
   - metadata: JSONB (free-form)
   - created_at, updated_at: TIMESTAMPTZ

2. `public.source_scores` (time-series evaluator scores)
   - id, source_id FK -> sources(id), evaluator, score, score_type, details JSONB, created_at

3. `public.article_source_map` (provenance mapping)
   - id BIGSERIAL PK
   - article_id BIGINT
   - source_id BIGINT FK -> sources(id)
   - confidence NUMERIC DEFAULT 1.0
   - detected_at TIMESTAMPTZ DEFAULT now()
   - metadata JSONB

4. `public.articles.source_id` (nullable FK)
   - A denormalized canonical pointer for fast joins; derived from `article_source_map` by canonical rule.

## Indexes & Performance

- `UNIQUE INDEX sources_url_idx ON lower(url)` ensures idempotent upserts.
- `INDEX sources_domain_lower_idx ON lower(domain)` for fast domain lookup.
- `INDEX sources_url_hash_idx ON url_hash` for sha256 lookups.
- `INDEX article_source_map(article_id)` and `INDEX article_source_map(source_id)` for fast joins.
- `INDEX articles(source_id)` for aggregations and joins.

## Ingest-time workflow (recommended)

1. Crawler extracts article and raw metadata (title, url, html). Save article in `articles` with metadata including the original URL.
2. Lookup `sources` by domain and by url_hash. If matched, insert a row into `article_source_map` with `confidence` and `metadata: {"matched_by": "ingest"}`.
3. Determine canonical source_id using the canonical rule (highest confidence, tie break most recent) and set `articles.source_id` (UPDATE) accordingly.
4. If no `sources` match, optionally create a `sources` row in a review queue (or insert automatically with `last_verified=NULL`) for later human verification.

## Canonical selection rule (recommended)

1. Select mapping row for article with highest `confidence`.
2. If tie, prefer the most recent `detected_at`.
3. If still tied, prefer mapping with `metadata->>'matched_by' = 'ingest'` over heuristics.

This rule should be implemented in a central backfill and optionally maintained via a db trigger or application-level logic at ingest.

## Paywall detection

- Populate `sources.paywall` boolean and `paywall_type` using a combination of heuristics:
  - HTML markers (class names/id strings like `paywall`, `metered`, `subscription`)
  - Presence of interstitial scripts recognized from a curated list.
  - Manual human review for ambiguous cases.

Store detection details in `sources.metadata.paywall_checks`.

## Example SQL: Upsert a source and a mapping (application-level)

```sql
-- Insert or update source
INSERT INTO public.sources (url, domain, name, description, metadata, url_hash, last_verified)
VALUES ('https://www.example.com', 'www.example.com', 'Example', 'Example news', '{"curated": true}', md5('https://www.example.com'), now())
ON CONFLICT ON CONSTRAINT sources_url_idx
DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description, metadata = public.sources.metadata || EXCLUDED.metadata, last_verified = now()
RETURNING id;

-- Insert article mapping
INSERT INTO public.article_source_map (article_id, source_id, confidence, metadata)
VALUES (1234, 42, 0.98, '{"matched_by": "ingest"}')
ON CONFLICT DO NOTHING;
```

## CLI tools provided

- `scripts/news_outlets.py --file <md> [--map-articles] [--dry-run]`
  - Upserts sources from the markdown file.
  - If `--map-articles` is passed, best-effort domain matching is used to insert rows into `article_source_map` and optionally update `articles.source_id`.

- `scripts/backfill_article_sources.py`
  - Adds `url_hash`, creates functional indexes, and backfills `articles.source_id` using the canonical rule.

## Use-cases and examples

1. Quick analytics: "Which sources produced the most articles yesterday?"
   - Use `articles.source_id` for fast grouping.

2. Audit: "Show all candidate sources for article X and when they were detected"
   - Query `article_source_map` filtered by `article_id`.

3. Score-driven alerts: "Find sources with bias score < 0.2"
   - Join `source_scores` and `sources` tables.

4. Paywall-aware fetching: avoid re-fetching paywalled content or route through paywall-handling pipelines when `sources.paywall = TRUE`.

## Maintenance & operations

- Periodically run `scripts/backfill_article_sources.py` after improving mapping heuristics.
- Maintain a curated sources review queue for newly discovered sources with `last_verified = NULL`.
- Expose a small REST endpoint (internal) to query and edit `sources` metadata for manual corrections.

## Tests

- Add unit tests for `scripts/news_outlets.py::parse_markdown_table_rows` to guarantee parsing robustness.
- Add tests for backfill script behavior on a small, in-memory test database or a test Postgres instance.

## Security and data considerations

- Treat `sources.metadata` and `source_scores.details` as potentially large JSON; ensure queries use indexes and avoid full-table scans.
- Do not store credentials or secrets in `sources.metadata`.

## Next steps for automation

- Implement an ingestion microservice that: receives article payloads, performs domain/source lookup, inserts article, inserts mapping, and sets canonical source_id in one transaction.
- Add a scheduled job to recompute canonical mappings for articles older than N days when your heuristics improve.

---

For implementation help (migrations, triggers, or API endpoints) see the `scripts/` directory in this repo and contact the repository owner for deployment instructions.
