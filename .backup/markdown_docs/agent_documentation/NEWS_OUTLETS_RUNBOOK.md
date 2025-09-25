---
title: News Outlets Loader & Backfill Runbook
description: Auto-generated description for News Outlets Loader & Backfill Runbook
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# News Outlets Loader & Backfill Runbook

This runbook explains how to safely run the canonical sources loader (`scripts/news_outlets.py`) and the backfill script (`scripts/backfill_article_sources.py`) against the `justnews` database. It covers prerequisites, dry-run and production runs, validation, backups and rollback guidance, scheduling notes, and common troubleshooting items encountered during development.

## Purpose & Scope
- Purpose: import and maintain canonical `public.sources` rows from a curated markdown list and create provenance mappings in `public.article_source_map`. Backfill computes `articles.source_id` from mappings using the canonical selection rule.
- Scope: runbook is intended for operators with DB access and repository checkout. These operations change database state — treat as production-sensitive.

---

## Safety first (before you run)
1. Work in a maintenance window for production if possible.
2. Ensure you have a recent DB backup (pg_dump) or snapshot. Always take a logical backup of affected tables before modifications:

```bash
# dump only the relevant tables (fast and small)
pg_dump --host=localhost --port=5432 --username=justnews_user --format=custom --file=backups/justnews_sources_pre_run_$(date +%F_%H%M).dump --table=public.sources --table=public.article_source_map --table=public.articles justnews
```

3. Ensure `~/.pgpass` or environment-based credentials are present for non-interactive runs. `~/.pgpass` must have mode 600.

```bash
# set safe permissions
chmod 600 ~/.pgpass
# sample .pgpass entry
# host:port:database:username:password
localhost:5432:justnews:justnews_user:REPLACE_WITH_PASSWORD
```

4. Run in a development or staging DB first and confirm results.
5. Use the `--dry-run` option on the loader before writing anything.
6. Run scripts with the repository root on PYTHONPATH to resolve imports:

```bash
export PYTHONPATH=.
```

---

## Prerequisites
- Local access to the `justnews` Postgres instance (or connection info for staging/prod).
- Python 3.x and required test dependencies installed (see `tests/requirements.txt`).
- `PYTHONPATH=.` set or run via `PYTHONPATH=. python3 ...` to resolve repo imports.
- `~/.pgpass` set up or `DATABASE_URL`/`PGPASSWORD` env var provided.
- Ensure migrations in `scripts/migrations/` applied. If not, apply before running loader/backfill.

---

## Quick commands (Dry-run -> Real run)

1) Dry-run loader (parse file, do not change DB):

```bash
PYTHONPATH=. python3 scripts/news_outlets.py --file markdown_docs/agent_documentation/potential_news_sources.md --dry-run --map-articles
```

- Expected output: "Dry run: found N rows" plus a sample of parsed rows. No DB changes.

2) Real loader (idempotent upsert + optional article mapping):

```bash
# real run
PYTHONPATH=. python3 scripts/news_outlets.py --file markdown_docs/agent_documentation/potential_news_sources.md --map-articles
```

- This will upsert sources into `public.sources` and (if `--map-articles`) attempt best-effort domain-based inserts into `public.article_source_map` and may update `articles.source_id` according to canonical rule.

3) Backfill (recompute `articles.source_id` and ensure url_hash/indexes):

```bash
PYTHONPATH=. python3 scripts/backfill_article_sources.py
```

- This script will create `url_hash` if missing, build required indexes, and run canonical selection to populate `articles.source_id`.

---

## Validation queries (run after each step)

1) Verify counts of sources and latest updated rows:

```sql
-- total sources
SELECT count(*) FROM public.sources;
-- recently updated/inserted (last 1 hour)
SELECT id, url, name, last_verified, updated_at FROM public.sources WHERE updated_at > now() - interval '1 hour' ORDER BY updated_at DESC LIMIT 50;
```

2) Verify provenance mappings and sample for a specific article (use a sample article id):

```sql
-- mappings for article 12345
SELECT * FROM public.article_source_map WHERE article_id = 12345 ORDER BY detected_at DESC;
-- counts per article
SELECT article_id, count(*) AS mappings FROM public.article_source_map GROUP BY article_id ORDER BY mappings DESC LIMIT 20;
```

3) Check distribution of `articles.source_id` (should be mostly populated after backfill):

```sql
SELECT count(*) FILTER (WHERE source_id IS NULL) AS null_source_count, count(*) AS total_articles FROM public.articles;
```

4) Confirm indexes exist (for performance):

```sql
-- list indexes we rely on
SELECT indexname, indexdef FROM pg_indexes WHERE tablename IN ('sources', 'article_source_map', 'articles') ORDER BY tablename, indexname;
```

---

## Rollback & recovery guidance
- If the loader inserted many incorrect `sources` rows and you need to revert the run, restore from the backup you took before the run (recommended).
- If you cannot restore a full backup and the loader only inserted rows with a distinct marker (e.g., `last_verified IS NULL` or metadata flag), you can remove those rows selectively. Example:

```sql
-- remove recently created sources (careful: adjust time window)
DELETE FROM public.sources WHERE last_verified IS NULL AND created_at > now() - interval '1 hour';

-- remove related provenance rows inserted in the same window
DELETE FROM public.article_source_map WHERE detected_at > now() - interval '1 hour' AND metadata->>'matched_by' = 'ingest';
```

- To revert the backfill on `articles.source_id`, you can set `source_id` back to NULL for the affected time window or re-run a restore of just that column from a dump. Example:

```sql
-- nullify recent source_id updates
UPDATE public.articles SET source_id = NULL WHERE updated_at > now() - interval '1 hour';
```

Note: selective deletes are risky. Prefer restoring from the logical dump taken prior to the run.

---

## Scheduling and automation
- For recurring maintenance (e.g., nightly recompute), create a scheduled job that runs `backfill_article_sources.py` on a staging instance first, then promotes changes or runs on production with supervision.
- Example cron (runs nightly at 02:30):

```cron
30 2 * * * cd /path/to/JustNewsAgent && export PYTHONPATH=. && /usr/bin/python3 scripts/backfill_article_sources.py &>> /var/log/justnews/backfill.log
```

- Use a job runner that supports notifications on failure (systemd timer, Airflow, or CI scheduled workflows). When automating, always include pre-run `pg_dump --schema-only` and post-run validation checks.

---

## Troubleshooting (common errors and fixes)

1) ModuleNotFoundError: No module named 'scripts'
- Cause: running python without repository root on PYTHONPATH.
- Fix: run with `PYTHONPATH=.` or export prior to running.

```bash
# run from repo root
PYTHONPATH=. python3 scripts/news_outlets.py --file <file> --dry-run
```

2) psycopg2 OperationalError: fe_sendauth: no password supplied / password authentication failed
- Cause: missing credentials for non-interactive connection.
- Fixes:
  - Add an entry for `justnews` in `~/.pgpass` with mode 600.
  - Or set `PGPASSWORD` or `DATABASE_URL` appropriately.

3) SQL error referencing a constraint name for ON CONFLICT that does not exist
- Cause: the code attempted `ON CONFLICT ON CONSTRAINT <name>` but the DB uses an expression index (for example `UNIQUE INDEX ON (lower(url))`) rather than a named constraint.
- Fix: use the CTE-based upsert pattern implemented in `scripts/news_outlets.py` (the loader is idempotent) or create a named unique constraint if you prefer to use `ON CONFLICT ON CONSTRAINT`.

4) CTE or upsert syntax errors during iteration on SQL
- Cause: complex CTEs with incorrect UNION/RETURNING ordering.
- Fix: prefer the tested CTE pattern (update returning id; insert where not exists returning id; then combine) as in the current `scripts/news_outlets.py`.

5) Permissions errors when using `psql`/pg_dump
- Ensure the DB user has the required permissions to SELECT, INSERT, UPDATE, CREATE INDEX (if migrations are run). Consider a role with limited permissions for loader-only runs.

6) Long-running locks or slow writes
- If load affects many rows, run during a maintenance window. Consider batching or using `pg_repack`/maintenance to reduce bloat after large churn.

---

## Example verification checklist (copy/paste)
1. Run dry-run and confirm parsed N rows
2. Take backup of `sources`, `article_source_map`, `articles`
3. Run real loader
4. Check `count(*)` delta on `public.sources`
5. Spot-check 10-20 `sources` rows to ensure URL parsing correct
6. Validate `article_source_map` insert counts and a sample `articles.source_id` updated
7. Run `backfill_article_sources.py` if required and validate `articles.source_id` distribution
8. Vacuum/analyze affected tables if large updates occurred

---

## Contact / escalation
- If you hit an issue that is not resolvable with the above steps, capture the error output and the state of the DB (row counts and a few sample rows) and contact the repository owner or on-call engineer.

---

## Notes & history
- This runbook was created on 2025-08-28 and captures fixes and troubleshooting from recent runs: adding `~/.pgpass` support, running with `PYTHONPATH=.`, and switching the loader to a CTE-style upsert to handle expression-based unique indexes.

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

