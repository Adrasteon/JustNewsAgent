---
title: Crawler Consolidation Plan — JustNewsAgent
description: Auto-generated description for Crawler Consolidation Plan — JustNewsAgent
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# Crawler Consolidation Plan — JustNewsAgent

Date: 2025-08-27
Author: Consolidation plan generated from interactive session

---

This document consolidates the recommendations and concrete refactor plan for merging and standardizing the repository's crawler implementations (Scout agent + repo-root donor scripts). It captures design rationales, canonical contracts, step-by-step changes, testing guidance, and follow-ups.

## Goals

- Merge the best behaviors from existing crawler scripts into single canonical implementations per type.
- Support two operational modes per canonical crawler:
  - sequential / agent-friendly (crawler_A behaviour): process one page at a time, clear handoffs to other agents, immediate persistence and provenance.
  - concurrent / throughput (crawler_B behaviour): batch/async processing across many pages/sites for high throughput.
- Centralize shared services (DB dedupe, NewsReader, config) so dedupe is durable and canonical.
- Archive donor scripts once canonical replacements are in place.
- Add minimal tests and documentation to make the consolidation maintainable.

## Files considered

(The following were the main inputs to this plan — canonical site modules and repo-root donors.)

- `agents/scout/tools.py`
- `agents/scout/main.py`
- `agents/scout/production_crawlers/orchestrator.py`
- `agents/scout/production_crawlers/sites/bbc_crawler.py` (UltraFast crawler canonical)
- `agents/scout/production_crawlers/sites/bbc_ai_crawler.py` (AI-enhanced canonical)
- `agents/scout/gpu_scout_engine.py` / `gpu_scout_engine_v2.py` (Scout intelligence engine)
- `agents/scout/practical_newsreader_solution.py` (NewsReader implementation)
- Repo-root donor scripts: `production_bbc_crawler.py`, `ultra_fast_bbc_crawler.py`
- DB helper: `scripts/db_dedupe.py` (ensure_table, register_url)

## High-level design decisions

1. Canonical types
   - UltraFastCrawler: optimized for throughput; aggressive modal dismissal JS; multi-browser batch processing; heuristic scoring.
   - ProductionAICrawler: AI-enhanced; integrates `PracticalNewsReader` visual/text analysis; more conservative concurrency and per-article analysis.

2. Modes
   - Each canonical crawler exposes:
     - `run_sequential(site_or_urls, ...)` — agent-friendly one-by-one processing.
     - `run_concurrent(site_or_urls, ...)` — high-throughput batch processing.

3. Central services
   - DB dedupe (`scripts/db_dedupe.register_url`) is called inside canonical `persist_article()` so all callers benefit.
   - `PracticalNewsReader` is canonicalized under `agents/scout/` and used by the AI crawler.
   - A small `agents/scout/config.py` is recommended to centralize environment-driven configuration.

4. Archival
   - Donor repo-root scripts will be moved to `archive_obsolete_files/development_session_[DATE]/` to preserve history but remove duplication.

## Contract / API for canonical crawlers

Class: UltraFastCrawler / ProductionAICrawler

Public methods (async):

- `async initialize() -> bool`
  - Prepare browsers, models, DB table; idempotent.

- `async fetch_urls(site: str, max_urls:int) -> List[str]`
  - Fast discovery of candidate article URLs.

- `async process_url(url: str, mode: str = 'sequential') -> Optional[Dict]`
  - Process and return normalized article dict or None.

- `async run_sequential(site_or_urls, max_articles: int) -> List[Dict]`
  - One-by-one processing suitable for agent handoffs.

- `async run_concurrent(site_or_urls, target_articles:int) -> Dict`
  - Batch processing that returns summary metrics and `articles` list.

- `persist_article(article: Dict) -> bool`
  - Call `ensure_table()` and `register_url()`; if `register_url` returns True, persist (or return True to caller) else don't persist.

Article dict minimal shape:

- `url` (str), `title` (str), `content` (str), `timestamp` (ISO str), `source_method` (str), `processing_time_seconds` (float), `status` ('success'|'error'), optional `analysis` and `news_score`.

## Edge cases & error handling

- DB failures: `persist_article()` must handle exceptions, log errors, and optionally buffer to a local queue rather than fail the crawl.
- OOM/Model failure: AI crawler should fall back to text-only analysis and report fallback metadata.
- Modal/cookie handling variance: dismissers should be heuristic and not block crawls; capture screenshot on repeated failure.
- Rate limiting: provide per-domain politeness and a global concurrency cap in `config.py`.

## Concrete step-by-step refactor plan (apply when approved)

1. Add/Update canonical site modules
   - `agents/scout/production_crawlers/sites/bbc_crawler.py` (UltraFast):
     - Add `run_sequential()` which uses `get_urls_ultra_fast()` (or takes a URL) and calls `process_url_ultra_fast()` per article, then `persist_article()`.
     - Add `persist_article()` that uses `scripts/db_dedupe.ensure_table` and `register_url`.
   - `agents/scout/production_crawlers/sites/bbc_ai_crawler.py` (AI-enhanced):
     - Add `run_sequential()` wrapper and `persist_article()` same as above.
     - Factor `process_single_url()` as the single-URL unit; `process_batch()` is the concurrent path.

2. Centralize config
   - Create `agents/scout/config.py` (env-driven defaults: DB creds, concurrency, timeouts, user_agent).

3. Centralize DB dedupe usage
   - Ensure both canonical site modules call `persist_article()` as single place for `ensure_table/register_url`.

4. Archive donor scripts
   - Move `production_bbc_crawler.py` and `ultra_fast_bbc_crawler.py` to `archive_obsolete_files/development_session_YYYYMMDD/` with a short README.

5. Orchestrator & tools adjustments
   - Update `agents/scout/production_crawlers/orchestrator.py` to allow `mode` param (sequential|concurrent) and to call the appropriate canonical methods.
   - Update `agents/scout/tools.py` production endpoints to accept `mode` as a kwarg and pass through.

6. Tests & docs
   - `tests/test_crawlers_imports.py`: import smoke for canonical classes and `get_supported_sites()`.
   - `tests/test_db_dedupe.py`: small test for `register_url()` semantics (may need a test DB or mocking).
   - Update `agents/scout/README.md` describing canonical classes and how to run sequential vs concurrent modes.

## Patch-level (what will change in which files)

- Modify (small edits) existing canonical site files to add `run_sequential()` and `persist_article()`.
- Add `agents/scout/config.py`.
- Move donor scripts into archive folder (no changes to their content; preserve for reference).
- Update orchestrator to accept `mode` param.
- Update `agents/scout/tools.py` endpoints mapping to accept `mode` and call orchestrator accordingly.
- Add two tests under `tests/`.

All changes are intended to be minimal (add methods, small helpers), preserve existing good logic (modal scripts, scoring), and centralize persistence/dedupe.

## Tests & quality gates

- Import smoke test (fast, no network): `test_crawlers_imports.py`
- DB dedupe unit test: `test_db_dedupe.py` (mock psycopg2 or use a local test DB)
- Run `ruff` lint and `pytest -q` after changes
- Optional smoke: run a one-URL sequential run with very short timeouts to confirm flows and `register_url()` behavior.

## Follow-ups & low-risk extras

- Add `url` and `url_hash` columns to `articles` table and write a backfill migration script.
- Centralize credentials (avoid hard-coded DB creds) and read from env. Add `agents/scout/config.py` for this.
- Add a small health-check for NewsReader and a fallback queue for failed persistence.

## Timeline & next action options

Choose one of the following:

- `Show patches` — I will produce the exact apply_patch diffs for review before applying.
- `Implement` — I will apply the small edits (canonical file methods, config, archive donor scripts), add tests, run the smoke tests, and report results.
- `Adjust plan` — request changes to the approach (e.g., strict separation into different classes or preserving repo-root scripts as wrappers).

---

Appendix: quick reference commands (optional)

Run tests (workspace task):

```bash
# run repository tests using the provided task
# from VS Code tasks: "Run tests (wrapper)" or run directly
./scripts/run_tests.sh
# or, if using conda env (present in workspace tasks)
conda run --name justnews-v2-prod pytest -q
```

Lint with ruff:

```bash
ruff check .
```

---

End of consolidation plan.

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

