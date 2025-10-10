---
title: "Refactor Plan — Scout, Crawler, Crawler Control → Single AI Crawler + Monitoring"
tags:
  - architecture
  - refactor
  - scout
  - crawler
  - monitoring
  - crawl4ai
status: draft
last_updated: 2025-10-08
---

# Executive summary

This document recommends a focused refactor to resolve functional drift and
duplication across three agents: `scout`, `crawler` (legacy utilities), and
`crawler_control`. The goal is to:

- A) Create a single, adaptable, AI-based crawler implementation (crawl4ai
  native where possible) that supports both sequential and high-throughput
  modes and can be trained / tuned over time.
- B) Provide full, observable metrics and real-time feedback for crawl jobs
  (Prometheus + WebSocket dashboard + job-level events).
- C) Provide a light-weight news-monitoring service (the original Scout
  periodic-check role) that detects new or updated stories and triggers
  targeted crawling, reducing wasted work and analysis load.

The plan is phased, low-risk, reversible, and includes compatibility shims so
legacy scripts and dashboards continue to operate during migration.

## 1. Deep dive — current state (functionality & code quality)

Summary of findings for each agent (source excerpts available in repo):

## 1.1 Scout

- Functionality: Broad responsibilities — source discovery, content
  crawl/extraction endpoints, ML-driven analysis (NextGen GPU Scout engine),
  wrappers for production crawlers (`production_crawl_ultra_fast`,
  `production_crawl_ai_enhanced`) and numerous high-level tools.
- Observations:
  - Scout is now both a high-level intelligence layer (analysis models) and
    an entrypoint for production crawling orchestrations (mixing concerns).
  - Implements many useful features: URL validation, rate-limiting,
    security logging, feedback logging, optional crawl4ai integration.
  - Heavy coupling to global singletons (e.g., `scout_engine`, module-level
    connection pools), making unit testing and isolation harder.
  - Large functions with many responsibilities (e.g., `intelligent_source_discovery`,
    `intelligent_content_crawl`) that combine network calls, I/O, and ML
    analysis without clear separation of concerns.
  - Good practices: structured logging via `common.observability`, metrics
    via `common.metrics`, pydantic endpoints in FastAPI.
  - Tests: scarce / missing for idempotency and concurrency; no explicit
    unit tests recorded for `intelligent_*` flows.

## 1.2 Crawler (agents/crawler)

- Functionality: Shared crawler utilities (rate-limiter, robots checker,
  modal dismissers, canonical metadata generation), DB pooling helpers.
- Observations:
  - Strong, reusable components (RateLimiter, RobotsChecker, ModalDismisser)
    with reasonable abstractions.
  - DB connection pool utils are provided but are global and assume a Postgres
    environment; error handling exists but can be tightened.
  - CanonicalMetadata helpers are comprehensive and suitable as a single
    canonical source of truth for crawl output.
  - Overall code quality for utilities: good; however these utilities are
    duplicated or re-imported across scout and orchestrator code paths.

## 1.3 Crawler Control

- Functionality: Web dashboard and API for starting/stopping crawls, reading
  metrics and status, and bridging user/UI commands to the crawler agent.
- Observations:
  - Implements helpful CLI/web API semantics (commands like `sources <N>`),
    metrics fallbacks, and dashboard endpoints.
  - Makes HTTP calls to `CRAWLER_AGENT_URL` for job control but the crawler
    agent currently lacks some job lifecycle APIs (stop, graceful shutdown).
  - The UI layer presumes synchronous behavior and uses fallback mock data
    when the crawler is unavailable — good for resilience, but hides missing
    behavior in the crawler.

## Cross-cutting technical issues

- Functional drift: responsibilities for scheduling, monitoring, crawling,
  and ML analysis are split across agents in inconsistent ways.
- Duplication of crawl orchestration logic: Scout orchestrator and
  crawler_control both implement high-level job orchestration behavior.
- Missing job lifecycle endpoints in crawler agent (stop, status details,
  per-job metrics), forcing crawler_control to emulate behavior.
- Sparse unit tests for idempotency, concurrency, and db dedupe semantics.
- Some mixed sync and async patterns (requests in async endpoints), which
  may block and reduce throughput.

## 2. Objectives of the refactor

1. Centralize crawling functionality into a single, well-tested `crawler`
   service that:
   - Exposes a clear HTTP/MCP API for operations and job lifecycle management.
   - Uses crawl4ai for stable, LLM-augmented extraction where available,
     with a Docker/CLI fallback.
   - Offers `sequential` and `concurrent` processing modes with idempotent
     persistence.

2. Preserve Scout as an *intelligence & monitoring* agent whose primary
   purpose is to detect new/changed content and to provide ML-driven
   signals (news_score, importance), not to act as the primary crawler.

3. Make Crawler Control the single UI & orchestration gateway to create,
   monitor, and control crawl jobs — with real-time feedback and rich
   metrics.

## 3. Proposed new architecture (conceptual)

High-level components:

- agents/scout (monitoring & intelligence)
  - `scout.monitor` — light-weight watcher that detects new/changed
    content across sources (RSS, sitemaps, site-change heuristics).
  - `scout.analysis` — ML model wrappers: scoring, bias/sentiment checks,
    and suggestion generator.

- agents/crawler (single production crawler)
  - `crawler.api` — FastAPI server exposing job submission, status, metrics,
    job control (start/stop/pause/resume/clear), job logs, and a Prometheus
    /metrics endpoint.
  - `crawler.engine` — central engine that schedules, executes, and tracks
    crawl jobs. Accepts a `mode` parameter (sequential/concurrent), supports
    run resumption, has a job queue and per-job state persisted to Postgres.
  - `crawler.extract` — crawl4ai integration + extraction strategies; fall
    back to headless browser if needed. Implements `extract_text()`,
    `extract_markdown()`, `extract_metadata()`.
  - `crawler.persist` — idempotent persister: canonicalization, dedupe via
    URL-hash register, transactional inserts, audit metadata.
  - `crawler.metrics` — job-level and domain-level Prometheus instrumentation
    and job event emission over a WS/RT channel.

- agents/crawler_control (UI & orchestration)
  - Web UI and WebSocket to show job progress, logs, and allow manual
    control (start/stop/retry). It becomes a thin control-plane that calls
    `crawler.api`.

## Inter-agent communication

- MCP Bus used for higher-level flows (e.g., Scout discovers a breaking
  story → posts a tool call to Crawler to enqueue a targeted job).
- Direct HTTP used for job control & metrics (crawler exposes fine-grained
  job APIs). Crawler Control calls crawler HTTP endpoints.

## 4. Detailed refactor & action plan (phased)

This plan is split into concrete, reviewable phases. Each phase includes
files to add/change, tests to add, and acceptance criteria.

## Phase 0 — Preparatory audit & tests (2–4 days)

- Actions:
  1. Add unit tests for existing critical pieces: idempotency of persister
     (if any), `RateLimiter`, `RobotsChecker`, small tests for
     `extract_article_content` logic and Scout `intelligent_*` flows using
     request mocks.
  2. Add a small `tests/test_jobidempotency.py` fixture that asserts
     duplicate persist attempts are no-ops.
  3. Tighten imports in Scout to avoid heavy module-level initializations on
     import (defer NextGen engine creation).

- Goals/Acceptance:
  - Baseline test coverage added for core utilities.
  - Identify failing assumptions before heavy refactor.

## Phase 1 — Scaffold single `crawler` service & API (3–5 days)

- Add new agent: `agents/crawler/` with files:
  - `__init__.py`, `api.py` (FastAPI server), `engine.py` (job engine stub),
    `extract.py` (crawl4ai wrapper stub), `persist.py` (idempotent persister
    stub), `metrics.py`, `types.py` (Pydantic models), `tests/`.

- Key API endpoints in `api.py`:
  - POST /jobs — submit job (domains or urls, mode, priority, metadata)
  - GET /jobs — list jobs
  - GET /jobs/{job_id} — job detail (status, progress, logs)
  - POST /jobs/{job_id}/stop — request stop
  - POST /unified_production_crawl — back-compat thin wrapper for existing
    `production` endpoints
  - GET /metrics — Prometheus exposition

- Tests to add:
  - test_api_submit_job_returns_jobid
  - test_job_lifecycle_state_transitions (pending->running->completed)

- Acceptance:
  - New `crawler` service responds to /jobs and /metrics locally.

## Phase 2 — Implement idempotent persist and canonical dedupe (2–4 days)

- Implement `crawler.persist.persist_article()`:
  - Use canonical `url_hash` insertion into `crawl_jobs` or `register_url`
  - Wrap in DB transaction — return `inserted: bool` and `article_id`.
  - Add unit tests: write same article twice -> second returns inserted=False.

- Acceptance:
  - Unit tests for idempotency pass.

## Phase 3 — Crawl4AI integration & extract strategies (4–7 days)

- Implement `crawler.extract` with two modes:
  - `native_crawl4ai` (preferred): call crawl4ai AsyncWebCrawler with
    LLMExtractionStrategy when CRAWL4AI_NATIVE_AVAILABLE is true.
  - `headless_browser` fallback: playwright-based extractor for sites
    requiring JS interaction and when crawl4ai unavailable.

- Ensure extraction returns canonical structure: {url, title, content, markdown, metadata}

- Acceptance:
  - Integration test with small set of sample sites (mocked) verifies
    extraction strategy selection and migration.

## Phase 4 — Job engine: sequential & concurrent (4–7 days)

- Implement `crawler.engine.JobRunner`:
  - Mode `sequential`: iterate URLs/sites one-by-one and persist.
  - Mode `concurrent`: schedule tasks using asyncio.Semaphore and domain
    rate-limiter; support cancellation and graceful stop.
  - Emit events (job progress) to a job-status table in DB and a WebSocket
    channel for real-time UI.

- Tests:
  - test_concurrent_run_respects_rate_limits
  - test_job_stop_requests_cancel_tasks

- Acceptance:
  - JobRunner can start/stop jobs; metrics `articles_processed_total` and
    `inserted_total` increment.

## Phase 5 — Crawler Control integration & UI real-time feedback (3–5 days)

- Update `agents/crawler_control` to consume new `/jobs` APIs and listen to
  WebSocket job event streams for real-time progress.
- Improve UI: show per-job progress bars, per-site throughput, and allow
  stop/pause/resume. Provide alerts on job failures.

- Acceptance:
  - Web UI shows live job progress from new `crawler` service.

## Phase 6 — Scout monitoring & trigger system (4–6 days)

- Implement `scout.monitor` (new module):
  - Lightweight scheduler (cron-like or APScheduler) that watches sources
    (RSS, sitemaps, delta checks on canonical URL, small content diffs).
  - When new/updated article detected, compute a `news_score` using
    Scout intelligence and create a targeted crawl job via MCP Bus call to
    `crawler` (e.g. `POST /jobs` with `priority=high`).
  - Provide backpressure and rate limiting: do not flood crawler.

- Detection heuristics:
  - RSS / Atom polling for new entries (with etag/last-modified caching).
  - Sitemap polling (lastmod), incremental sitemap parsing.
  - Periodic lightweight HEAD/If-Modified-Since checks for canonical urls.
  - Content fingerprinting (first 2k chars hash) to detect meaningful updates
    vs. cosmetic changes.

- Alerts & escalation:
  - If many sources update in short window, mark as "breaking" and raise
    confidence threshold for downstream analysis.

- Tests:
  - test_monitor_detects_new_rss_item -> enqueues job
  - test_monitor_debounces_rapid_updates

- Acceptance:
  - Scout monitor enqueues targeted crawl jobs and reduces full-site
    multi-site crawling in typical runs.

## Phase 7 — Migration, shims, and decommission (2–3 days)

- Provide compatibility wrappers for legacy endpoints and scripts:
  - `agents/scout/production_crawlers/*` becomes thin wrappers calling the
    new `crawler` endpoints until removed.
- Deprecate legacy modules in stages and document changes in
  `markdown_docs/development_reports/REFactor_SCOUT_CRAWLER_PLAN.md`.

## 5. Metrics, telemetry & real-time feedback (exact definitions)

Instrument these Prometheus metrics (naming conventions):

- crawler_jobs_submitted_total{mode,priority}
- crawler_jobs_running_gauge
- crawler_job_duration_seconds_bucket{job_id}
- crawler_articles_processed_total{job_id,domain,extraction_method}
- crawler_articles_inserted_total{job_id,domain}
- crawler_articles_deduped_total{job_id,domain}
- crawler_extract_errors_total{domain,error_type}
- crawler_queue_depth_gauge
- scout_monitor_events_total{event_type}

Real-time feedback transport:

- Job event stream over WebSocket: emits JSON events: job.progress,
  job.complete, article.processed (with inserted/deduped), domain.metrics.
- Job audit table (Postgres): `crawl_jobs` + `crawl_job_events` (append-only)

Dashboard augmentations:

- Live job table with per-job throughput, domain breakdown, top errors.
- Historical job graphs (rate vs time), per-domain success rates.

## 6. News monitoring algorithm (detailed)

Design goals: precise, light-touch detection that avoids heavy crawling.

Data sources for detection:

- Feeds: RSS/Atom/endpoints — processed first, low-cost; include `guid` and
  `published` times.
- Sitemaps: parse `<lastmod>` and queue changes.
- Canonical URL fingerprinting: maintain small (sha256) fingerprint of the
  first ~4KB of article text; compare to detect substantive changes.

Detection flow (per source):

1. Poll feed/sitemap with exponential backoff (configurable per-source).
2. On candidate new URL, perform a lightweight HEAD / conditional GET to
   fetch headers & ETag. If changed, fetch first 2k of content (or ask
   crawl4ai to fetch a small preview) and compute fingerprint.
3. If fingerprint is new, compute `news_score = scout_engine.estimate_news_score(...)`
   (low-latency model). If above `score_threshold` or `breaking_threshold`,
   create a high-priority job in `crawler` and notify `crawler_control` UI.
4. If many sources return updates near-simultaneously, escalate to
   "system-level breaking" and optionally increase sampling frequency.

Debounce & cost control:

- Each source has a `poll_interval` and `daily_budget` (max requests per day).
- Deduplicate by domain + url-hash across monitoring queue.

## 7. Data schema (suggested additions)

Add these tables (Postgres):

1. crawl_jobs (id PK, status, mode, priority, submitted_by, submitted_at, started_at, finished_at, metrics jsonb)
2. crawl_job_events (id, job_id FK, event_time, event_type, payload jsonb)
3. crawl_results (id, job_id FK, url, url_hash, domain, title, canonical, content_snippet, status, inserted bool, article_id bigint, metadata jsonb)
4. monitor_sources (id, source_id FK, last_polled, etag, last_fingerprint, poll_interval, daily_budget, enabled)

## 8. Testing & CI

- Add tests as described per phase; run on CI matrix with short network mocks.
- Add `verify-model-bootstrap` CI job to ensure model manager APIs are healthy
  and `crawl4ai` availability is checked in `edged` mode (native vs fallback).

## 9. Backward compatibility & migration strategy

- Keep current Scout analysis APIs; point `production_crawl_*` endpoints to
  call new crawler `/unified_production_crawl`. Create compatibility layer
  under `agents/scout/production_crawlers/compat.py` for 2–3 weeks.
- Start with non-destructive reads: run new crawler in `dry-run` mode to
  produce metrics & verify results before enabling full production runs.

## 10. Risks, mitigations, and operational notes

- Risk: crawl4ai native unavailability in some deploys; mitigation: robust
  fallback to headless browser and pre-seeded `model_cache` for RAG.
- Risk: sudden load spikes from monitoring -> many high-priority jobs; mitigation:
  per-source daily budgets and global emergency throttle in crawler engine.
- Risk: data duplication; mitigation: canonical URL hashing + DB-level unique
  constraints and idempotent persist.

## 11. Rough estimate (person-days)

- Phase 0: 2–4 days
- Phase 1: 3–5 days
- Phase 2: 2–4 days
- Phase 3: 4–7 days
- Phase 4: 4–7 days
- Phase 5: 3–5 days
- Phase 6: 4–6 days
- Phase 7: 2–3 days

Total estimate: 24–41 person-days (can be broken into smaller PRs for review)

## 12. Acceptance criteria

1. New `agents/crawler` service exposes the job API and Prometheus metrics.
2. `persist_article` is demonstrably idempotent (unit test asserts second
   persist yields inserted=False).
3. Scout monitor enqueues targeted crawl jobs for new/updated items and
   reduces full multi-site crawl frequency in typical scenarios.
4. Crawler Control displays live job progress and can stop/clear jobs.
5. Legacy endpoints continue to work (compat shim) until deprecation.

## Appendix A — Quick patch checklist (first PRs)

1. Add tests for `RateLimiter`, `CanonicalMetadata.generate_metadata`, and
   an idempotency test for persister.
2. Add `agents/crawler` scaffold (api + engine stub + tests).
3. Implement `persist_article` DB contract with tests.

---

If you want I will scaffold the exact `agents/crawler/` files and add the
initial unit tests (Phase 1 + Phase 2 start). Which phase should I open the
first PR for: scaffolding (Phase 1) or idempotent persist (Phase 2)?
