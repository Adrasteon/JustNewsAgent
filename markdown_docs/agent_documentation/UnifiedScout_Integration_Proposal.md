---
title: UnifiedScout Integration Proposal — JustNewsAgent
description: Proposal to integrate Scout, Crawler, Crawler Control and NewsReader into a single, production-safe UnifiedScout agent
tags: [proposal, scout, crawler, newsreader, integration]
status: draft
last_updated: 2025-10-01
---

# UnifiedScout Integration Proposal — JustNewsAgent

Date: 2025-10-01
Author: Auto-generated proposal (based on repository audit)

## Executive summary

This proposal defines a concrete plan to merge and standardize the crawling, rendering and page-level extraction responsibilities currently split across the Scout, Crawler, CrawlerControl and NewsReader agents into a single, modular service called "UnifiedScout".

UnifiedScout preserves the best of existing implementations:
- Crawl orchestration and advanced strategies (Crawl4AI / Playwright)
- Production-validated page extraction and visual analysis (NewsReader)
- GPU-accelerated scoring and analysis (Scout Intelligence Engine)
- Operator control and observability (CrawlerControl semantics)

Goals
- Provide a canonical, testable and maintainable API for sequential (agent-friendly) and concurrent (throughput) crawling modes.
- Centralize extraction, dedupe and persistence so downstream agents receive canonical article artifacts.
- Respect resource limits (GPU/Memory/Browser) and retain safe fallbacks.
- Keep migration risk low via thin compatibility wrappers and small incremental PRs.

---

## Proposed high-level architecture

UnifiedScout is a single service with internal modular subcomponents (process-level modules/pluggable classes). It exposes an MCP-compatible API and a small HTTP admin interface.

Components:
1. SourceManager (Discovery)
   - Seed ingestion, periodic refresh, per-source policies (rate limits, robots rules).
2. Scheduler & Job Store
   - Persistent job store (Postgres or Redis) with job lifecycle, heartbeats, cancellation tokens.
3. Renderer Layer (Fetcher)
   - RendererAdapter abstraction with implementations:
     - Crawl4AIAdapter (preferred, high-level extraction + LLM strategies)
     - PlaywrightAdapter (fallback, low-level navigation/control)
4. Extractor / Normalizer
   - `newsreader_bridge` wrapping the production NewsReader extraction & visual analysis logic.
5. Analyzer (Scout Intelligence)
   - GPU/tensorrt-backed scoring; integrates with GPU orchestrator for leases and batching.
6. Persister & Dedupe
   - `persist_article()` contract that centralizes `ensure_table()` and `register_url()` and stores attachments references.
7. Control & Observability
   - Endpoints: `/start_crawl`, `/stop_crawl`, `/job_status`, `/jobs`, `/metrics`, `/health` and streaming progress APIs.
8. Resource Manager
   - Concurrency limits (browser contexts), GPU leases, memory budgets and backpressure.

Rationale for this layout:
- Crawl4AI provides a high-level, LLM-aware crawling orchestration (streaming, LLMExtractionStrategy) and manages Playwright under the hood; using it reduces boilerplate and provides schema-driven extraction.
- NewsReader already implements production-grade image/ocr/LLaVA extraction with safe quantization; reusing it avoids rework and preserves crash fixes.
- Playwright remains necessary for low-level interactions and as a fallback when Crawl4AI strategies are insufficient.

---

## Canonical Article model & persist contract

Canonical Article dict (minimal):
- `url` (str)
- `title` (str)
- `content` (str)
- `timestamp` (ISO str)
- `source_method` (str) — `'crawl4ai'|'playwright'|'newsreader'`
- `status` (`'success'|'error'`)
- `processing_time_seconds` (float)
- `provenance` (dict) — crawler, version, worker_id
- `dedupe_hash` (str)
- `news_score` (float)
- `visual_assets`: list[{path|url, type, width, height, checksum}]
- `ocr_text`: Optional[str]
- `newsreader_analysis`: Optional[dict]

Persist contract: `persist_article(article: dict) -> bool`
- Validate shape, call `scripts/db_dedupe.ensure_table()` then `register_url()`.
- If `register_url()` returns False → skip persist and return False.
- Store image assets as object-store references and only metadata in DB.

---

## Revised Persist / Handoff Contract (alignment with Canonical Dataflow)

Important change: per the canonical dataflow, UnifiedScout MUST NOT
be the authoritative source of truth for article persistence. The
Memory Agent is the canonical persistence and embedding service.

- Hand-off flow: UnifiedScout computes canonical Article dict and
  a `dedupe_hash`, performs lightweight local validations, then
  issues an MCP Bus call to the Memory Agent's `store_article`
  tool:

  ```python
  # Pseudocode
  mcp_call("memory", "store_article", [article_dict])
  # Memory Agent returns: {persisted: bool, article_id: int, reason?: str}
  ```

- Memory Agent responsibilities (canonical, per dataflow):
  - Run `scripts/db_dedupe.ensure_table()` and `register_url()` semantics.
  - Compute/confirm dedupe decisions and return whether the article
    was persisted. If Memory declines to persist (duplicate), it
    responds with `{persisted: False}` and UnifiedScout should drop
    further persistence work for that article.
  - Generate embeddings and insert into `public.articles` and
    `article_source_map` provenance records.

- UnifiedScout local responsibilities:
  - Compute `dedupe_hash` and include it in the `article_dict` passed
    to Memory to aid efficient lookups.
  - Persist only ephemeral job-level metadata locally (job progress,
    metrics); final article storage is delegated to Memory Agent.

This adjustment ensures compliance with the **Canonical Dataflow**
requirements (Memory is the ingestion/embedding authority) and avoids
duplicate DB-writing logic across agents.

---

## Modes of operation

Sequential mode (agent-friendly)
- `run_sequential(site_or_urls, ...)` — process 1–5 pages per site; immediate persist and fast handoffs.
- Intended for interactive usage and tool-like agent interactions (MCP).

Concurrent mode (throughput)
- `run_concurrent(site_or_urls, concurrency, batch_size, stream=True)` — streaming results, high throughput.
- Use Crawl4AI streaming mode and a pool of browser contexts.

Flags & tunables
- `enable_ai` (bool): whether to run LLM extraction and GPU scoring.
- `analyze_images` (bool): whether to call NewsReader visual/ocr analysis.
- GPU batch size, per-page timeout, per-domain politeness limits.

---

## Crawl4AI + Playwright recommended patterns (concrete)

Crawl4AI (use when available)
- Use `AsyncWebCrawler.arun()` for single URLs and streaming for throughput runs.
- Prefer `LLMExtractionStrategy` for structured extraction only (costly). Define a Pydantic schema for the Article model and pass it to the strategy when `enable_ai=True`.
- Use `BrowserConfig(headless=True)` and run with `CrawlerRunConfig(cache_mode=CacheMode.BYPASS)` for fresh content.

Playwright (fallback / fine-grained control)
- Use `async_playwright` with `browser.new_context()` per worker and pool contexts to reduce memory.
- Use persistent contexts for authenticated sites.
- Apply `--shm-size`, ulimit and `apt` system deps as per Crawl4AI docs (Context7 recommendations).

Deployment choices
- Embedded SDK: tight integration, better performance, simpler debugging.
- Dockerized Crawl4AI server: isolation and separation of concerns; use REST interface for decoupling if you prefer.

---

## NewsReader integration

Why include NewsReader?
- It is the production-validated page-level reader (visual + OCR + LLaVA) with correct BitsAndBytes quantization and crash mitigations.

Integration approach
- Add `agents/scout/newsreader_bridge.py` that normalizes NewsReader outputs to the canonical Article dict.
- When `analyze_images=True` or images are present, UnifiedScout calls `newsreader_bridge.process_url(url, ...)`.
- Preserve NewsReader engine configuration and quantization defaults; do not re-implement heavy model loading logic in UnifiedScout.

Testing guidance
- Unit tests for normalization (no GPU needed).
- Integration tests with mocked NewsReader API for sequential flows.

Updated NewsReader integration (concrete)

To align with the canonical dataflow where the NewsReader exposes
`/analyze_article_with_newsreader`, the `newsreader_bridge` will:

- Prefer local import (if the NewsReader package is available in the
  same runtime) to avoid network overhead; otherwise call the
  NewsReader FastAPI endpoint `/analyze_article_with_newsreader`.
- Normalize the NewsReader response to the canonical Article dict
  and include `newsreader_analysis` and `visual_assets` fields.
- Example bridge logic:

  ```python
  def process_url(url, analyze_images=True, **kwargs):
      if local_newsreader_available:
          result = local_newsreader.analyze_article_with_newsreader(url, **kwargs)
      else:
          result = requests.post(f"{NEWSREADER_URL}/analyze_article_with_newsreader", json={"url": url, ...}).json()
      return normalize_to_canonical_article(result)
  ```

- The UnifiedScout orchestrator will call `newsreader_bridge.process_url()`
  only when `analyze_images=True` or when the page contains images.

---

## Job store, cancellation and stop semantics

- Replace in-memory `crawl_jobs` with a DB-backed job table or Redis-backed job queue.
- Maintain job states: `pending`, `running`, `paused`, `cancel_requested`, `cancelled`, `failed`, `completed`.
- Cooperative cancellation: workers periodically check token; cancel and save partial results.
- `stop_crawl(job_id)` sets `cancel_requested=True` and triggers cooperative cancellation.

---

## GPU orchestration & batching

- Integrate with existing `agents/gpu_orchestrator` to request GPU leases before running NewsReader or heavy Scout analysis.
- For sequential mode, prefer CPU or light-weight LLM scoring; reserve GPU for batches or when `analyze_images=True`.
- Batch N articles per GPU lease (tunable; start with 8–16) to amortize model load time.
- Fall back to text-only analysis on `OOM` or lease denial.

---

## Tests, CI & quality gates

Unit tests (fast, no network)
- `tests/test_newsreader_bridge.py` — normalization logic
- `tests/test_persist_article.py` — mocked dedupe
- `tests/test_orchestrator_sequential.py` — sequential flow with local HTTP server

Integration tests (mocked components)
- Mock Crawl4AI REST/SDK to confirm streaming + normalization
- Mock NewsReader FastAPI for image analysis

GPU smoke tests (marked, run on GPU-enabled runners)
- Small set verifying NewsReader + Scout scoring pipelines

Static checks
- ruff/black/isort pre-commit
- Codacy per-file analyze after each file edit (repo policy)
- Run `trivy` only when dependencies change

---

## Migration & PR plan (incremental)

PR 1 — Foundation (Day 0–1)
- Add `agents/scout/config.py` and `agents/scout/newsreader_bridge.py` (wrapper/normalizer).
- Add `tests/test_newsreader_bridge_import.py` (import smoke) and minimal docs update.

PR 2 — Sequential mode & persist contract (Day 2–3)
- Add canonical `persist_article()` and wire `orchestrator.py` sequential path to call `newsreader_bridge`.
- Add `tests/test_persist_article.py` (mock DB register_url).

PR 3 — Crawl4AI integration & RendererAdapter (Day 4–8)
- Implement `Crawl4AIAdapter` and `PlaywrightAdapter` behind `RendererAdapter`.
- Add integration tests mocking Crawl4AI streaming mode.

PR 4 — Job store & cancellation (Day 9–14)
- Replace `crawl_jobs` with DB-backed jobs and implement cancellation protocol.
- Update `crawler_control` to call real stop endpoints (no fake stop).

PR 5 — GPU orchestrator integration & batching (Day 15–21)
- Integrate GPU leases; implement analyzer batching and fallbacks.
- Add GPU-marked smoke tests and update operator runbook.

PR 6 — Performance & docs (Day 22–28)
- Observability (Prometheus), performance tuning, stress tests, finalize docs & runbook.

Notes
- Start with small, reversible PRs and run per-file Codacy analyzes after each change (repo policy).
- Maintain `agents/crawler` as a thin compatibility proxy in the first release cycle to reduce operator disruption.

---

## Risk matrix

- GPU OOM / crashes: mitigation — conservative defaults, `analyze_images` toggle, GPU leases, BitsAndBytes quantization.
- Operator disruption: mitigation — compatibility proxy layer and systemd drop-in updates.
- LLM cost & latency: mitigation — enable LLM extraction only for structured extractions; use heuristics otherwise.
- Storage growth from visual assets: mitigation — store references, configure retention/archival policies.

---

## Acceptance criteria

- Functional: UnifiedScout supports `/unified_production_crawl` and returns canonical Article dicts in sequential and concurrent modes.
- Safety: No GPU crashes or OOMs during a 24-hour stress test on target hardware.
- Observability: Metrics and job status visible; stop/cancel works cooperatively.
- Backwards compatibility: `crawler_control` MCP tools continue to function; proxy present during migration.
- Performance: meet or exceed crawling throughput targets (AI-enhanced: 1+ articles/sec) and search latency requirements (<500ms).

---

## Next actions (pick one)

- Implement PR1 (newsreader_bridge + config + tests).
- Produce full patch diffs for review before any code changes.
- Run a POC: wire `bbc_ai_crawler` sequential path to Crawl4AI + NewsReader and run smoke tests.
- Produce resource & cost model for desired throughput.

---

## References

- Crawl4AI docs (AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, LLMExtractionStrategy) — Context7
- Playwright docs (async API patterns, context & isolation) — Context7
- NewsReader production README and engine implementations (`agents/newsreader/*`)
- Existing repo docs: `Crawler_Consolidation_Plan.md`, `SCOUT_*` docs, `Agent Upgrade Plan` and GPU reports

## Re-evaluation: alignment with Crawler and Ingestion Dev Plan (2025-09-14)

I re-evaluated the UnifiedScout proposal against the "JustNewsAgent Crawling
and Ingestion System Analysis & Development Plan" (2025-09-14). The dev
plan confirms many implemented capabilities and prescribes additional
requirements that must be explicitly accommodated in UnifiedScout's
architecture and implementation plan.

Key alignment changes and additions

1. Explicit support for crawl modes and throughput targets
   - UnifiedScout will preserve and expose the existing crawling modes
     described in the dev plan (except deprecated modes removed from
     the primary plan):
     - AI-Enhanced Mode (deep analysis) — target ~1+ articles/sec
     - Mixed and Dynamic Multi-Site Modes (balanced)
   - The job scheduler will select an appropriate mode per job and
     provide metrics for mode usage and throughput.

2. RSS / Feed & Social Media ingestion (Future Expansion Possibility)
   - The dev plan recommends RSS/Atom and social media monitoring as
     important future capabilities. These should be implemented as a
     dedicated Future Development project after initial UnifiedScout
     stabilization. See Appendix A (Future Development) for design,
     operational considerations, and connector guidance.

3. Job store, prioritization and high-value content
   - The dev plan highlights the lack of prioritization for high-value
     content. UnifiedScout's persistent job store will include priority
     levels and scoring metadata (source_quality_score, breaking_news_flag)
     used to re-order queues and accelerate important work.

---

## Appendix A — Future Development: RSS/Feed & Social Media Ingestion

Scope and motivation
- RSS/Atom feeds and social media streams are powerful complementary
  discovery channels that increase source coverage and early detection
  of breaking or trending content. Implementing these connectors expands
  the system's discovery surface beyond site crawling.

High-level architecture
- SourceManager extension: a pluggable connector system that supports
  multiple input types (feed, social stream, web seed).
- Feed Poller: lightweight scheduled workers that poll RSS/Atom feeds,
  perform feed discovery, de-duplicate entries, and convert feed items
  into crawl jobs for UnifiedScout.
- Social Connector: streaming ingestion (or periodic polling where
  streaming APIs restricted) that normalizes social posts into discovery
  events with optional URLs to follow.

Connector design
- Feed connector responsibilities:
  - Discover and validate feed URLs (atom/rss discovery, `<link rel="alternate">`).
  - Parse feed entries, extract canonical URLs, titles, timestamps and authors.
  - Apply source quality scoring and rate-limit policies per source.
  - Enqueue crawl jobs with priority mapping (e.g., breaking news => high priority).
- Social connector responsibilities:
  - Authenticate and respect provider TOS and rate limits (Twitter/X API, Reddit API).
  - Normalize posts into discovery events (text, urls, media, timestamp, author).
  - Use heuristics to decide whether to enqueue a crawl or only flag for later review.

Data modeling & storage
- Feed metadata stored in `sources.feeds` table with `last_polled`, `etag`, `poll_interval`.
- Social source metadata stored with `last_seen_id`, `cursor`, `rate_limit_state`.
- All discovered items converted to canonical crawl jobs and persisted in job-store.

Operational considerations
- Respect robots.txt and site-specific TOS; feed ingestion does not bypass target site politeness.
- Implement exponential backoff and back-pressure to avoid hitting remote APIs.
- Use Kafka for event streaming of discovered items; allows downstream systems to subscribe.

Testing & validation
- Unit tests for feed parsing, normalization and dedupe.
- Integration tests with a sandboxed feed server and mocked social APIs.
- Performance tests to validate poll rates and queue saturation handling.

Security & legal
- Social connectors must log consent and API usage terms; avoid automated posting.
- Rate limit compliance and API key rotation must be part of connector design.

Operational cost & scaling
- Polling many feeds increases I/O; use distributed worker pools.
- Social streaming (if used) requires stable connections and scaling gateways.
- Estimate cost for streaming connectors and plan for optional paid tiers (e.g., X API v2/v3).

Roadmap placement
- Treat as a Future Expansion project (Priority: High but deferred). Prefer to stabilize UnifiedScout core (discovery, extraction, Memory handoff, Analyst handoff) before enabling feed/social connectors.

---

## Appendix B — Deprecated: Ultra-Fast Mode

Summary
- The "Ultra-Fast" crawling mode (previously targeted at ~8+ articles/sec)
  is deprecated and removed from UnifiedScout's primary operational plan.

Rationale
- Ultra-Fast mode encourages aggressive crawling patterns that frequently
  violate site-specific politeness guidelines (robots.txt, rate limits)
  and increases the risk of being blocked, throttled, or blacklisted.
- It produces low-quality artifacts that raise ethical and legal concerns
  around scraping etiquette and content reuse.
- The marginal utility of the extreme throughput did not outweigh the
  operational and reputational risks observed during production runs.

Operational consequence
- The codebase will retain historical artifacts and the ultra-fast
  crawler scripts in `archive_obsolete_files/` for reference and audits,
  but UnifiedScout will not offer an ultra-fast mode API or expose the
  aggressive throughput settings.
- Where high throughput is genuinely needed in controlled environments,
  operators may run specialized, isolated workflows outside UnifiedScout
  with explicit operator approval and documented exceptions.

Migration notes
- Remove Ultra-Fast references from operator runbooks and systemd units.
- Update dashboards and KPIs to reflect AI-Enhanced and Balanced modes
  as the primary performance targets.


