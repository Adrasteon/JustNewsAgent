# Potential Development Paths

This document captures a compact summary of recent analysis and recommendations about the project's crawlers, the new `sources` schema, and suggested engineering/product paths. It is intended as a snapshot for planning and as a checklist for short-term implementation work.

## 1. Conversation Overview

- Primary objectives:
  - Identify which crawler classes the orchestrator uses.
  - Compare `SOURCES_SCHEMA_AND_WORKFLOW.md` database changes with crawler scripts and propose workflows to use source URL data to expand beyond BBC.
  - Incorporate strategic docs in `/docs/` and choose among three product modalities (BBC-first, multi-site clustering, comprehensive archive).

- Session context:
  - The repository was inspected for the orchestrator, BBC crawlers, and the `SOURCES_SCHEMA_AND_WORKFLOW.md` specification.
  - The outcome was a phased hybrid recommendation (start BBC-first, add clustering, then archive-scale research).

## 2. Technical Foundation

- Crawl4AI is the primary crawler/extraction technology in the stack; Playwright is used as a pragmatic fallback for sites or flows where Crawl4AI is not available or when fine-grained browser interaction is required (the existing `UltraFastBBCCrawler` and `ProductionBBCCrawler` use Playwright patterns).
- The proposed architecture uses ingest-time transactional mapping into `public.sources` and `public.article_source_map`, with a denormalized `articles.source_id` for analytics.
- Canonical selection rule: choose the source record with the highest confidence, then most recent, then prefer `matched_by = 'ingest'`.

## 3. Codebase Status (key files)

- `agents/scout/production_crawlers/orchestrator.py`:
  - Orchestrates ultra-fast and AI-enhanced crawls; dynamically loads site crawlers and constructs `self.sites['bbc']` when available.

- `agents/scout/production_crawlers/sites/bbc_crawler.py`:
  - `UltraFastBBCCrawler` returns article dicts and JSON summaries; includes heuristics and modal dismissal JS. Does not currently upsert into `public.sources`.

- `agents/scout/production_crawlers/sites/bbc_ai_crawler.py`:
  - `ProductionBBCCrawler` integrates AI analysis (PracticalNewsReader) and writes JSON summaries; also does not upsert into `public.sources`.

- `markdown_docs/agent_documentation/SOURCES_SCHEMA_AND_WORKFLOW.md`:
  - Canonical schema for `public.sources`, `public.source_scores`, and `public.article_source_map` and ingest/backfill workflow (upsert sources, insert article_source_map, compute canonical, update `articles.source_id`).

## 4. Problem & Recommended Fixes

- Problem:
  - Crawlers (Crawl4AI outputs and Playwright-based fallbacks) currently emit article payloads but do not consistently include canonical source metadata (e.g., `url_hash`, `domain`, canonical link) nor do they perform DB upserts into `sources`/`article_source_map`.

- Recommended fixes:
  - Enrich crawler outputs with `url_hash`, `domain`, canonical link, `publisher_meta`, `paywall_flag`, and `extraction_metadata`.
    - Note on the `paywall_flag`: this flag is primarily an operational signal that the site or page is not crawlable under our ethical constraints. It should not be used to drive logic that attempts to bypass or defeat paywalls; instead, route paywalled content to snapshot-only workflows and human review where appropriate.
  - Implement an ingest adapter/library (see guidance below) to map crawler payloads to DB-ready payloads and perform transactional upserts/inserts into `public.sources` and `public.article_source_map`.
  - Implement canonical selection centrally (database stored procedure or a coordinated agent-driven transaction) to set `articles.source_id` following the canonical selection rule.

## 5. Actionable Next Steps (priority order)

1. Crawler enrichment (low-risk, quick win):
  - Update Crawl4AI extraction configs and the Playwright fallback crawlers (`UltraFastBBCCrawler`, `ProductionBBCCrawler`) to include canonical metadata and a stable payload shape (`url`, `url_hash`, `domain`, `canonical`, `publisher_meta`, `paywall_flag`, `extraction_metadata`, `confidence`).

2. Ingest adapter (library within the agent framework):
  - Prefer adding an ingest adapter as a shared library inside the Scout agent or `agents/common` (e.g., `agents/scout/production_crawlers/ingest_adapter.py` or `agents/common/ingest.py`) rather than creating a new microservice. This reduces operational/maintenance burden and leverages the existing agent orchestration and autonomy.
  - The adapter should expose a simple transactional API that other agents can call (for example, the Scout agent after a crawl, or a balancer/mcp_bus-driven worker).

3. Transactional canonical selection (database or coordinated agent):
  - Implement canonical selection either as a database stored procedure (recommended for atomicity) or as a coordinated transaction orchestrated by existing agents (for example using `mcp_bus` to request and confirm the canonical write). Ensure the rule (confidence → recency → matched_by) is enforced and auditable.

4. Paywall handling & routing:
  - Use the `paywall_flag` as an ethical indicator: do not attempt to bypass paywalls. Route flagged pages into snapshot-only storage and the human-review queue (for `chief_editor`/`fact_checker`) or a separate evidence-only pipeline.

5. Reuse other agents & shared capabilities instead of new services:
  - The JustNews system includes many specialized agents (for example `analyst`, `balancer`, `chief_editor`, `critic`, `fact_checker`, `mcp_bus`, `memory`, `newsreader`, `reasoning`, `synthesizer`). Prefer invoking these agents or their shared libraries for ingestion, review, canonical selection, evidence storage, and downstream processing rather than adding new microservices. This lowers maintenance and keeps the system autonomous.

6. Expand to more sources & clustering:
  - Once ingestion is solid, add per-source Crawl4AI configs and a clustering pipeline (implemented inside existing agents or `agents/common`) to group the same article across multiple outlets.

## 6. Quick tests & validation

- Unit tests for canonicalization logic (confidence ties, recency, matched_by preference).
- End-to-end smoke test: run a crawl, pass payload to the ingest adapter, verify `sources` and `article_source_map` inserts and `articles.source_id` assignment.

## 7. Next decision for the team

Which quick task should we start with? I recommend starting with crawler enrichment and the ingest adapter (step 1 + 2). After you confirm, I can implement the code changes and run unit tests locally.

## 8. Provenance & Evidence

- Files inspected to prepare this document:
  - `agents/scout/production_crawlers/orchestrator.py`
  - `agents/scout/production_crawlers/sites/bbc_crawler.py`
  - `agents/scout/production_crawlers/sites/bbc_ai_crawler.py`
  - `markdown_docs/agent_documentation/SOURCES_SCHEMA_AND_WORKFLOW.md`
  - `docs/IMPLEMENTATION_PLAN.md`, `docs/JustNews_Plan_V4.md`, `docs/New_Blueprint_Agents.md`


---

Generated: 2025-08-29
