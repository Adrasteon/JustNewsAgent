---
title: Scout Refactor — Research Summary & Action Plan
description: Research findings and a practical, incremental path to refactor Scout and Crawler into a unified, production-safe, async-first multimodal crawler stack.
status: draft
last_updated: 2025-09-28
---

Summary of intent
- The Scout Multimodal Transformation Plan (attached) defines a clear strategic direction: convert Scout from a hybrid/heuristic crawler into a GPU-first, async, multimodal crawler that produces replayable datasets for continuous learning (phases: quality contracts → observability → perception → adaptive policies → knowledge graph → RL → multimedia).
- The codebase already provides most of the building blocks: high-throughput "ultra-fast" crawlers, a generic Crawl4AI + Playwright fallback, database-driven source management, MCP tool registrations, performance monitoring, and multiple AI analysis flows (some local, some delegated to other agents).
- The practical refactor goal is twofold: (1) eliminate duplication and fragmentation between `agents/scout` and `agents/crawler`, consolidating shared crawling primitives into a central, reusable library; (2) incrementally migrate Scout toward the multimodal, GPU-first architecture described in the roadmap while preserving production stability via feature flags, staged rollout, and backward-compatible adapters.

Key findings from code & docs
- Modes and duplication
  - There are multiple, near-duplicate implementations of the "unified production crawler" in `agents/scout/production_crawlers/unified_production_crawler.py` and `agents/crawler/unified_production_crawler.py`. Both expose similar strategy selection logic (ultra_fast / ai_enhanced / generic) and share GenericSiteCrawler, MultiSiteCrawler, SiteConfig, and crawler utilities.
  - `agents/scout/production_crawlers/sites/generic_site_crawler.py` implements a robust Crawl4AI-first + Playwright fallback with modal dismissal, selector defaults, OCR/CanonicalMetadata hooks, and performance recording.

- AI & delegation
  - Scout currently contains a GPU "V2" engine in `agents/scout/gpu_scout_engine_v2.py`, but some modules (crawler unified file) purposely delegate heavy AI analysis to the Analyst agent via MCP (`_call_analyst_tool`, AsyncMCPClient). That suggests a hybrid deployment model where perception may be centralized in Analyst/GPU orchestrator.

- Async-first architecture
  - Most crawler primitives are async (Playwright, AsyncWebCrawler) and the agents use background tasks, async context managers, and explicit event-loop-safe clients (AsyncMCPClient). However, the code still includes blocking helpers and process-level cleanups (subprocess + pkill) that need careful handling when migrated or re-used.

- Observability & data capture
  - The code already records performance and has `performance_monitoring` and `_record_scout_performance`. There is an opportunity to extend these to produce replayable crawl_sessions and standardized perception records required by Phase 1.

- Run-time contracts & ops
  - There are runtime guards for DB initialization and MCP registration that make the system resilient (best-effort registration). The refactor should preserve these stability features while adding GPU readiness probes and stricter REQUIRE_GPU gating for production-critical jobs.

- Tests & staging
  - Tests and stubs exist (`tests/conftest.py`, `tests/smoke_e2e_stub.py`) to emulate MCP Bus. However, there are only limited unit tests for the crawler primitives; the refactor must expand test coverage and add synthetic multimodal fixtures (documented in the roadmap).

Constraints and non-goals
- Do not replace the system's UX or downstream APIs in one go. Backwards compatibility of MCP tool names and response schemas must be maintained until teams have validated upgrades.
- Do not centralize GPU inference into Scout if Analyst/GPU Orchestrator is required for operational reasons; instead, design adapters so perception can live either local-to-scout or centralized in Analyst.

Action Plan — incremental, code-focused path
The following action plan converts the strategic phases from the roadmap into a set of concrete, incremental engineering tasks that can be delivered in small reviews/PRs.

Phase A — Inventory & safety (1–2 sprints)
1. Inventory & duplication report (1 PR)
   - Create a short repo report enumerating duplicated files + public APIs across `agents/scout` and `agents/crawler` (list candidate files: `unified_production_crawler.py`, `generic_site_crawler.py`, `crawler_utils.py`, `performance_monitoring.py`).
   - Deliverable: Markdown report + mapping table.
2. Add runtime feature flags and compatibility adapters (small PRs)
   - Introduce `SCOUT_USE_COMMON_CRAWLER` env flag (default false) and a thin adapter in `agents/scout` that forwards to a common implementation when enabled.
   - Add `REQUIRE_GPU` gating in crawler entrypoints (soft-fail non-critical crawls first).
   - Deliverable: adapter code + unit tests confirming both paths work.
3. Add GPU readiness probe and `/ready` enhancements (small PR)
   - Implement GPU check endpoint and surface it in existing health endpoints. Reuse or add a helper in `agents/gpu_orchestrator`.
   - Deliverable: updated `/ready` contract and small unit test.

Phase B — Consolidation & shared library (2–4 sprints)
1. Create `common/crawlers` (module) and migrate shared primitives (1 PR each)
   - Move `GenericSiteCrawler`, `SiteConfig`, `MultiSiteCrawler`, `crawler_utils` helpers and `CanonicalMetadata` utilities into `common/crawlers`.
   - Keep original files as thin imports during migration (backwards-compatible shim) and add deprecation notes.
   - Deliverable: new `common/crawlers` package with tests.
2. Replace `agents/crawler` and `agents/scout` unified crawlers with shared `common` implementation behind feature flags (2 PRs)
   - Update both agents to import the common UnifiedProductionCrawler and adapt only agent-specific registration/ingestion responsibilities.
   - Deliverable: two PRs updating imports and removing duplication.
3. Add async safety linters and tests (concurrent PR)
   - Add tests that assert no blocking operations in coroutines (small runtime harness) and add ruff/flake rules or a custom lint rule to detect common blocking calls in async code.
   - Deliverable: CI job that fails on detected blocking I/O in coroutine code paths.

Phase C — Observability & data capture (2 sprints)
1. Standardize crawl session format and storage (1 PR)
   - Define `crawl_sessions` bundle schema (DOM snapshot, screenshot references, actions log, selector outcomes, model versions). Store metadata in DB and payloads in object storage.
   - Update GenericSiteCrawler to emit a session bundle every crawl (configurable flag).
   - Deliverable: schema + DB migration + updated extractor emitting session bundles to staging storage.
2. Extend performance metrics and export (1 PR)
   - Extend `performance_monitoring` to emit selector precision, DOM drift signals, and throughput breakdown by strategy.
   - Deliverable: updated exporter/metrics and Grafana dashboard updates (ops task).

Phase D — Perception & policy integration (3–6 sprints)
1. Decide perception placement (design task, 1 sprint)
   - Option 1: Centralized perception in Analyst GPU Orchestrator (single inference service) — less memory fragmentation, easier model ops.
   - Option 2: Local Scout-internal perception using `gpu_scout_engine_v2.py` — lower latency but heavier resource usage per agent instance.
   - Deliverable: design doc with trade-offs and recommended option.
2. Implement perception adapter (2–3 PRs)
   - Implement a perception client that can call either local engine or Analyst via MCP. Replace direct AI calls in unified crawler to call the adapter.
   - Deliverable: perception adapter + tests; toggled via env flags.
3. Incremental selector generator (2 PRs)
   - Add a minimal selector proposal service (use LayoutLM or simple visual+DOM heuristics) that returns candidate selectors with confidence.
   - Integrate candidate selectors into GenericSiteCrawler as one optional pre-extraction step.
   - Deliverable: selector generator service + integration test capturing selector success rate.

Phase E — Policies, replay, and training (3–6 sprints)
1. Add detailed action-logging and replay harness (2 PRs)
   - Record action traces and feed them into an offline replay environment for training and simulation.
   - Deliverable: replay service + example replay script.
2. Deploy contextual bandit policy (1–2 PRs)
   - Implement an initial LinUCB bandit that selects action variants (click, scroll, depth) and logs rewards.
   - Deliverable: bandit policy implementation + shadow-run evaluation reports.
3. Schedule nightly training jobs (ops + infra)
   - Add a pipeline (Prefect / Airflow) to periodically train selector/policy models using captured sessions.
   - Deliverable: scheduled training DAG + model registry integration.

Phase F — Multimedia & governance (3–6 sprints)
1. Add audio/video capture hooks, ASR and summarisation (parallel PRs)
   - Implement media capture pipeline and call ASR (Whisper) and video summarizers (Video-LLaMA) in staged mode.
   - Deliverable: media capture + processing microservices (shadow mode).
2. Add governance playbook & chaos tests (ops)
   - Publish model update governance and run quarterly GPU outage/chaos drills.
   - Deliverable: governance doc + scheduled chaos tests.

NewsReader agent analysis (new)
- The NewsReader V2 agent is a production-validated, screenshot-first perception engine built around LLaVA-style vision-language models. It demonstrates the practical pattern that the Scout refactor should follow for a safe, GPU-aware perception layer.

Key NewsReader patterns to adopt
- Screenshot-first processing: NewsReader captures screenshots (via Crawl4AI) and passes them to a LLaVA-based engine that performs visual/text extraction together. This avoids fragile OCR chains and aligns with the roadmap's multimodal goal.
- Quantization and memory-safe model loading: NewsReader uses BitsAndBytesConfig (INT8/INT4) and conservative max_memory limits to reduce GPU footprint; it documents and enforces these settings in `NewsReaderV2Config`.
- Singleton engine with GPU manager integration: NewsReader creates a singleton engine (`get_engine`) that requests GPU allocation from `common.gpu_manager_production` before initialization and refuses to create the engine when allocation fails. This prevents overcommit and system crashes.
- Aggressive and gentle memory management: Background memory monitor, `_attempt_memory_cleanup` and `_force_memory_cleanup`, and explicit engine cleanup ensure low crash rates observed in production.
- Production fallback strategy: NewsReader implements graceful fallbacks (no LLaVA → use lighter pipelines) and exposes `v2_compliance` and `fallback_triggered` flags in outputs to help downstream agents adapt.
- Safe initialization & lifecycle: NewsReader initializes models with device checks, logs pre/post memory states, and uses context managers to guarantee cleanup on exit.

Implications for the Scout refactor
1. Perception adapter design
   - Implement a perception adapter (PerceptionClient) that exposes the same async signature as `process_article_content(...)` and can be configured to call either:
     - Local engine (instantiate per NewsReader pattern with `get_engine()`), or
     - Centralized Analyst/Orchestrator via MCP (async RPC).
   - The adapter must request GPU allocation via `common.gpu_manager_production` before creating local engines; if allocation refused, it should return a structured fallback response rather than crash.

2. Engine lifecycle & safety
   - Reuse the NewsReader singleton + locking pattern for engine initialization with an initialization-in-progress guard and memory safety checks.
   - Add a configurable memory monitor thread (off by default) that can be enabled in staging/production to proactively free memory.

3. Quantization & model config
   - Provide default NewsReader-compatible quantization configs (BitsAndBytesConfig) in `common/perception/configs.py` and ensure `trust_remote_code` and cache-dir usage are standardized.
   - For small-perception models (e.g., OneVision-0.5B), set conservative max_memory and batched inference strategy to allow multiple agents to run on a node.

4. Outputs & schema compatibility
   - Standardize perception outputs to include: extracted_text, visual_description, layout_analysis, confidence_score, model_versions, v2_compliance, fallback_triggered, and screenshot_info. This should match NewsReader's `ProcessingResult` shaped object so downstream Scout code needs minimal changes.

5. Operational policies
   - Require GPU readiness probes and `REQUIRE_GPU` gating for any production path that depends on local perception. If GPU not available, fall back to a 'degraded' mode that either queues the job or calls centralized Analyst.
   - Use NewsReader's aggressive cleanup and conservative quantization as the default safe path; make more aggressive settings opt-in under feature flags.

Concrete amendments to the action plan
- Phase D (Perception & policy integration): Prefer starting with a NewsReader-pattern local-perception prototype using the OneVision-0.5B model with INT8 quantization as the first proof-of-concept. Parallel-track a centralized Analyst service design for larger models.
- Add a new Task: "Perception adapter prototype (NewsReader pattern)" that implements the singleton engine, GPU request workflow, BitsAndBytesConfig defaults, and the `process_article_content` async signature. Toggle via `PERCEPTION_MODE=local|central`.
- Update Phase A immediate steps to include: audit `common.gpu_manager_production` usage and ensure a stable, testable GPU allocation interface; add unit tests simulating allocation denial.

New immediate actions (this sprint)
- Prototype PerceptionClient that wraps NewsReader's `process_article_content` behavior but lives in `common/perception` and supports both local and MCP-based modes.
- Add quantization templates under `common/perception/configs.py` for int8 and int4, plus a small doc referencing `agents/newsreader/INT8_QUANTIZATION_RATIONALE.md`.
- Create a safety test that deliberately simulates low GPU conditions and confirms the perception adapter returns a graceful fallback rather than raising an unhandled exception.

Repository TODOs added
- Add `common/perception` package scaffolding and move or reference NewsReader's safe initialization patterns. (todo created)
- Add GPU allocation unit tests that mock `request_agent_gpu` and verify allocation/rejection flows. (todo created)
- Standardize perception output schema (ProcessingResult-like dataclass) in `common/perception/schema.py`. (todo created)

Conclusion
- NewsReader is the closest production example of the multimodal, memory-safe perception engine the Scout refactor seeks. The refactor should intentionally reuse NewsReader's patterns for engine lifecycle, quantization, memory monitoring, and fallback strategies.
- Implementing a perception adapter modeled on NewsReader will reduce risk, speed validation, and allow a clean toggle between local and centralized perception while preserving MCP API compatibility.

Rollout strategy and safety
- Feature flags are required for every major change. Each refactor step should start as a flag-enabled opt-in path, shadowed against production traffic, and only toggled global when metrics are stable.
- Preserve MCP tool compatibility by providing thin adapter shims that maintain existing input/output signatures while the underlying implementation migrates.
- Start migration with low-risk components (shared utils → common library) and prove tests/metrics before moving the orchestration layer.

Immediate next actions (this week)
1. Create the duplication/inventory report (Phase A.1) and open a PR with the report markdown. (owner: me/you)
2. Add `SCOUT_USE_COMMON_CRAWLER` env flag and a compatibility shim in `agents/scout` that imports from `common/crawlers` when enabled. Unit test both paths. (owner: engineer)
3. Add GPU readiness probe endpoint and integrate it into `/ready` checks for Scout & Crawler. (owner: ops/infra)
4. Draft the perception placement design doc and circulate to Analyst, Scout, and Ops teams. (owner: architect)
5. Prototype PerceptionClient that wraps NewsReader's `process_article_content` behavior but lives in `common/perception` and supports both local and MCP-based modes. (owner: engineer)
6. Add quantization templates under `common/perception/configs.py` for int8 and int4, plus a small doc referencing `agents/newsreader/INT8_QUANTIZATION_RATIONALE.md`. (owner: engineer)
7. Create a safety test that deliberately simulates low GPU conditions and confirms the perception adapter returns a graceful fallback rather than raising an unhandled exception. (owner: engineer)

Follow-up todos added to the repo board (high level)
- Create `common/crawlers` package and migrate GenericSiteCrawler.
- Add async blocking I/O detection tests and CI rule.
- Implement crawl session bundle schema and object storage exporter.
- Prototype perception adapter that toggles between local and Analyst inference.
- Add `common/perception` package scaffolding and move or reference NewsReader's safe initialization patterns.
- Add GPU allocation unit tests that mock `request_agent_gpu` and verify allocation/rejection flows.
- Standardize perception output schema (ProcessingResult-like dataclass) in `common/perception/schema.py`.

Estimated roadmap & resource note
- Minimal consolidation (Phase A–B small tasks): 4–8 engineer-weeks.
- Observability + data capture (Phase C): 3–5 engineer-weeks + storage/infra work.
- Perception & policy integration (Phase D–E): 6–12 engineer-weeks (GPU infra, model ops, training pipelines). Parallelization of tasks reduces calendar time.

Risks & mitigations (code-specific)
- Duplicate logic may hide behavior differences (sanity: run end-to-end smoke tests on a staging snapshot before switching).
- Playwright/process cleanup can kill unrelated processes if not constrained; ensure process filtering is PID/session-scoped and optionally namespaced per-agent.
- Database migrations for performance tracking must be backwards-compatible; use feature flags and migration scripts with reversible steps.

Concluding recommendation
- Treat this refactor as a multi-phase migration: (1) extract and centralize shared primitives (low-risk, high-value), (2) add production-grade observability & session capture, (3) incrementally enable perception and policy components behind flags, and (4) operationalize training & governance.
- Start with the inventory + adapter PRs this sprint so the team can validate the small, reversible changes and build confidence before tackling GPU/perception complexity.


