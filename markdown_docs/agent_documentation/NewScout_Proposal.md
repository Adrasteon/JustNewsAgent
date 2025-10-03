---
title: NewScout — Next-Generation Unified Crawler & Page Intelligence
description: A forward-looking proposal to design NewScout — a from-scratch, production-ready unified crawling, rendering, extraction and analysis service that replaces and improves on existing Scout/Crawler/NewsReader functionality.
tags: [proposal, architecture, crawler, newsreader, ai, design]
status: draft
last_updated: 2025-10-01
---

# NewScout — Next-Generation Unified Crawler & Page Intelligence

Date: 2025-10-01
Author: GitHub Copilot (design)

Executive summary
- NewScout is a single, modern service designed from the ground up to discover, fetch, render, extract, and normalize news content at web scale while respecting ethics, site policies, and resource constraints.
- It favors an event-driven, microservice-friendly architecture (Kafka + k8s) with a single control-plane API and pluggable runtime workers for rendering (Playwright/Crawl4AI-like), multimodal extraction (NewsReader-grade), lightweight online triage, and authoritative handoff to canonical services (Memory, Analyst).
- Key innovations: adaptive model selection, incremental multimodal extraction, RAG-assisted structured extraction, GPU-batch orchestration with lease pooling, automatic politeness enforcement, and continuous cost/performance tuning via feedback loops.

Principles and goals
- Single responsibility: NewScout discovers and canonicalizes content; heavy analysis and long-term storage are delegated to specialized agents (Memory, Analyst).
- Politeness-first: Strict robots/terms enforcement and adaptive backoff; default conservative crawling policies.
- Resource-aware: Dynamic GPU/browser context allocation with global orchestrator coordination and fine-grained batching.
- Observability & safety: Full provenance, traceable ML decisions, human-in-the-loop controls, and gating on high-risk extraction.
- Extensibility: Pluggable connectors (web, RSS, social streams), adaptable extraction strategies, and simple migration paths.

Architecture overview
- Control Plane (API & Scheduler)
  - Exposes MCP tools: `start_job`, `stop_job`, `job_status`, `get_capabilities` and HTTP admin endpoints for operators.
  - Enqueues canonical Job events to Kafka topics: `jobs.new`, `jobs.update`, `jobs.cancel`.

- Worker Plane (stateless, autoscaled)
  - Renderer Workers: PlaywrightPool and optional Crawl4AI-integration worker. Responsible for page rendering, screenshotting, and raw DOM capture.
  - Extractor Workers: Multimodal extraction modules that run NewsReader-grade pipelines (vision+ocr+llava + DOM parsing) with lightweight local caches for model artifacts.
  - Triage Workers: Fast heuristics (no GPUs) that compute `dedupe_hash`, `news_score` and decide `analyze_deeply` flags.
  - Persist/Bridge Workers: Package canonical Article dicts and call Memory via MCP: `memory.store_article(article)`.

- Coordination Services
  - Kafka (event streaming): reliable, decoupled event transport and buffering for discovered items and job progress.
  - Redis (ephemeral state): leader election, job locks, worker heartbeats, short-lived caches.
  - GPU Orchestrator (external or integrated): central leasing API to request/release GPU resources.

- Data & storage
  - Artifacts: small canonical JSON records in Postgres via Memory; large binary assets stored into object store (S3) and referenced by `visual_assets` metadata.
  - Provenance: every artifact stores `source`, `worker_id`, `tooling_version`, `timestamp`, and `extraction_strategy` for audits and ML retraining.

Component deep-dive

1. Discovery & Queuing
- Source Registry: maintains `sources` table with per-source policy (crawl_delay, allowed_paths, feed_urls, owner, priority).
- Connectors (pluggable): web seeds, optional RSS/social connectors (Appendix — deferred for later), and webhooks.
- Job model: `Job(id, type, seeds, mode, priority, created_by, policy)` where `mode` in {sequential, concurrent, balanced}.
- Scheduler: decides when to launch workers, enforces politeness, and applies adaptive throttling using real-time error/ban detection.

2. Rendering (Renderer Workers)
- PlaywrightPool: a managed pool of browser contexts per node. Uses `new_context()` isolation and reuses lightweight contexts for sites that permit it.
- Headless isolation policies: sandboxing, per-domain cookie isolation, optional stealth mode when legally permitted.
- Crawl4AIAdapter: optional high-level orchestration strategy that can be used for complex extraction strategies (LLM-extraction) when `enable_ai=True`.
- Outputs: raw HTML, cleaned HTML, rendered screenshots, network logs, and console logs (sensitive info redaction applied before persistence).

3. Extraction (Extractor Workers)
- Multimodal Extractor Pipeline:
  - DOM cleaning & segmentation (boilerplate removal using heuristics + model-assisted parsing).
  - Text extraction (main body, headings, metadata, bylines, dates).
  - Visual analysis (image captioning, OCR via LLaVA/BLIP variants): executed only when `analyze_images=True` or `page_has_images=True`.
  - LLM-structured extraction: RAG-assisted LLM (local or private endpoint) to produce structured Article schema; LLM calls are gated and cost-accounted.
- Incremental extraction: workers store intermediate artifacts in object store; post-processing stages can pick up partial results.

4. Triage & Handoff
- Triage computes `dedupe_hash` (canonicalized URL + content fingerprint), `news_score` (fast model or rule-based), and `analyze_deeply` decision using a lightweight ensemble.
- If triage approves, NewScout calls `memory.store_article(article)` via MCP; Memory returns `article_id` and dedupe status.
- If Memory declines persist (duplicate), NewScout updates job state and continues or abandons follow-ups.

5. Analyst handoff
- For deep analysis, NewScout issues MCP event to Analyst: `analyst.analyze(article_id, tasks=[ner,sentiment,bias])`.
- NewScout does not replicate Analyst duties; it supplies only `article_id` and relevant metadata.

Resource orchestration innovations
- Lease pooling: GPU Orchestrator grants pooled leases (e.g., 1 lease for N workers) to amortize model load/unload costs.
- Adaptive batching: dynamic batch size for GPU inference based on current memory and observed latency, using online learning to tune batch size.
- Cooperative eviction: workers register models in a shared registry; orchestrator can signal model eviction to avoid OOMs.
- Elastic Playwright scaling: context pooling and smart pre-warming of browser contexts for sites with heavy JS.

Advanced techniques & innovations
- RAG-assisted structured extraction: use small retrieval-enhanced LLMs to extract fields with higher reliability and reduced token costs.
- Multimodal fusion at token-level: fuse OCR text and visual captions into the same LLM context to produce richer `newsreader_analysis`.
- Differential model selection: cheap models on CPU for triage; quantized LLMs (8-bit AWQ) for extraction; opt for remote LLMs only when local models cannot fulfill requirements.
- Continuous optimization loop: measure cost/latency/quality and use a small RL-style optimizer to select extraction strategy per-site.
- Ethical policy engine: automatic policy enforcement rules (no-go domains, max images per site, data retention policies).

Data model — canonical Article (expanded)
- `article_id` (int, assigned by Memory)
- `url`, `canonical_url`, `title`, `subtitle`, `authors`, `published_at`
- `content`, `summary` (optional, generated)
- `dedupe_hash`, `source_id`, `source_quality_score`
- `visual_assets`: [{key, s3_path, width, height, checksum, alt_text}]
- `metadata`: {http_headers, response_status, fetch_time_ms, browser_context}
- `provenance`: {worker_id, pipeline_version, extraction_strategy}
- `quality_metrics`: {readability, coverage, confidence}
- `analysis_flags`: {analyze_deeply, breaking_news, priority}

APIs and MCP tools
- Exposed tools (UnifiedScout agent):
  - `start_job(args, kwargs)` — schedule or run a job
  - `stop_job(job_id)` — cooperative stop
  - `job_status(job_id)` — progress and partial results
  - `get_capabilities()` — returns modes, available local models, and extraction profiles
- Event topics (Kafka): `jobs.*`, `articles.extracted`, `articles.persisted`, `alerts.breaking`

Security, compliance & privacy
- Robots & TOS enforcement engine: per-source policy that can block or downgrade extraction.
- PII detection & redaction: pipelines detect likely PII and redact before storage or flag for human review.
- Access control: operator roles and audit logs for all data access and extraction decisions.
- Retention & summarization: raw OCR and high-fidelity visual assets have configurable retention; summaries replace raw data after TTL.

Testing & validation
- Unit & integration tests using local Playwright testbed and mocked Memory/Analyst endpoints.
- Synthetic user testing harness to detect blocking and crawling failures.
- Continuous benchmarks: per-site latency/throughput harness and GPU smoke tests gated behind markers.

Deployment & operations
- Containerized (Docker) services deployed on k8s with HPA for workers.
- Kafka for durable buffering; Redis for leader election and short-term locks.
- Observability: Prometheus metrics (per-job latency, fetch errors, GPU occupancy), Grafana dashboards, and traces via OpenTelemetry.
- Operator CLI and runbook: operators can inspect job state, cancel jobs, and set per-source policies.

Migration strategy
- Greenfield approach: NewScout runs in parallel; initial mode: sequential-only with `analyze_images=False` to validate flows.
- Gradual ramp: add concurrent workers for controlled sites; enable `analyze_images=True` selectively.
- Canonical handoff: Memory remains canonical; NewScout only writes ephemeral job data locally.

Roadmap & PR plan (initial)
- PR-A (Week 0): control-plane skeleton, Kafka topics, simple renderer worker with Playwright pool, and triage worker.
- PR-B (Week 1): newsreader-like extractor module (local quantized models), persist handoff to Memory (mocked), basic tests.
- PR-C (Week 2): GPU lease integration, batch inference path, and smoke tests on target hardware.
- PR-D (Week 3+): observability, operator CLI, stress tests and selective site enablement.

Acceptance criteria
- Correctness: canonical Article objects are persisted via Memory and `article_id` returned.
- Safety: no policy violations detected in a 72-hour dry-run across 50 sites.
- Robustness: cooperative cancellation completes within 30s for running fetch tasks.
- Performance: stable AI-enhanced extraction at ~1+ articles/sec per cluster unit (configurable); search latency <500ms when retrieving via Memory.

Appendix — future connectors and ethical considerations
- RSS/social connectors as separate pluggable services (deferred); design notes and pricing estimation to be created later.
- Ethical guidelines for scraping, rate-limiting, and content reuse — require legal review per target jurisdiction.

References & inspiration
- Crawl4AI and Playwright (Context7): adaptive crawling, BrowserConfig and LLMExtraction strategies.
- NewsReader production design in-repo: quantization strategies and LLaVA integration.
- Modern data platforms: Kafka, Redis, Kubernetes, OpenTelemetry.


