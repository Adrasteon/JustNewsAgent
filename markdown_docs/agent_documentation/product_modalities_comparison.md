---
title: Product Modalities Comparison
description: Auto-generated description for Product Modalities Comparison
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# Product Modalities Comparison

This document compares three high-level product modalities the JustNews system can pursue, aligns each with the repository's current code and DB schema, and gives recommended priorities, milestones, risks, tests, and resource implications.

Modalities
- BBC-first: Focus the product on high-quality, reliable coverage from the BBC (tight scope).
- Multi-site clustering: Ingest multiple news outlets and cluster matching articles across publishers to surface consolidated coverage and provenance.
- Comprehensive research archive: Aggressively crawl, store, and index a broad web-scale archive of news content for research and analysis (higher cost, longer timeline).

## Executive summary / Recommendation

Choose a hybrid phased approach:
1. Phase 1 (BBC-first): ship quickly with BBC-focused crawls, canonical ingestion, and editorial review paths. This minimizes scope and operations while delivering a polished product.
2. Phase 2 (Multi-site clustering): extend ingestion to more trusted outlets with per-source Crawl4AI configs and implement clustering for cross-source dedupe and canonical selection.
3. Phase 3 (Comprehensive archive): ramp up to research-scale archiving and KG building after robust infringement, paywall, provenance, and privacy controls are in place.

This path balances speed-to-value, legal/ethical risk, and engineering effort.

## The "AI News Reporter Agent" — a system-level concept

In the New Blueprint, the term "AI News Reporter Agent" describes the entire JustNews system as a product: an autonomous, auditable, open-source news reporting system built from many cooperating agents, infrastructure components (crawlers, evidence ledger, KG, DB, model registry), and human-in-the-loop workflows. It is not a single agent process or service inside the repository.

Key clarifications and responsibilities:

- System vs. agent: treat the distinct agents in the codebase (`scout`, `fact_checker`, `analyst`, `chief_editor`, `balancer`, `mcp_bus`, `memory`, `synthesizer`, `critic`, etc.) as reusable components that implement pieces of the AI News Reporter product. Do not conflate the product-level AI News Reporter Agent with any single agent process; instead, orchestrate these components to realize the product-level goals.

- Product responsibilities: the system-level AI News Reporter must own end-to-end mandates: factual accuracy, provenance and evidence, bias elimination, sentiment-free tone, transparency, and ethical crawling (honor robots.txt and paywall semantics).

- Orchestration & coordination: use multi-agent orchestration frameworks (CrewAI / AutoGen / LangChain patterns) and the existing `mcp_bus` to coordinate task flows, model rollouts, and transactional updates (for canonical selection and article provenance). The product-level orchestrator may be realized via an orchestrator process or a composition of `balancer`, `mcp_bus` and role-based Crews, not a newly invented monolithic microservice.

- Auditability & provenance: enforce evidence-capture at every step (crawler snapshots from Crawl4AI, `article_source_map` entries, evidence ledger records, KG provenance). When model outputs influence editorial or canonical decisions, record model id, version, and metrics in an evidence trail.

- Governance & safety: implement gating for model promotions (validation suites, canary rollouts coordinated by `balancer`, and manual approval hooks for `chief_editor` when necessary). Maintain clear ethical rules preventing paywall circumvention and aggressive scraping.

- Self-learning & retraining: the product (system) owns the training lifecycle: collection of labeled signals, active learning loops, scheduled retraining, validation, registry updates, and coordinated deployment. Use `training_system/`, `agents/common` helpers, and the evidence ledger rather than ad-hoc services.

- User-facing behaviour: the system must surface provenance, confidence, and explanation for generated outputs (which KG evidence, which sources, and what validation checks passed). This is a product requirement implemented by composing agent outputs, not by a single "reporter" process.

Mapping to Modalities:

- BBC-first: the system is the product shipped to users; `scout` (Crawl4AI + Playwright fallbacks) and `chief_editor` workflows implement the narrow-for-scope reporting channel.
- Multi-site clustering: the system-level clustering pipeline is composed from `scout` (ingest), `analyst`/`reasoning` (embedding + clustering), and `fact_checker` (validation) agents working together.
- Comprehensive archive: the product integrates large-scale Crawl4AI ingestion, KG enrichment (Nucleoid / Neo4j), and researcher APIs — again, composed from many agents and shared services.

## Mapping to current code & infra

- Crawling: primary tech is Crawl4AI; Playwright is used as fallback. The repo contains Playwright-based BBC crawlers under `agents/scout/production_crawlers/sites`.
- Ingestion: `markdown_docs/agent_documentation/SOURCES_SCHEMA_AND_WORKFLOW.md` defines `public.sources` and `public.article_source_map` and suggests ingest-time canonical selection.
- Agents: The system already has many specialized agents (`analyst`, `balancer`, `chief_editor`, `critic`, `fact_checker`, `mcp_bus`, `memory`, `newsreader`, `reasoning`, `synthesizer`) — prefer reusing them rather than adding microservices.

## Detailed comparison

### 1) BBC-first

Goal
- Rapidly deliver a high-quality, editorially-vetted feed based primarily on BBC coverage.

Why choose this first
- BBC is a large, high-quality source with consistent structure (easier extraction), low legal risk if crawled ethically, and strong editorial appeal.
- Faster to implement: only a few Crawl4AI configs + Playwright fallbacks and immediate editorial review workflows.

Work required
- Crawler enrichment (Crawl4AI templates + Playwright fallback) to emit canonical metadata (`url_hash`, `domain`, `canonical`, `publisher_meta`, `paywall_flag`, `confidence`).
- Ingest adapter library to upsert into `public.sources` and insert `article_source_map`.
- Wire in editorial/human-review flows (use `chief_editor`, `fact_checker`, `analyst` agents).
- Add smoke tests and unit tests for canonical selection logic.

Milestones & timeline (suggested)
- Week 0–1: Implement crawler-enrichment and stable payload shape; unit tests for payload.
- Week 1–2: Implement ingest adapter library and a simple transactional ingest flow (db stored proc or agent-coordinated write); smoke e2e tests.
- Week 2–4: Editorial UI and human-review integration (chief_editor & fact_checker agents), A/B testing, monitoring.

Risks & mitigations
- Paywall/robot rules: use `paywall_flag` and robots.txt checks; do not bypass paywalls.
- Source drift: maintain Crawl4AI templates and Playwright fallbacks; add monitoring for extraction failures.

Success criteria
- End-to-end pipeline from crawl → ingest → canonicalization → editorial review completes for BBC with < 5% manual fixes after 2 weeks.

### 2) Multi-site clustering

Goal
- Ingest multiple trusted outlets and cluster matching articles to present consolidated views and source provenance.

Why choose this second
- Adds clear product value: shows how multiple outlets cover the same story and surfaces primary sources.
- Helps the system learn canonical selection under real multi-source conditions.

Work required
- Per-source Crawl4AI configs and Playwright fallbacks for each new outlet.
- Clustering pipeline (canonicalization + dedupe) implemented in an existing agent (prefer `analyst` or `reasoning`) or `agents/common`.
- Enhanced `article_source_map` metadata (confidence, matched_by, notes) and scoring.
- Tests for clustering edge cases (near-duplicates, syndicated content, rewrites).

Milestones & timeline (suggested)
- Month 0–1: Add ingestion + Crawl4AI config for 5 additional outlets (e.g., Reuters, AP, Guardian, NYTimes, CNN) and ensure payload parity.
- Month 1–2: Implement clustering algorithm, initial metrics (precision/recall) and canonical selection improvements.
- Month 2–3: Integrate cluster-based UI and provenance display; monitoring and failover.

Risks & mitigations
- Syndicated content / wire stories: add heuristics for syndication source detection (byline patterns, syndication markers).
- Increased maintenance: prioritize trusted list and incremental on-boarding; use existing agents to share logic and reduce duplication.

Success criteria
- Clustering accuracy (precision) > 85% on a 1k-article evaluation set, and canonical selection precision > 90% on cluster heads.

### 3) Comprehensive research archive

Goal
- Build a broad, research-grade archive with KG, full-text indexing, and long-term storage of historical news content.

Why choose this last
- Highest value for research, but largest cost and legal/ethical complexity (archival rights, paywalls, PII, storage cost).

Work required
- Scale crawling with Crawl4AI to many domains and efficient storage (S3 + cold storage lifecycling) plus metadata indexing.
- Build knowledge graph (KG) and evidence ledger integration (rdflib/neo4j/dgraph) and provenance chains.
- Privacy and legal compliance processes (data retention policies, takedown workflows).
- Significant compute and storage resources; training and evaluation infrastructure for KG models.

Milestones & timeline (suggested)
- Months 0–3: Pilot 1M-article ingest, evidence ledger snapshots, and basic KG schema.
- Months 3–9: Scale indexing, KG enrichment, and researcher APIs.

Risks & mitigations
- Legal/compliance: consult legal; implement opt-out and takedown processes; avoid paywall circumvention.
- Cost: budget for storage, compute, and retrieval costs; implement tiered storage.

Success criteria
- A 1M-article pilot ingest with complete provenance and evidence-snapshots; KG populated with core entity relations and queryable by researchers.

## Cross-cutting considerations

- Ethical crawling & paywalls
  - Always honor robots.txt and publishers' terms. `paywall_flag` is an ethical signal; do not use it to attempt circumvention.

- Agent re-use & low-maintenance architecture
  - Prefer re-using existing agents' capabilities and shared `agents/common` utilities instead of introducing new microservices. Use `mcp_bus` for coordination where transactional atomicity or cross-agent commits are needed.

- Canonical selection rule
  - Confidence → Recency → matched_by preference (prefer `ingest`) — implement in DB stored proc or agent-coordinated transaction.

- Tests & monitoring
  - Unit tests for canonicalization, clustering tests, extraction monitoring, paywall detection coverage, and end-to-end smoke tests.

## AI/ML throughout the JustNews system

This project is model-driven: agents rely on specialized models (see `markdown_docs/agent_documentation/AGENT_MODEL_MAP.md`) and follow the repository's model usage/caching guidelines (`markdown_docs/agent_documentation/MODEL_USAGE.md`). The following items describe how AI/ML should be integrated across the three modalities and the system as a whole.

- Agent-specialized models
  - Each agent has a small set of specialized models (scout, fact_checker, analyst, memory, synthesizer, etc.). Keep models per-agent and follow `MODEL_USAGE.md` for caching, atomic installs, and shared helper APIs to avoid duplication and runtime contention.

- Self-training loops and on-the-fly training
  - Implement continuous learning loops where agents collect labeled signals from downstream workflows (editorial decisions, fact-checker outcomes, critic feedback, user interactions) and feed those signals into lightweight retraining or fine-tuning jobs.
  - Early phases (Phase 1) should collect signals and store them in a labeled dataset (evidence ledger + memory agent) without immediate model updates. Use this period to design data curation and validation pipelines.
  - Phase 2 should enable scheduled incremental fine-tuning runs (e.g., nightly or weekly) that produce new model checkpoints; these are validated automatically (unit tests, holdout eval, and smoke E2E) before being promoted to production by existing agent coordination (for example `balancer` or `chief_editor` coordinating rollout via `mcp_bus`).
  - Advanced: support very small ‘on-the-fly’ fine-tuning for lightweight adapters (LoRA/QLoRA) or prompt-tuning on per-agent basis for quick specialization when high-value signals exist. These must be gated by automated validation and limited resource sandboxes.

- Training infra & model registry
  - Reuse existing `agents/common` helpers for model downloads and caching. Maintain a simple model registry (can be a DB table or flat JSON in `agents/common`) keyed by agent and semantic version, and use the `AGENT_MODEL_MAP.md` as the canonical mapping.
  - Prefer scheduled training pipelines orchestrated by existing components (training coordinator in `common` or `training_system/`) rather than new microservices. Training jobs should write artifacts to canonical model paths and emit a manifest with metrics and evaluation results.

- Validation & safety
  - Automated validation suites must include: factuality checks via `fact_checker`, toxicity checks via `critic`, and performance benchmarks (precision/recall on canonical selection and clustering tasks).
  - A/B rollout and canary testing should be coordinated by the `balancer` agent; rollbacks should be automatic on metric regression.

- Agent-driven model usage and APIs
  - Expose model inference via small synchronous APIs inside agent processes (use `agents/common` helpers for shared models) to minimize cross-process RPCs. For heavier ops (training, large-batch embedding generation), use the `mcp_bus` to hand off jobs to worker agents with GPU access.
  - Track model provenance in the evidence ledger when model outputs affect canonical decisions or editorial content.

- Resource and cost considerations
  - On-the-fly fine-tuning and frequent retraining require GPU capacity and storage for model checkpoints. Begin with conservative schedules (nightly/weekly) and monitor gains before increasing cadence.

## Integration of AI/ML with Modalities

- BBC-first
  - Use agent models to improve extraction quality, paywall detection, and initial classification of article types. Collect editorial feedback to build labeled datasets for the BBC domain.

- Multi-site clustering
  - Use embedding models (from `MODEL_USAGE.md` recommended sentence-transformers) to produce dense vectors for clustering. Continuously fine-tune the embedder on in-domain pairs derived from human-labeled clusters.

- Comprehensive archive
  - Use larger-scale models and KG models for entity linking and relation extraction. Retrain KG models periodically with curated evidence from the evidence ledger.

## Cost & resource implications (high-level)

- BBC-first: Low-to-moderate engineering effort, small infra increase, fast time-to-value.
- Multi-site clustering: Moderate engineering and maintenance, moderate infra increase, more monitoring and extraction templates.
- Comprehensive archive: High infra cost (storage & compute), legal overhead, long timeline and research resources.

## Recommended next steps

1. Approve phased approach and pick Phase 1 scope (which BBC sections / feeds to support first).
2. Task: Implement crawler enrichment + ingest adapter library in `agents/common` or `agents/scout` (I can implement this next).
3. Create a short test-plan for Phase 1 including canonicalization unit tests and an end-to-end smoke test.

---

Generated: 2025-08-29

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

