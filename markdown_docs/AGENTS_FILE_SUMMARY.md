## Agents file summary — generated report

Generated: 2025-10-20

This document lists each top-level agent under `agents/` and summarizes the purpose of the scripts and important files found in each agent directory. Where a module-level docstring or comment exists it is used as the canonical short description; otherwise a concise inferred description is provided.

---

## analyst/

Files (selected):

- `Dockerfile`, `Dockerfile.simple`, `Dockerfile.v4` — container images for different deployment profiles.
- `NATIVE_AGENT_README.md`, `NATIVE_TENSORRT_README.md`, `TENSORRT_QUICKSTART.md` — operational docs and quickstarts for TensorRT/native engines.
- `main.py` — FastAPI entrypoint for the Analyst agent: registers with MCP bus, exposes endpoints for sentiment, bias, entity extraction, metrics and readiness/health. Includes security middleware and DB helpers.
- `tools.py` — Core Analyst business logic: entity extraction, text statistics, key metric extraction, trend analysis, heuristic and GPU-accelerated sentiment/bias analysis, and many helper utilities. Provides `analyze_sentiment`, `detect_bias`, `identify_entities`, `analyze_text_statistics`, `extract_key_metrics`, `analyze_content_trends`, and combined analysis functions.
- `tensorrt_tools.py` — Helpers to integrate with native TensorRT compiled engines: engine lifecycle, global engine instance, and fallback handling.
- `tensorrt_acceleration.py` — High-level TensorRT acceleration layer and a `TensorRTAnalyst` class. Handles engine discovery, building engine markers, and orchestration between native engine and HuggingFace fallback. Includes CLI entrypoints to build/test engines.
- `native_tensorrt_engine.py` — Native TensorRT engine implementation: CUDA initialization, engine loading/deserialization, tokenization hooks, safe GPU context management, inference (single and batch), and careful cleanup.
- `native_tensorrt_compiler.py` — (tooling) helper for compiling or stubbing engine builds (compiler-related helper script).
- `rtx_manager.py` — RTX / device orchestration helper used by hybrid/v4 tooling.
- `gpu_analyst.py` — thin GPU wrapper used by `tools.py` to run GPU-accelerated model scoring (sentiment/bias) and batch inference.
- `hybrid_tools_v4.py` — V4 hybrid architecture glue: Docker Model Runner + GPU-first fallbacks, hybrid manager, Docker client, and V4 performance-focused helpers. Implements GPU-first scoring and batching, feedback logs, and RTX integration points.
- `online_learning_trainer.py` — Online training integration / feedback pipeline hooks (deferred import friendly).
- `security_utils.py` — Security helpers used by endpoints (sanitization, validation, logging).
- `tensorrt_engines/` — directory holding compiled engine artifacts and JSON metadata (`native_sentiment_roberta.json`, `native_bias_bert.json` etc.).

Notes:

- Analyst is the most complex agent: it contains GPU-accelerated paths, TensorRT native engine integration, fallback HuggingFace pipelines, and heuristic fallbacks. The modules are written defensively to allow operation with or without heavy dependencies.

---

## analytics/

Files:

- `__init__.py` — Analytics integration helpers and convenience functions.
- `main.py` — Main FastAPI entrypoint for the Analytics Agent. Registers with MCP Bus and exposes endpoints for system health, performance metrics, agent profiles, optimization recommendations, and metric ingestion. Mounts the analytics dashboard under `/dashboard`.
- `dashboard.py` — Advanced analytics dashboard: a FastAPI app that provides HTML templates and API endpoints for realtime analytics, agent profiles, trends, and reports. Uses the shared `advanced_analytics` engine to provide data and recommendations.
- `analytics/` (templates) — Jinja2 templates and static assets used by the dashboard.

---

## archive/

Files (selected):

- `archive_manager.py` — Manager that coordinates archive storage, metadata indexing and Knowledge Graph integration (Phase 3). Key classes and functions from the module:
	- `ArchiveStorageManager` — multi-backend storage manager (local filesystem by default; S3 support scaffolded but boto3 commented out). Methods: `store_article`, `_store_local`, `_store_s3` (placeholder), `retrieve_article`, `archive_batch` (concurrent batch storage with performance metrics).
	- `ArchiveMetadataIndex` — placeholder metadata indexer for archived articles: `index_article`, `search_articles` (TODO: uses SQLite by default in the placeholder config).
	- `ArchiveManager` — high-level orchestration: composes storage manager, metadata index and `KnowledgeGraphManager` (KG integration). Method: `archive_from_crawler` that archives crawler results, indexes metadata and sends artifacts through the Knowledge Graph processor.
	- `demo_phase3_archive()` — a runnable demo that simulates crawler results, archives them and runs KG processing; useful as an integration example and smoke test.

- `main.py` — FastAPI agent entrypoint for the Archive agent. Important behaviors and endpoints:
	- Lifespan startup registers the agent with the MCP Bus (gracefully degrades to standalone mode if MCP is unreachable).
	- Exposed endpoints include `/archive_articles` (accepts crawler-like batch payloads and runs `ArchiveManager.archive_from_crawler`), `/retrieve_article` (retrieve archived JSON by storage key), `/search_archive` (search metadata index), `/get_archive_stats` (filesystem + knowledge-graph stats), and `/store_single_article` (store a single article payload).
	- Integrates with `JustNewsMetrics` for Prometheus-style metrics and registers optional shared `shutdown` and `reload` endpoints from `agents.common`.

Notes and developer hints:

- Archive code is written for Phase 3 research-scale archiving: provenance metadata, knowledge graph integration and batch performance metrics are first-class. S3 support is scaffolded but currently falls back to local storage when boto3 isn't available.
- The KG integration is implemented via `KnowledgeGraphManager` (imported from `knowledge_graph.py`) and is invoked after batch storage to produce `kg_summary` results.
- The `archive_batch` method uses asyncio.gather for concurrent storage and returns a detailed summary including rate and success counts — useful when benchmarking crawler → archive throughput.

## auth/
- `tools.py` — utility functions used by the chief editor flow.

## Agents file summary — generated report

Generated: 2025-10-20

This document lists each top-level agent under `agents/` and summarizes the purpose of the scripts and important files found in each agent directory. Where a module-level docstring or comment exists it is used as the canonical short description; otherwise a concise inferred description is provided.

---

## analyst/

Files (selected):

- `Dockerfile`, `Dockerfile.simple`, `Dockerfile.v4` — container images for different deployment profiles.
- `NATIVE_AGENT_README.md`, `NATIVE_TENSORRT_README.md`, `TENSORRT_QUICKSTART.md` — operational docs and quickstarts for TensorRT/native engines.
- `main.py` — FastAPI entrypoint for the Analyst agent: registers with MCP bus, exposes endpoints for sentiment, bias, entity extraction, metrics and readiness/health. Includes security middleware and DB helpers.
- `tools.py` — Core Analyst business logic: entity extraction, text statistics, key metric extraction, trend analysis, heuristic and GPU-accelerated sentiment/bias analysis, and many helper utilities.

Notes:

- Analyst is the most complex agent and contains GPU-accelerated paths, TensorRT native engine integration, and fallback HuggingFace pipelines.

---

## analytics/

Files:

- `main.py`, `dashboard.py`, `__init__.py` — analytics FastAPI app and dashboard templates.

---

## archive/

Files (selected):

- `archive_manager.py` — storage manager, metadata indexer and Knowledge Graph integration. Key classes: `ArchiveStorageManager`, `ArchiveMetadataIndex`, `ArchiveManager`. Includes `demo_phase3_archive()` demo helper.
- `main.py` — FastAPI entrypoint exposing `/archive_articles`, `/retrieve_article`, `/search_archive`, `/get_archive_stats`, and `/store_single_article`.

Notes:

- Archive focuses on provenance, batch archiving and KG integration. S3 support is scaffolded but local filesystem is the default fallback.

---

## balancer/

Files (selected):

- `main.py` — FastAPI entrypoint exposing `distribute_load`, `get_agent_status`, `balance_workload`, `monitor_performance`, plus health/metrics endpoints.
- `balancer.py` — core orchestration logic with lazy model loaders and multi-agent delegation helpers.
- `tools.py` — small utility toolkit (echo, sum_numbers, distribute_load, get_agent_status).

---

## auth/

Files:

- `__init__.py` and small helpers for authentication models and initialization.

---

## chief_editor/

Files:

- `main.py`, `handler.py`, `tools.py`, `chief_editor_v2_engine.py`, `evidence_review_queue.jsonl`.

---

## common/

This directory holds shared infra: DB helpers, model loader/cache, GPU management and observability utilities.

---

## crawler/

Files/dirs:

- `main.py`, `crawler_utils.py`, `orchestrator.py`, `unified_production_crawler.py`, `sites/`.

---

## crawler_control/

Files:

- `main.py` and a small web interface for crawler fleet control.

---

## critic/

Files:

- `main.py`, `critic_v2_engine.py`, `tools.py`, `gpu_tools.py`.

---

## dashboard/

Files:

- `main.py`, `config.py`, `gui.py`, `public_api.py`, `storage.py`, `templates/`, `static/`.

---

## fact_checker/

Files:

- `main.py`, `fact_checker_v2_engine.py`, `tools.py`, `tools_v2.py`, `gpu_tools.py`.

---

## gpu_orchestrator/

Files:

- `main.py`, `preload.py`, `nvml.py`, `utils.py` — GPU leasing and NVML wrappers.

---

## mcp_bus/

Files:

- `main.py` — MCP Bus central router (register/call/agents endpoints).

---

## memory/

Files:

- `main.py`, `memory_v2_engine.py`, `tools.py`, `db_migrations/`.

---

## newsreader/

Files:

- `main.py`, `newsreader_agent.py`, `newsreader_v2_engine.py`, `llava_newsreader_agent.py`, `documentation/`.

---

## reasoning/

Files:

- `main.py`, `enhanced_reasoning_architecture.py`, `nucleoid_implementation.py`, `local_nucleoid/`.

---

## scout/

Files:

- `main.py`, `gpu_scout_engine.py`, `production_crawlers/`, `tools.py`.

---

## synthesizer/

Files:

- `main.py`, `tools.py`, `synthesizer_v2_engine.py`, `synthesizer_v3_production_engine.py`.

---

## How this report was generated

- I enumerated the `agents/` directory and read module docstrings and top comments where present (notably `agents/analyst/*`). For other agents I used file names and repository conventions to provide concise inferred summaries.

## Next steps / improvements

- Extract and insert first module docstrings from each Python file for literal per-file summaries.
- Produce a machine-readable inventory (JSON/CSV) if needed.

---

Completion: initial full-agent inventory with in-depth Analyst summaries. Tell me which agent to expand next (or I can continue automatically).
- `dashboard.py` — Advanced analytics dashboard: a FastAPI app that provides HTML templates and API endpoints for realtime analytics, agent profiles, trends, and reports. Uses the shared `advanced_analytics` engine to provide data and recommendations.
- `analytics/` (templates) — Jinja2 templates and static assets used by the dashboard.

---

## archive/

Files (selected):

- `archive_manager.py` — Manager that coordinates archive storage, metadata indexing and Knowledge Graph integration (Phase 3). Key classes and functions from the module:
	- `ArchiveStorageManager` — multi-backend storage manager (local filesystem by default; S3 support scaffolded but boto3 commented out). Methods: `store_article`, `_store_local`, `_store_s3` (placeholder), `retrieve_article`, `archive_batch` (concurrent batch storage with performance metrics).
	- `ArchiveMetadataIndex` — placeholder metadata indexer for archived articles: `index_article`, `search_articles` (TODO: uses SQLite by default in the placeholder config).
	- `ArchiveManager` — high-level orchestration: composes storage manager, metadata index and `KnowledgeGraphManager` (KG integration). Method: `archive_from_crawler` that archives crawler results, indexes metadata and sends artifacts through the Knowledge Graph processor.
	- `demo_phase3_archive()` — a runnable demo that simulates crawler results, archives them and runs KG processing; useful as an integration example and smoke test.

- `main.py` — FastAPI agent entrypoint for the Archive agent. Important behaviors and endpoints:
	- Lifespan startup registers the agent with the MCP Bus (gracefully degrades to standalone mode if MCP is unreachable).
	- Exposed endpoints include `/archive_articles` (accepts crawler-like batch payloads and runs `ArchiveManager.archive_from_crawler`), `/retrieve_article` (retrieve archived JSON by storage key), `/search_archive` (search metadata index), `/get_archive_stats` (filesystem + knowledge-graph stats), and `/store_single_article` (store a single article payload).
	- Integrates with `JustNewsMetrics` for Prometheus-style metrics and registers optional shared `shutdown` and `reload` endpoints from `agents.common`.

Notes and developer hints:

- Archive code is written for Phase 3 research-scale archiving: provenance metadata, knowledge graph integration and batch performance metrics are first-class. S3 support is scaffolded but currently falls back to local storage when boto3 isn't available.
- The KG integration is implemented via `KnowledgeGraphManager` (imported from `knowledge_graph.py`) and is invoked after batch storage to produce `kg_summary` results.
- The `archive_batch` method uses asyncio.gather for concurrent storage and returns a detailed summary including rate and success counts — useful when benchmarking crawler → archive throughput.

---

## balancer/

Files (selected):

- `main.py` — FastAPI entrypoint for the Balancer agent. Key behaviors:
	- Lifespan and MCP Bus registration pattern (registers tools: `distribute_load`, `get_agent_status`, `balance_workload`, `monitor_performance`).
	- Exposes endpoints: `/distribute_load`, `/get_agent_status`, `/balance_workload`, `/monitor_performance`, plus `/health`, `/ready`, `/status`, `/resource_status`, and a `/metrics`-style integration via `JustNewsMetrics`.
	- Integrates optional `shutdown` and `reload` endpoints from `agents.common` and uses a Prometheus-style metrics middleware.

- `balancer.py` — Core balancing logic and helpers (local module-level tools and workflows). Contains a V1 prototype docstring describing model integrations, lazy model loaders for sentiment/bias/fact-check/summarization/quote extraction, and multi-agent delegation helpers (`call_agent_tool`, `analyze_article`, `extract_quotes`, `summarize_article`, `neutralize_text`, etc.). Also includes resource monitoring and MCP Bus health checks.

- `tools.py` — Small utility toolkit used by `main.py` with functions: `echo`, `sum_numbers`, `distribute_load` (round-robin distribution), `get_agent_status` (simulated/status aggregator), `balance_workload` (recommendations), and `monitor_performance` (performance metrics generator).

Notes and developer hints:

- The Balancer agent orchestrates cross-agent workload distribution and includes both lightweight utilities (`tools.py`) and heavier orchestration glue (`balancer.py`) that defers heavy model loads via lazy getters to avoid costly startup.
- Many functions in `balancer.py` are written defensively with fallbacks so the agent can operate in constrained test environments (e.g., missing HF models or unavailable MCP Bus).

---

## auth/

Files:

- `__init__.py` and related small helpers — authentication models and initializers.

---

## chief_editor/

Files:

- `main.py` — agent entrypoint for chief editor responsibilities.
- `handler.py` — request handling / orchestration logic.
- `tools.py` — utility functions used by the chief editor flow.
- `chief_editor_v2_engine.py` — engine implementation used by chief editor v2.
- `evidence_review_queue.jsonl` — evidence queue used by editor tooling.

---

## common/

This directory contains shared libraries used throughout agents. Key files include:

- `database.py`, `async_database.py` — DB connection & helpers used by many agents.
- `model_loader.py`, `model_store.py` — helpers to load and cache models.
- `gpu_*` (gpu_manager.py, gpu_config_manager.py, gpu_metrics.py, gpu_orchestrator_client.py, gpu_monitoring_enhanced.py, gpu_optimizer_enhanced.py) — GPU management utilities and clients used by multiple agents.
- `observability`, `tracing.py`, `notifications.py`, `schemas.py`, `shutdown.py`, `reload.py` — cross-cutting infra utilities.

---

## crawler/

Files/dirs:

- `main.py` — crawler agent entrypoint.
- `crawler_utils.py`, `orchestrator.py`, `performance_monitoring.py` — crawler helpers.
- `unified_production_crawler.py` — production-grade crawler implementation.
- `sites/` — site-specific crawlers such as `bbc_crawler.py`, `bbc_ai_crawler.py`, `generic_site_crawler.py`.

---

## crawler_control/

Files:

- `main.py` — orchestration / control entrypoint for crawler fleet.
- `web_interface/` — simple web UI to control or preview crawler state.

---

## critic/

Files:

- `main.py` — critic agent entrypoint.
- `critic_v2_engine.py` — critic logic/engine.
- `tools.py`, `gpu_tools.py` — helper functions.

---

## dashboard/

Files:

- `main.py`, `config.py`, `gui.py`, `public_api.py`, `storage.py`, `tools.py` — dashboard server, web UI and storage helpers.
- `templates/`, `web_interface/`, `static/` — front-end assets.

---

## fact_checker/

Files:

- `main.py` — fact-checker agent entrypoint.
- `fact_checker_v2_engine.py` — engine implementation for fact checking.
- `tools.py`, `tools_v2.py`, `gpu_tools.py` — helpers and GPU helpers used by fact-checker.

---

## gpu_orchestrator/

Files:

- `main.py`, `preload.py`, `nvml.py`, `utils.py` — orchestrator for GPU leasing/monitoring and NVML wrappers. This agent is used to coordinate GPU allocation across agents.

---

## mcp_bus/

Files:

- `main.py` — MCP Bus central router (register/call/agents health endpoints) — central piece of agent-to-agent comms.

---

## memory/

Files:

- `main.py` — memory agent entrypoint (vector DB, training examples).
- `memory_v2_engine.py` — V2 memory engine.
- `tools.py`, `db_migrations/` — migrations and helpers.

---

## newsreader/

Files:

- `main.py`, `newsreader_agent.py`, `newsreader_v2_engine.py`, `newsreader_v2_true_engine.py` — newsreader agent entrypoints and engines.
- `llava_newsreader_agent.py` — another reader integration.
- `documentation/` — agent-specific docs (INT8_QUANTIZATION_RATIONALE.md, IMPLEMENTATION_SUMMARY.md, LIFESPAN_MIGRATION.md).

---

## reasoning/

Files:

- `main.py`, `enhanced_reasoning_architecture.py`, `nucleoid_implementation.py` — reasoning agent, nucleoid local implementation and language handlers in `local_nucleoid/`.

---

## scout/

Files:

- `main.py` — scout agent entrypoint.
- `gpu_scout_engine.py`, `gpu_scout_engine_v2.py` — GPU-enabled scout engines.
- `production_crawlers/` — production crawler adapters used by scout including `sites/` with `bbc_crawler.py` variants.
- `tools.py`, `regenerate_hashes.py`, `practical_newsreader_solution.py` — utilities and scripts aiding scouting/crawling.

---

## synthesizer/

Files:

- `main.py` — synthesizer agent entrypoint.
- `tools.py`, `synthesizer_v2_engine.py`, `synthesizer_v3_production_engine.py` — synthesis engines and helpers.
- `gpu_tools.py` — GPU utilities used by the synthesizer.

---

## How this report was generated

- I enumerated the `agents/` directory and read module docstrings and top comments where present (notably `agents/analyst/*`). For other agents I used file names and repository conventions to provide concise inferred summaries. This document is intended as a readable inventory and quick developer reference.

## Next steps / improvements

- Expand per-file summaries by extracting the first module docstring from each Python file (I can iterate and insert those verbatim if you want a fully literal docstring report).
- Add cross-links to other markdown docs (agent-level READMEs are present for many agents).
- Optionally produce a JSON or CSV inventory for automated tooling.

---

Completion: initial full-agent inventory + in-depth Analyst agent summaries. Ask me to expand any agent's section into literal docstrings for every file and I'll iterate and update this file.
