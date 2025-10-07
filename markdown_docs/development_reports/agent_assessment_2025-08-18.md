---
title: Agent Assessment — 2025-08-18
description: Auto-generated description for Agent Assessment — 2025-08-18
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# Agent Assessment — 2025-08-18

This document summarizes an inspection of the `agents/` directory and how each agent maps to the JustNews V4 plan (docs/JustNews_Plan_V4.md).

Date: 2025-08-18

---

## Summary

I inspected representative `main.py` entrypoints for the following agents: `scout`, `analyst`, `fact_checker`, `synthesizer`, `chief_editor`, `critic`, `memory`, `newsreader`, `reasoning`, and `balancer`. Each agent is implemented as a FastAPI-compatible service that registers with an MCP Bus at startup (with graceful fallback if MCP Bus is unavailable). Agents expose tool endpoints (ToolCall-style inputs: `{args: [], kwargs: {}}`), health/readiness endpoints, and often provide GPU-accelerated endpoints (via `gpu_tools`) with CPU fallbacks.

This matches the high-level design in `docs/JustNews_Plan_V4.md` which specifies specialized agents, RTX/TensorRT optimization for performance, and a hybrid fallback architecture.

## Per-agent assessment (contract + notes)

### Scout
- Purpose: discovery & crawling, intelligent source discovery, production crawlers.
- Inputs: ToolCall (url(s), crawler parameters)
- Outputs: lists of discovered content, article payloads
- Key endpoints: `/discover_sources`, `/crawl_url`, `/deep_crawl_site`, `/enhanced_deep_crawl_site`, `/production_crawl_ultra_fast`, `/production_crawl_ai_enhanced`, `/get_production_crawler_info`, `/health`, `/ready`
- Notes: central place for discovery and batch operations. Delegates to `agents.scout.tools` implementations.

### Analyst
- Purpose: entity extraction, text statistics, numerical metric extraction, trend analysis. RTX/TensorRT-first inference strategy referenced in docs.
- Inputs: ToolCall (text or document lists)
- Outputs: entities, metrics, trend structures
- Key endpoints: `/identify_entities`, `/analyze_text_statistics`, `/extract_key_metrics`, `/analyze_content_trends`, `/log_feedback`, `/health`, `/ready`
- Notes: Plan references high performance (730+ art/sec) for Analyst with TensorRT.

### Fact Checker
- Purpose: fact verification, claim validation, GPU-accelerated checks.
- Inputs: ToolCall (content/claim)
- Outputs: validation scores, verification results
- Key endpoints: `/validate_is_news`, `/verify_claims`, `/validate_claims`, `/validate_is_news_gpu`, `/verify_claims_gpu`, `/performance/stats`, `/log_feedback`
- Notes: GPU endpoints gracefully fall back to CPU implementations.

### Synthesizer
- Purpose: cluster and synthesize articles, neutralize text, GPU-accelerated synthesis with CPU fallback.
- Inputs: ToolCall (articles or clusters)
- Outputs: synthesized articles, themes, performance metadata
- Key endpoints: `/cluster_articles`, `/aggregate_cluster`, `/neutralize_text`, `/synthesize_news_articles_gpu`, `/get_synthesizer_performance`, `/log_feedback`
- Notes: Plan mentions a 5-model synthesizer architecture (BERTopic, BART, T5, DialogGPT, SentenceTransformer).

### Chief Editor
- Purpose: coordinate editorial workflow: request briefs, publish, lifecycle management.
- Inputs: ToolCall (story brief params / content)
- Outputs: orchestration/status messages
- Key endpoints: `/request_story_brief`, `/publish_story`, `/coordinate_editorial_workflow`, `/manage_content_lifecycle`
- Notes: Orchestration-focused; small surface area.

### Critic
- Purpose: critique synthesized content, neutrality and logical quality assessment, GPU critique.
- Inputs: ToolCall (articles)
- Outputs: critiques, quality scores, bias indicators, performance stats
- Key endpoints: `/critique_synthesis`, `/critique_neutrality`, `/critique_content_gpu`, `/get_critic_performance`, `/log_feedback`
- Notes: CPU fallback present; Plan lists multi-model Critic architecture.

### Memory
- Purpose: persistent storage for articles and training examples, vector search via embeddings, DB-backed storage (Postgres)
- Inputs: JSON article payloads, VectorSearch queries
- Outputs: DB save results, article retrieval, vector search results
- Key endpoints: `/save_article`, `/store_article`, `/get_article/{id}`, `/vector_search_articles`, `/log_training_example`, `/health`, `/ready`
- Notes: uses `psycopg2`; expects DB env vars; will return HTTP 500 on DB connectivity failures.

### NewsReader
- Purpose: LLaVA-based webpage analysis and screenshot capture, image reasoning for news pages
- Inputs: URLs or image paths
- Outputs: extracted content, screenshot paths, LLaVA analysis
- Key endpoints: `/extract_news_content`, `/capture_screenshot`, `/analyze_screenshot`, `/analyze_image_content`, `/health`, `/ready`
- Notes: integrates `PracticalNewsReader` class and supports async processing.

### Reasoning (Nucleoid)
- Purpose: symbolic logic, facts/rules ingestion, contradiction detection, explainability for editorial workflows.
- Inputs: structured facts/rules or string queries
- Outputs: query results, contradiction detection, explanations
- Key endpoints: `/add_fact`, `/add_facts`, `/add_rule`, `/query`, `/evaluate`, `/validate_claim`, `/explain_reasoning`, `/facts`, `/rules`, `/status`, `/health`, `/ready`, `/call` (MCP)
- Notes: Implements fallback `SimpleNucleoidImplementation` if import/clone of full Nucleoid fails. CPU-only; Plan mentions <1GB CPU usage.

### Balancer
- Purpose: call routing/utility; exposes a `/call` proxy to `agents.balancer.tools` functions and a `/health` endpoint.
- Inputs: `name` (tool name) + ToolCall
- Outputs: {status, data} or errors
- Notes: lightweight router used in tests/integration and possibly for internal orchestration.

## Alignment with JustNews_Plan_V4.md
- The agents implement the same responsibilities and endpoints described in the Plan (Reasoning endpoints match exactly, Synthesizer/Critic/Facts reference GPU paths and 5-model architectures, Analyst references RTX/TensorRT optimizations). The code and docs are consistent in intent: specialized agents, MCP Bus registration, GPU-first with CPU fallback, and a training/feedback loop.

## Gaps and risks
- GPU dependency: absence of `gpu_tools` or missing runtime leads to fallbacks and performance loss. Need CI checks and a `gpu_health` indicator.
- DB availability: `memory` will raise HTTP 500 if DB unreachable. Add DB readiness check and retry/backoff.
- Repeated MCP registration code across agents: extract helper to `agents/common/` for consistent behavior and better testability.
- Tests: Plan mentions many benchmarks and CI tests; add lightweight unit tests and small integration mocks to validate registration and `/call` flows without requiring GPU/Docker.

## Recommendations & next steps
1. Add `agents/common/mcp_client.py` to centralize registration logic and error handling.
2. Add unit tests:
   - `tests/test_balancer.py` (mock tools) — verify `call_tool` behavior.
   - `tests/test_mcp_registration.py` — run agents' register logic against a fake MCP Bus.
   - `tests/test_memory_db_fallback.py` — mock DB to test error handling.
3. Add a small smoke integration test that launches a fake MCP Bus (FastAPI lightweight app) and an agent's `call` handler in-process.
4. Document requirements per-agent (models needed, GPU expectations, DB env vars) in `markdown_docs/agent_documentation/`.
5. Add a system-level health aggregator script that polls `/ready` endpoints and returns cluster readiness.

## Conclusion
Agents are implemented as FastAPI services with clear tool endpoints and match the roles described in Plan V4. The primary work remaining is integration testing, centralizing repeated logic (MCP client), and adding robust health/monitoring for GPU/DB dependencies.

---

Generated by repository inspection on 2025-08-18.

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

