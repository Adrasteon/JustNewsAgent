---
title: Entrypoints and Orchestration Flows — 2025-08-18
description: Auto-generated description for Entrypoints and Orchestration Flows — 2025-08-18
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# Entrypoints and Orchestration Flows — 2025-08-18

This document lists entry points into the JustNewsAgentic system that accept a URL or "news topic as text" and maps minimal orchestration flows to gather, analyze, assess, and synthesize news content.

Date: 2025-08-18

---

## Entrypoints (by intent)

These are the endpoints (agent + route) found in the repository that accept either a URL or a text/topic and can be used to start a news gathering / analysis / synthesis pipeline.

### A. Direct URL → content extraction
- `agents/newsreader`  
  - `POST /extract_news_content` — primary entry to extract article content from a URL (async). Accepts URL and optional screenshot path.
  - `POST /capture_screenshot` — capture page screenshot (URL + output path).
  - `POST /analyze_screenshot` / `POST /analyze_image_content` — LLaVA image analysis of screenshots.
- `agents/scout`  
  - `POST /crawl_url` — crawl a single URL (returns crawl results / discovered content).
  - `POST /enhanced_newsreader_crawl` — specialized crawl that integrates with newsreader; may accept URL(s).
- `agents/balancer`  
  - `POST /call?name=<tool>` — generic router that can call a named tool (useful for centralized entry/relay).

### B. Topic-text / query → source discovery and batch gather
- `agents/scout`  
  - `POST /discover_sources` — discover source URLs given topic/query parameters.
  - `POST /intelligent_source_discovery` — topic-driven discovery (uses AI to find sources).
  - `POST /intelligent_content_crawl` / `POST /intelligent_batch_analysis` — fetch content for a topic in batches.
  - `POST /production_crawl_ai_enhanced` or `POST /production_crawl_ultra_fast` — production crawl endpoints (batch/topic-enabled).
- `agents/memory`  
  - `POST /vector_search_articles` — query stored articles by text (topic) to get related material (useful to seed synthesis).
- `agents/balancer`  
  - `POST /call?name=<tool>` — can be used to route a Topic->discover->crawl sequence via configured tool names.

### C. Analysis / assessment / synthesis endpoints (consumers of extracted content)
- `agents/analyst`  
  - `POST /identify_entities` — entity extraction on text.  
  - `POST /analyze_text_statistics` — text statistics / readability.  
  - `POST /analyze_content_trends` — cross-article trend analysis.
- `agents/fact_checker`  
  - `POST /validate_claims` | `POST /verify_claims` | `POST /validate_is_news` — claim/source validation and fact checking.  
  - `POST /verify_claims_gpu` / `POST /validate_is_news_gpu` — GPU-accelerated variants (fallbacks exist).
- `agents/critic`  
  - `POST /critique_synthesis` / `POST /critique_neutrality` / `POST /critique_content_gpu` — quality/bias/neutrality critique (GPU+fallback).
- `agents/synthesizer`  
  - `POST /cluster_articles` / `POST /aggregate_cluster` — cluster/aggregate.  
  - `POST /synthesize_news_articles_gpu` — GPU-accelerated synthesis (with CPU fallback).  
  - `POST /get_synthesizer_performance` — performance info.

### D. Reasoning / Explainability
- `agents/reasoning`  
  - `POST /add_fact`, `POST /add_facts`, `POST /add_rule`, `POST /query`, `POST /evaluate`, `POST /validate_claim`, `POST /explain_reasoning` — ingest facts/rules and provide symbolic checks/explanations (useful for editorial transparency).

### E. Orchestration / utility
- Every agent exposes `/health` and `/ready` to gate orchestration.
- MCP bus integration: many agents register to an MCP Bus (`MCP_BUS_URL`) and expose `POST /call` handler (or accept tool calls) so another component (MCP Bus or balancer) can call them by tool name.

---

## Two practical orchestration flows

Below are minimal, practical sequences you can run (each arrow indicates “send output of” → “call next endpoint with”).

### Flow 1 — URL-driven (single URL)
1. Input: user provides a URL (single article)
2. Extract:
   - Call `newsreader` `POST /extract_news_content` with the URL.
   - Output: article text, metadata, images, optional screenshot path.
3. Store / related retrieval (optional):
   - Call `memory` `POST /save_article` or `POST /store_article` to persist the article.
   - Optionally call `memory` `POST /vector_search_articles` with article text to find related items.
4. Analyze:
   - Call `analyst` `POST /identify_entities` and `POST /analyze_text_statistics`.
   - Call `fact_checker` `POST /validate_claims` for claims extracted or `POST /validate_is_news`.
5. Critique:
   - Call `critic` `POST /critique_synthesis` or `POST /critique_neutrality`.
6. Synthesize (if you want a synthesized story / summary):
   - If you have a set (single or multiple articles), call `synthesizer` `POST /cluster_articles` then `POST /aggregate_cluster` and finally `POST /synthesize_news_articles_gpu` to produce an aggregated synthesis.
7. Reasoning & Explanation:
   - Send any claims to `reasoning` `POST /validate_claim` and `POST /explain_reasoning` for symbolic validation and audit trail.

### Flow 2 — Topic-driven (text/topic seed)
1. Input: user provides topic text (e.g., "UK inflation and energy subsidies")
2. Discover sources:
   - Call `scout` `POST /intelligent_source_discovery` or `POST /discover_sources` with the topic text.
   - Output: candidate source URLs.
3. Batch crawl / fetch:
   - Call `scout` `POST /intelligent_content_crawl` or `POST /production_crawl_ai_enhanced` with the list of discovered URLs or a query parameter for the topic.
   - Or directly call `newsreader` `POST /extract_news_content` for each discovered URL (async).
4. (Optional) Enrich from memory:
   - Call `memory` `POST /vector_search_articles` with topic text to pull prior articles matching the topic.
5. Aggregate & Analyze:
   - Perform `analyst` and `fact_checker` calls (as in Flow 1) over the gathered set.
6. Cluster & Synthesize:
   - Use `synthesizer` `/cluster_articles` + `/aggregate_cluster` + `/synthesize_news_articles_gpu` to create a synthesized report for the topic.
7. Finalize:
   - Chief Editor (`/request_story_brief`, `/publish_story`) can be called to create editorial artifacts or trigger human-in-the-loop steps.
8. Record facts:
   - Push important claims to `reasoning` `POST /add_fact` and store outputs in `memory`.

---

## Minimal example ToolCall payloads

ToolCall shape used across agents is typically:
```json
{
  "args": [...],
  "kwargs": { ... }
}
```

### A. URL → extract (newsreader)
POST to `http://<newsreader-host>:8009/extract_news_content`
```json
{
  "args": ["https://example.com/news/some-article"],
  "kwargs": {"screenshot_path": "out/some-article.png"}
}
```

### B. URL → scout crawl
POST to `http://<scout-host>:8002/crawl_url`
```json
{
  "args": ["https://example.com/news/some-article"],
  "kwargs": {"depth": 1, "follow_links": false}
}
```

### C. Topic → discover sources (scout)
POST to `http://<scout-host>:8002/intelligent_source_discovery`
```json
{
  "args": [],
  "kwargs": {"topic": "UK inflation energy subsidies 2025", "max_sources": 20}
}
```

### D. Topic → production crawl (batch)
POST to `http://<scout-host>:8002/production_crawl_ai_enhanced`
```json
{
  "args": [],
  "kwargs": {"query": "UK inflation energy subsidies 2025", "limit": 50}
}
```

### E. Synthesis (synthesizer) — GPU synthesis accepting list of article dicts
POST to `http://<synthesizer-host>:8005/synthesize_news_articles_gpu`
```json
{
  "args": [
    [
      {"title": "Article A", "content": "text A", "url":"..."},
      {"title": "Article B", "content": "text B", "url":"..."}
    ]
  ],
  "kwargs": {"target_style": "neutral_summary", "max_articles": 10}
}
```

### F. Generic router (balancer) — call a named tool
POST to `http://<balancer-host>:<port>/call?name=identify_entities`
Body:
```json
{
  "args": [["This is the article text to analyze"]],
  "kwargs": {}
}
```

### G. Fact-check claim (fact_checker)
POST to `http://<fact-checker-host>:8003/validate_claims`
```json
{
  "args": [{"content": "The unemployment rate fell in June 2025 from 5% to 4%."}],
  "kwargs": {}
}
```

### H. Save article to memory
POST to `http://<memory-host>:8007/save_article`
```json
{
  "args": [],
  "kwargs": {"content": "full article text", "metadata": {"url":"...", "source":"example.com"}}
}
```

---

## Practical operational notes & tips
- Use `/health` and `/ready` on each agent before orchestration. Many agents set `ready` only after startup tasks; orchestration should gate on `ready`.
- Prefer the MCP Bus or `agents/balancer` as a single entry point if you plan centralized orchestration; it simplifies discovery and retries.
- GPU endpoints exist (synthesizer/fact_checker/critic) and provide high throughput; they have CPU fallbacks — ensure you check `get_*_performance` endpoints if performance matters.
- For topic-driven pipelines, combining `memory` vector search with `scout` discovery is powerful: memory returns related historical articles while scout discovers fresh sources.
- For reliable production runs, add retries for external fetches, concurrency limits for batch synthesis, and implement rate limiting for crawlers.
- Data shape consistency: many endpoints accept either MCP-style {args, kwargs} or direct dicts. Test both to ensure the agent accepts the payload you plan to send.

---

## Next steps you can request
- Implement a simple orchestrator script (Python) that accepts URL or topic and runs a full pipeline (newsreader → memory → analyst → fact_checker → synthesizer).
- Create a small "fake MCP Bus" FastAPI app and a smoke test that registers and executes a pipeline against `scout` and `newsreader`.
- Add minimal example client functions (Python) to call the example payloads above and print formatted results.

Which would you like me to implement next?

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

