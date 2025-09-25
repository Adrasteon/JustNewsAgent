---
title: Balancer Agent V1 - Integration & Debugging Guide
description: Auto-generated description for Balancer Agent V1 - Integration & Debugging Guide
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# Balancer Agent V1 - Integration & Debugging Guide

## Overview
The Balancer Agent is a production-grade component of the JustNews V4 system, designed to neutralize, balance, and synthesize news articles using multi-agent collaboration. It leverages GPU-accelerated models and is fully integrated with the MCP bus for distributed, robust operation.

---

## Architecture & Workflow

### 1. MCP Bus Integration
- **Registration:** Balancer registers with the MCP bus via `/register` endpoint, providing metadata and endpoint listing.
- **Health Checks:** `/health` endpoint for MCP compliance; `/status` endpoint for agent health, MCP bus connectivity, and model status.
- **Inter-Agent Calls:** Uses `call_agent_tool()` to delegate tasks to Analyst, Fact Checker, Synthesizer, and other agents via MCP bus HTTP calls.

### 2. Endpoints & API
- `/register`: Registers agent with MCP bus.
- `/health`: Basic health check for MCP bus.
- `/status`: Reports agent health, MCP bus status, and model loading status.
- `/resource_status`: Reports CPU, memory, and GPU usage.
- `/balance_article`: Balances an article using alternative sources and quotes (rate-limited).
- `/extract_quotes`: Extracts quoted statements from articles (rate-limited).
- `/analyze_article`: Analyzes sentiment, bias, and fact-checking (rate-limited).
- `/chief_editor/balance_article`: Chief Editor workflow integration for balancing articles.

### 3. Model Stack
- **Sentiment:** RoBERTa (Analyst Agent)
- **Bias:** martin-ha/toxic-comment-model (Scout Agent)
- **Fact Checking:** DistilBERT, RoBERTa, BERT-large, SentenceTransformers, spaCy NER (Fact Checker Agent)
- **Summarization/Neutralization:** BART, T5 (Synthesizer Agent)
- **Embeddings:** SentenceTransformers
- **Quote Extraction:** Jean-Baptiste/roberta-large-ner-quotations

All model loading is wrapped in error handling and logs failures for debugging.

### 4. Production Robustness Features
- **Structured Logging:** Uses `structlog` for all operations, errors, and status events.
- **Validation:** Pydantic models for all requests/responses; custom FastAPI exception handlers for validation and HTTP errors.
- **Error Codes:** All endpoints return structured error codes/messages for known failure cases.
- **Rate Limiting:** `slowapi` integration for all public endpoints, with clear error responses.
- **Resource Monitoring:** `/resource_status` endpoint using `psutil` and `torch`.
- **Model Loading Error Handling:** All model initializations wrapped in try/except with logging and RuntimeError.
- **MCP Bus Health Check:** Periodic health check via `/status` endpoint, with latency reporting.

---

## Debugging & Usage

### 1. Starting the Agent
- Run with FastAPI/Uvicorn: `python balancer.py` or via MCP bus systemd script.
- Ensure all dependencies are installed: `structlog`, `slowapi`, `psutil`, `transformers`, `sentence-transformers`, `torch`, `bs4`, `requests`, `fastapi`, `uvicorn`, `pydantic`.

### 2. Endpoint Testing
- Use `/health` and `/status` to verify agent and MCP bus connectivity.
- Use `/resource_status` to monitor system resources.
- Test `/balance_article`, `/extract_quotes`, `/analyze_article` with valid payloads; check for rate limit errors and validation errors.
- For debugging, inspect logs (structlog) for operation, error, and status events.

### 3. Error Handling
- All model loading failures, request validation errors, and HTTP errors are logged and returned with structured error codes.
- If an endpoint fails, check logs for `model_load_error`, `request_validation_error`, or specific endpoint error codes.

### 4. MCP Bus & Multi-Agent Workflows
- Balancer delegates analysis, fact-checking, and synthesis to other agents via MCP bus HTTP calls.
- Chief Editor agent can orchestrate balancing via `/chief_editor/balance_article`.
- All inter-agent calls are logged; failures are handled with fallback to local models if possible.

### 5. Resource Monitoring
- `/resource_status` reports CPU, memory, and GPU usage for debugging performance and resource issues.

### 6. Extending & Debugging
- To add new models or endpoints, follow the established patterns: wrap all model loading in try/except, use Pydantic for validation, and log all operations/errors with structlog.
- For debugging, use the `/status` endpoint to check MCP bus and model health, and inspect logs for detailed error traces.

---

## Example Request Payloads

**Balance Article:**
```json
{
  "main_article": "...",
  "alt_articles": ["...", "..."]
}
```

**Extract Quotes:**
```json
{
  "article": "..."
}
```

**Analyze Article:**
```json
{
  "article": "..."
}
```

---

## Troubleshooting Checklist
- [ ] Agent starts without errors
- [ ] `/health` and `/status` endpoints return `status: ok`
- [ ] All models report `ok` in `/status` (otherwise see logs for error)
- [ ] Rate limits are enforced and errors returned as expected
- [ ] Resource usage is within expected bounds
- [ ] Inter-agent calls via MCP bus succeed (see logs for failures)
- [ ] All endpoints validate requests and return structured errors on failure

---

## References
- See `balancer.py` for implementation details
- See JustNews V4 architecture documentation in `markdown_docs/TECHNICAL_ARCHITECTURE.md`
- For agent conventions, see `markdown_docs/agent_documentation/`

---

**Maintainer:** Adrasteon / JustNews V4 Team
**Last Updated:** August 15, 2025

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

