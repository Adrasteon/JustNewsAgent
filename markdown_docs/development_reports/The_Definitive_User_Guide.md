<!--
	THE DEFINITIVE USER GUIDE: JustNews Agentic System (V4)
	This guide is a living document, integrating and expanding upon all major documentation, agent guides, production reports, and technical references in the workspace as of August 5, 2025.
-->

# The Definitive User Guide: JustNews Agentic System (V4)

---

## Table of Contents
1. [Introduction & System Overview](#introduction--system-overview)
2. [Architecture & Agent Roles](#architecture--agent-roles)
3. [Installation & Environment Setup](#installation--environment-setup)
4. [Service Management & Deployment](#service-management--deployment)
5. [Agent Functionality & Usage](#agent-functionality--usage)
6. [Data Flow & Pipeline](#data-flow--pipeline)
7. [API Endpoints & Tool Calls](#api-endpoints--tool-calls)
8. [Advanced Options & Customization](#advanced-options--customization)
9. [Troubleshooting & Best Practices](#troubleshooting--best-practices)
10. [Documentation Index & Further Reading](#documentation-index--further-reading)

---

## 1. Introduction & System Overview

JustNews Agentic V4 is a production-grade, multi-agent news analysis ecosystem designed for high-throughput, high-quality news discovery, analysis, and synthesis. It leverages GPU acceleration (TensorRT, LLaVA, LLaMA-3-8B) and a modular, agentic architecture for scalable, real-time news processing.

**Key Production Achievements:**
- **Production-Scale Crawling**: 8.14+ articles/sec (BBC, others)
- **Visual + Text Analysis**: LLaVA-1.5-7B, INT8 quantization
- **MCP Bus**: Central message bus for agent communication
- **Database**: PostgreSQL with vector search
- **GPU Stack**: RTX 3090, TensorRT, PyCUDA

**Recent Milestones:**
- **Cookie/modal handling solved** (BBC, Sky News, etc.)
- **Scout + NewsReader integration**: Visual and DOM-based content extraction
- **Memory optimization**: 6.4GB savings, 5.1GB buffer (see [Deployment Success](markdown_docs/production_status/DEPLOYMENT_SUCCESS_SUMMARY.md))
- **Full pipeline test passing**: 8/8 tests, end-to-end validation

---

## 2. Architecture & Agent Roles

### System Diagram

```
┌────────────┐   ┌────────────┐   ┌────────────┐
│  MCP Bus   │<->│   Agents   │<->│  Database  │
└────────────┘   └────────────┘   └────────────┘
```

**Agents** (each runs as a FastAPI service, typically on its own port):

| Agent         | Model/Tech                | Port  | Functionality                        |
|---------------|--------------------------|-------|--------------------------------------|
| Analyst       | RoBERTa/BERT TensorRT    | 8004  | Sentiment, bias, entity analysis     |
| Scout         | LLaMA-3-8B, Crawl4AI     | 8002  | News discovery, deep/production crawl|
| NewsReader    | LLaVA-1.5-7B (INT8)      | 8009  | Screenshot/image/DOM analysis        |
| Fact Checker  | DialoGPT (deprecated)-medium          | 8003  | Fact validation                     |
| Synthesizer   | DialoGPT (deprecated)-medium, Embeds  | 8005  | Clustering, synthesis               |
| Critic        | DialoGPT (deprecated)-medium          | 8006  | Quality assessment                  |
| Chief Editor  | DialoGPT (deprecated)-medium          | 8001  | Editorial orchestration             |
| Memory        | Vector DB, Embeddings    | 8007  | Semantic search, storage            |
| Reasoning     | Nucleoid, NetworkX       | 8008  | Symbolic logic, contradiction check |

**See also:** [Workspace Organization Summary](WORKSPACE_ORGANIZATION_SUMMARY.md)

---

## 3. Installation & Environment Setup

### Hardware/OS Requirements
- NVIDIA RTX 3090 (24GB VRAM recommended)
- Ubuntu 24.04 (native preferred)
- 32GB+ RAM, NVMe SSD

### Conda Environment
```bash
source /home/adra/miniconda3/etc/profile.d/conda.sh
conda activate rapids-25.06
```

### GPU Validation
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Database Setup
- PostgreSQL with user `justnews_user`
- Apply migrations in `agents/memory/db_migrations/`

---

## 4. Service Management & Deployment

### Start All Services
```bash
./start_services_daemon.sh
```
- Starts MCP Bus, Scout, Memory, Reasoning, and others as daemons

### Stop All Services
```bash
./stop_services.sh
```

### Check Service Status
```bash
ps aux | grep -E "(mcp_bus|scout|memory|reasoning)"
```

### Health Check
```bash
curl http://localhost:8000/agents
```

---

## 5. Agent Functionality & Usage

### Analyst Agent

**Purpose:** High-throughput sentiment, bias, and entity analysis using native TensorRT acceleration.

**Key Endpoints:**
- `/score_sentiment`, `/score_bias`, `/identify_entities`
- `/score_sentiment_batch`, `/score_bias_batch`
- `/analyze_article`, `/analyze_articles_batch`

**Performance:** 406.9 articles/sec (TensorRT, FP16)

**Standalone:**
```bash
python start_native_tensorrt_agent.py
```

**See also:** [NATIVE_AGENT_README.md](agents/analyst/NATIVE_AGENT_README.md)

### Scout Agent

**Purpose:** Content discovery, deep crawling, and production-scale news gathering.

**Deep Crawl:**
- Crawl4AI, BestFirstCrawlingStrategy, user-configurable parameters
- Quality filtering with LLaMA-3-8B (GPU-accelerated)

**Production Crawling:**
- Ultra-fast (8.14 art/sec), AI-enhanced (0.86 art/sec, NewsReader integration)
- Cookie/modal handling, multi-browser concurrency

**Key Tools:**
- `production_crawl_ultra_fast`, `get_production_crawler_info`, `enhanced_deep_crawl_site`

**Supported Sites:** BBC (production), CNN/Reuters/Guardian/NYT (expandable)

**See also:** [SCOUT_ENHANCED_DEEP_CRAWL_DOCUMENTATION.md](markdown_docs/agent_documentation/SCOUT_ENHANCED_DEEP_CRAWL_DOCUMENTATION.md)

### NewsReader Agent

**Purpose:** Visual and DOM-based content extraction using LLaVA-1.5-7B (INT8 quantized).

**Key Features:**
- Screenshot analysis, hybrid DOM + image extraction
- INT8 quantization for memory efficiency (6.8GB GPU)
- Zero model warnings, robust modal handling

**Key Endpoints:** `/analyze_screenshot`, `/analyze_dom`

**See also:** [agents/newsreader/README.md](agents/newsreader/README.md)

### Fact Checker, Synthesizer, Critic, Chief Editor

**Fact Checker:** Real-time claim validation (DialoGPT (deprecated)-medium)

**Synthesizer:** Clustering, aggregation, feedback loops (DialoGPT (deprecated)-medium + embeddings)

**Critic:** LLM-based critique, feedback logging (DialoGPT (deprecated)-medium)

**Chief Editor:** Editorial orchestration (DialoGPT (deprecated)-medium)

### Memory Agent

**Purpose:** PostgreSQL storage, semantic search, and vector retrieval.

**Key Features:**
- Articles, article_vectors, training_examples tables
- Hybrid endpoint handling (direct + MCP Bus)

### Reasoning Agent

**Purpose:** Symbolic logic, contradiction detection, and explainability (Nucleoid, NetworkX)

**Key Features:**
- AST parsing, variable assignments, dependency graphs
- Contradiction detection, graph-based logic

---

## 6. Data Flow & Pipeline

### End-to-End Pipeline

```
Scout → NewsReader → Analyst → Fact Checker → Synthesizer → Critic → Chief Editor → Memory → Reasoning
```

**Step-by-Step:**
1. **Scout** discovers/crawls news (deep/production)
2. **NewsReader** analyzes screenshots/DOM (visual + text)
3. **Analyst** scores sentiment/bias (TensorRT)
4. **Fact Checker** validates claims (DialoGPT (deprecated))
5. **Synthesizer** clusters/aggregates (embeddings)
6. **Critic** reviews quality (LLM-based)
7. **Chief Editor** orchestrates workflow
8. **Memory** stores articles/vectors (PostgreSQL)
9. **Reasoning** checks logic/contradictions (Nucleoid)

**See also:** [SCOUT_MEMORY_PIPELINE_SUCCESS.md](markdown_docs/agent_documentation/SCOUT_MEMORY_PIPELINE_SUCCESS.md)

---

## 7. API Endpoints & Tool Calls

### MCP Bus
- `/register` - Register agent/tools
- `/call` - Invoke tool on agent
- `/agents` - List registered agents

### Agent Endpoints (examples)
- `/score_sentiment`, `/score_bias`, `/analyze_article` (Analyst)
- `/production_crawl_ultra_fast`, `/get_production_crawler_info` (Scout)
- `/analyze_screenshot`, `/analyze_dom` (NewsReader)
- `/fact_check`, `/synthesize`, `/critique`, `/edit` (others)

### Usage Example
```python
import requests
response = requests.post("http://localhost:8002/production_crawl_ultra_fast", json={"args": ["bbc", 100], "kwargs": {}})
print(response.json())
```

---

## 8. Advanced Options & Customization

- **Agent Standalone Mode**: Run any agent with `uvicorn main:app --reload --port <PORT>`
- **Production Crawler Expansion**: Add new site crawlers in `agents/scout/production_crawlers/sites/`
- **Feedback Logging**: All agents log feedback for continual learning
- **Retraining**: Use feedback logs for online/scheduled retraining
- **GPU/CPU Fallback**: If GPU unavailable, agents fallback to CPU
- **Docker Support**: `docker-compose up --build` for containerized deployment

---

## 9. Troubleshooting & Best Practices

- **GPU Issues**: Check `nvidia-smi`, ensure drivers and CUDA toolkit are correct
- **Database Issues**: Ensure correct user/schema, apply all migrations
- **Model Loading**: Verify model files, check paths in config
- **Agent Registration**: MCP Bus must be running before agents for full integration
- **Logs**: Check agent-specific logs (e.g., `analyst_agent.log`, `feedback_scout.log`)
- **Workspace Cleanliness**: Use provided scripts to keep workspace organized

---

## 10. Documentation Index & Further Reading

- `README.md` - System overview and quick start
- `WORKSPACE_ORGANIZATION_SUMMARY.md` - File structure and organization
- `CHANGELOG.md` - Release notes and version history
- `docs/JustNews_Plan_V4.md` - Full architecture and planning
- `agents/<agent>/README.md` - Agent-specific guides (where available)
- `agents/newsreader/documentation/` - NewsReader technical docs
- `archive_obsolete_files/` - Development history and legacy files

---

*For the most up-to-date information, always refer to the root `README.md` and the organized documentation in `markdown_docs/`.*

---

**Status: August 5, 2025 - Production-Ready, Fully Documented, and Validated**
