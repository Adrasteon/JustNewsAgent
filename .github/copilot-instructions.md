---
description: Auto-generated description for JustNews V4 - GitHub Copilot Instructions
applyTo: "**/*"
---

# JustNews V4 - GitHub Copilot Instructions

## Copilot Quickstart — 1-screen guide for AI coding agents

- Big picture: this repo is a distributed multi-agent system. The central
  router is the MCP Bus (`agents/mcp_bus/main.py`, default port `8000`).
  Specialized agents live under `agents/<name>/` and register themselves
  with the MCP Bus on startup.

- Key patterns to emulate:
  - Registration and health: each agent implements an `MCPBusClient`
    and calls `register_agent(...)`. See `agents/synthesizer/main.py` and
    `agents/memory/main.py` for canonical examples.
  - MCP payloads use the JSON shape `{agent, tool, args, kwargs}` and
    most calls flow through `MCP_BUS_URL` (default `http://localhost:8000`).
  - Agent endpoints are FastAPI apps using typed Pydantic models and a
    `ToolCall`-style input pattern. Follow `agents/*/main.py` as a template.

- Startup & deployment:
  - Development: run agents directly (e.g. `python -m agents.mcp_bus.main`)
    or use `uvicorn agents.dashboard.main:app --port 8013` for the dashboard.
  - Production: use `deploy/systemd/` (`justnews@.service`) and the
    startup ordering / orchestration logic in `start_services_daemon.sh`.

- GPU & model conventions:
  - GPU config is centralized: `agents/common/gpu_config_manager.py` +
    `config/gpu/*` profiles. Native TensorRT code is in
    `agents/analyst/native_tensorrt_engine.py` — be conservative: add
    explicit cleanup and CPU fallbacks when modifying that code.

- Tests & e2e: many tests are MCP-dependent. Use `pytest -q` and run
  `scripts/mini_e2e_runner.py` or `comprehensive_100_article_test_fixed.py` with
  `MCP_BUS_URL` set to a local MCP instance. Add mocks when possible.
  
  Testing helpers: a lightweight in-process MCP stub is provided at
  `tests/mcp_scaffold.py`. Use the pytest fixtures `mcp_server` and
  `agent_server` (in `tests/conftest.py`) to simulate agent registration,
  custom handlers, or to forward calls to a small local `AgentServer`.
  - Set `MCP_BUS_URL` via `monkeypatch.setenv('MCP_BUS_URL', mcp_server.url())`
    before importing modules that read the env at import-time.
  - Use `mcp_server.add_handler(agent, tool, fn)` to program custom
    responses, or `mcp_server.forward_calls = True` to forward to a real
    HTTP-based agent (useful for integration-style tests).

- Conventions checklist (must-follow): PEP8 (88 chars), type hints,
  Google-style docstrings, structured JSON logging, and `JustNewsMetrics`
  usage for new metrics. Pre-commit hooks live in `.githooks/`.

- Quick search terms for implementation tasks: `MCP_BUS_URL`, `/call`,
  `register_agent`, `JustNewsMetrics`, `gpu_config_manager`.

Please review any gaps you want me to expand (examples for a new agent, a
sample PR checklist, or additional code patterns to follow).

## Project Overview

JustNews V4 is a production-ready multi-agent news analysis system featuring GPU-accelerated processing, continuous learning, and distributed architecture. The system processes news content through specialized AI agents that communicate via MCP (Model Context Protocol) for collaborative analysis, fact-checking, and synthesis.

**Current Status** (September 24, 2025): Production deployment with **Synthesizer V3** complete, documentation organization restructured, and **V2 engines completion** phase initiated.

## Development Standards

### Code Quality Requirements
- **Python Style**: Follow PEP 8 with 88-character line limit
- **Type Hints**: Required for all function signatures and class attributes
- **Docstrings**: Google-style docstrings for all public functions and classes
- **Error Handling**: Comprehensive exception handling with specific error types
- **Logging**: Structured logging with appropriate levels (DEBUG, INFO, WARNING, ERROR)

### Code Organization Patterns
```python
# Standard agent structure pattern
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ToolCall(BaseModel):
    """Standard MCP tool call format"""
    args: list
    kwargs: dict

@app.post("/tool_name")
def tool_endpoint(call: ToolCall) -> Dict[str, Any]:
    """Tool endpoint with proper error handling"""
    try:
        from tools import tool_function
        result = tool_function(*call.args, **call.kwargs)
        logger.info(f"Tool {tool_name} executed successfully")
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Tool {tool_name} failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Performance Guidelines
- **GPU Memory Management**: Use context managers and proper cleanup
- **Batch Processing**: Implement 16-32 item batches for optimal GPU utilization
- **Memory Monitoring**: Include memory usage logging in GPU operations
- **Fallback Systems**: Always provide CPU fallback for GPU operations

## Developer Workflows

### Environment Setup
- Use `activate_environment.sh` to activate the development environment.
- Install dependencies using `pip install -r requirements.txt`.

### Running Tests
- Use `pytest` for running tests:
  ```bash
  pytest -q
  ```
- For GPU-specific tests, ensure the GPU is available and properly configured.

### Linting
- Use `ruff` for linting:
  ```bash
  ruff check .
  ```

### Debugging
- Use the `fastapi_test_shim.py` script for debugging FastAPI endpoints.
- Enable detailed logging by setting the environment variable `LOG_LEVEL=DEBUG`.

## Architecture Guidelines

### Agent Communication Protocol
**Critical**: All agents must follow the MCP (Model Context Protocol) pattern:

```python
# MCP Bus Integration Pattern
def call_agent_tool(agent: str, tool: str, *args, **kwargs) -> Any:
    """Standard pattern for inter-agent communication"""
    payload = {
        "agent": agent,
        "tool": tool,
        "args": list(args),
        "kwargs": kwargs
    }
    response = requests.post(f"{MCP_BUS_URL}/call", json=payload)
    response.raise_for_status()
    return response.json()
```

### Core Components Architecture

#### MCP Bus (Port 8000)
- Central communication hub using FastAPI
- Agent registration with `/register`, `/call`, `/agents` endpoints
- Health check and service discovery

#### Agents (Ports 8001-8008)
- **Scout** (8002): Content discovery with 5-model AI architecture
- **Analyst** (8004): TensorRT-accelerated sentiment/bias analysis
- **Fact Checker** (8003): 5-model verification system
- **Synthesizer** (8005): **V3 Production** - 4-model synthesis (BERTopic, BART, FLAN-T5, SentenceTransformers)
- **Critic** (8006): Quality assessment and review
- **Chief Editor** (8001): Workflow orchestration
- **Memory** (8007): PostgreSQL + vector search storage
- **Reasoning** (8008): Nucleoid symbolic logic engine

### RESTful API Endpoints
- Example endpoints:
  ```
  GET    /api/v1/articles/{id}           # Get article by ID
  POST   /api/v1/articles/search         # Search articles
  GET    /api/v1/entities/{id}           # Get entity details
  POST   /api/v1/entities/search         # Search entities
  ```

## Security Guidelines

### API Security
- **Input Validation**: Use Pydantic models for all API inputs
- **Error Exposure**: Never expose internal paths or sensitive info in errors
- **Resource Limits**: Implement timeouts and memory limits
- **Health Checks**: Secure health endpoints without sensitive data

### Model Security
- **Model Validation**: Verify model checksums before loading
- **Safe Loading**: Use `trust_remote_code=False` unless explicitly required
- **Memory Protection**: Clear sensitive data from GPU memory after use

## Documentation Requirements

### Code Documentation
- **Inline Comments**: Explain complex logic, especially GPU operations
- **Performance Notes**: Document memory usage and optimization decisions
- **Error Handling**: Comment on exception handling strategy

### Change Documentation
- **CHANGELOG.md**: Required for all releases with performance metrics
- **Agent Updates**: Document in appropriate `markdown_docs/agent_documentation/`
- **Technical Changes**: Add to `markdown_docs/development_reports/`

### Validation Checklist
Before any commit:
- [ ] Code follows PEP 8 and type hint requirements
- [ ] Error handling implemented with specific exceptions
- [ ] Performance logging included for GPU operations
- [ ] Documentation updated in correct `markdown_docs/` location
- [ ] Tests include error cases and performance validation
- [ ] GPU memory cleanup verified
- [ ] MCP integration tested if applicable
- [ ] CHANGELOG.md updated with metrics

---

**Remember**: This is a production system. All changes must maintain system stability, follow established patterns, and include comprehensive error handling and performance monitoring.
