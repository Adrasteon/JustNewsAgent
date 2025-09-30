---
title: MCP Bus Operations Guide (Port 8000)
description: Operating the MCP Bus for agent registration and tool routing in production.
tags: [operations, mcp, bus, registration, tools, routing]
status: current
last_updated: 2025-09-12
---

# MCP Bus Operations Guide (Port 8000)

See also: [Operator Guide — Systemd](OPERATOR_GUIDE_SYSTEMD.md), [GPU Orchestrator Operations](GPU_ORCHESTRATOR_OPERATIONS.md), [Preflight Runbook](preflight_runbook.md), [Ops Quick Reference](OPERATIONS_QUICK_REFERENCE.md)

The MCP Bus coordinates agent registration and tool routing across all services.

## Endpoints
- GET /health → {status}
- GET /agents → { name: address }
- POST /register → body: { name, address, tools? }
- POST /call → body: { agent, tool, args, kwargs }

## Typical flows
- Agent startup self-registers:
```bash
curl -s -X POST http://127.0.0.1:8000/register \
	-H 'Content-Type: application/json' \
	-d @- <<'JSON'
{
	"name": "scout",
	"address": "http://localhost:8002",
	"tools": ["deep_crawl", "discover"]
}
JSON
```

- Operator ensures registration (fallback):
```bash
curl -s http://127.0.0.1:8000/agents | jq
```

- Call a tool via bus (example):
```bash
curl -s -X POST http://127.0.0.1:8000/call \
	-H 'Content-Type: application/json' \
	-d @- <<'JSON'
{
	"agent": "archive",
	"tool": "fetch_article",
	"args": ["https://example.com/article"],
	"kwargs": {}
}
JSON
```

## Readiness and registration
- The bus is healthy once /health returns ok.
- Agents register automatically on start; the crawler can backfill using known ports if needed.
- If /agents is empty but agents are running, restart an agent or use the crawler’s ensure_agents_registered().

## Troubleshooting
- Bus active but port closed: verify ExecStart launches uvicorn for agents/mcp_bus/main:app.
- Registration failing: confirm agent is reachable on its port and has correct MCP_BUS_URL.
- Logs: `journalctl -u justnews@mcp_bus -e`

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

