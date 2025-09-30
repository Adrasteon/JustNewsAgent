---
title: Operator Guide — Systemd Deployment and Operations
description: Production operations for systemd deployment, preflight gating, GPU orchestrator, MCP bus, and common service tasks.
tags: [operations, systemd, preflight, orchestrator, mcp, deployment]
status: current
last_updated: 2025-09-12
---

# JustNews V4 Operator Guide — Systemd Deployment and Operations

See also: [GPU Orchestrator Operations](GPU_ORCHESTRATOR_OPERATIONS.md), [MCP Bus Operations](MCP_BUS_OPERATIONS.md), [Preflight Runbook](preflight_runbook.md), [Ops Quick Reference](OPERATIONS_QUICK_REFERENCE.md)

This guide consolidates production operations: systemd deployment, preflight gating, GPU Orchestrator model management, MCP Bus operations, and common service tasks.

## 1) Prerequisites
- Python env installed and services configured (`/etc/justnews/*.env`)
- Model Store materialized; STRICT mode enabled (no downloads in prod)
- Ports reserved: MCP Bus 8000, Orchestrator 8014, Agents 8001–8012

## 2) Install systemd scaffold
Files live under `deploy/systemd`.

1. Install unit template and drop-ins:
```bash
sudo install -D -m0644 deploy/systemd/units/justnews@.service /etc/systemd/system/justnews@.service
sudo mkdir -p /etc/systemd/system/justnews@.service.d
sudo install -m0644 deploy/systemd/units/overrides/10-preflight-gating.conf /etc/systemd/system/justnews@.service.d/
sudo install -m0644 deploy/systemd/units/overrides/20-restart-policy.conf /etc/systemd/system/justnews@.service.d/
```
2. Configure environment files:
```bash
sudo mkdir -p /etc/justnews
sudo install -m0644 deploy/systemd/examples/justnews.env.example /etc/justnews/global.env
sudo install -m0644 deploy/systemd/examples/gpu_orchestrator.env.example /etc/justnews/gpu_orchestrator.env
# Create per-service envs as needed: /etc/justnews/<instance>.env
```
3. Reload systemd:
```bash
sudo systemctl daemon-reload
```

## 3) Preflight gating (models readiness)
- Gate script: `deploy/systemd/preflight.sh`
- Gate-only mode is wired via ExecStartPre and enforces `all_ready=true` from the orchestrator.
- NOPASSWD helper: `deploy/systemd/setup_preflight_nopasswd.sh` to install sudo exemptions and run gates.

Examples:
```bash
# Install NOPASSWD for preflight
sudo deploy/systemd/setup_preflight_nopasswd.sh --install

# Manually gate mcp_bus
deploy/systemd/preflight.sh --gate-only mcp_bus

# Gate with longer timeout
GATE_TIMEOUT=300 deploy/systemd/preflight.sh --gate-only mcp_bus
```

## 4) GPU Orchestrator (8014)
Purpose: model preload, GPU telemetry, policy/lease endpoints.

Key endpoints:
- `GET /health` — orchestration service health
- `POST /models/preload {"refresh": false}` — start warmup job
- `GET /models/status` — readiness: `{in_progress, summary, errors, all_ready}`
- `GET /gpu/info` — optional GPU info

Operational patterns:
- PROD gate requires `all_ready=true` before mcp_bus and agents start.
- To rebuild readiness:
```bash
curl -X POST -H 'Content-Type: application/json' \
  -d '{"refresh": true}' http://127.0.0.1:8014/models/preload
watch -n1 curl -s http://127.0.0.1:8014/models/status
```

## 5) MCP Bus (8000)
Purpose: central agent registration and tool routing.

Key endpoints:
- `GET /health` — bus health
- `GET /agents` — map of registered agents
- `POST /register {name, address, tools?}` — agent self-registration
- `POST /call {agent, tool, args, kwargs}` — tool routing

Operational tips:
- On service restart, agents may auto-register; otherwise the crawler can ensure registration.
- To verify:
```bash
curl -s http://127.0.0.1:8000/agents | jq
```

## 6) Manage services
Enable and start core instances:
```bash
for s in mcp_bus gpu_orchestrator chief_editor scout fact_checker analyst synthesizer critic memory reasoning newsreader archive dashboard balancer; do
  sudo systemctl enable --now justnews@"$s".service
done
```

Common operations:
```bash
sudo systemctl status justnews@scout.service
sudo systemctl restart justnews@scout.service
sudo journalctl -u justnews@scout.service -e
```

Adjust restart policy (already preconfigured):
- Start limits in drop-in are under [Unit]; Restart/RestartSec under [Service].

## 7) Model Store and strict mode
Environment variables (in `/etc/justnews/*.env`):
- `MODEL_STORE_ROOT` — root of immutable model snapshots
- `BASE_MODEL_DIR` — optional subdir, agent-relative
- `STRICT_MODEL_STORE=1` — enforce no network downloads

Checks:
```bash
curl -s http://127.0.0.1:8014/models/status | jq .summary
```

## 8) Health and readiness sweep
Quick local sweep:
```bash
for p in 8000 8014 8001 8002 8003 8004 8005 8006 8007 8008 8009 8010 8011 8012; do
  echo "--- :$p /health"; curl -sS http://127.0.0.1:$p/health; echo; done
```

## 9) Troubleshooting
- Preflight failing despite `all_ready=true`: ensure gate-only script is updated (whitespace-tolerant check) and GATE_TIMEOUT is sufficient.
- MCP bus active but port closed: verify ExecStart actually launches uvicorn; check logs and env files.
- systemd warning: ensure `StartLimitIntervalSec` and `StartLimitBurst` are under [Unit] (fixed in overrides/20‑restart‑policy.conf).

## 10) Change log hooks
- Update `CHANGELOG.md` for performance changes and endpoint additions.
- Keep `markdown_docs/agent_documentation/` current for operator guidance.

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

