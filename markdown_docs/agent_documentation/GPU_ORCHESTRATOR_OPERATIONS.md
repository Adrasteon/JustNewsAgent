---
title: GPU Orchestrator Operations (Port 8014)
description: Operating the GPU Orchestrator for model preload/readiness gating and GPU telemetry in production.
tags: [operations, orchestrator, models, preload, readiness, gpu]
status: current
last_updated: 2025-09-12
---

# GPU Orchestrator Operations (Port 8014)

See also: [Operator Guide — Systemd](OPERATOR_GUIDE_SYSTEMD.md), [MCP Bus Operations](MCP_BUS_OPERATIONS.md), [Preflight Runbook](preflight_runbook.md), [Ops Quick Reference](OPERATIONS_QUICK_REFERENCE.md)

The GPU Orchestrator manages model preload/readiness and provides GPU telemetry.

## Environment
- MODEL_STORE_ROOT: path to immutable model snapshots
- BASE_MODEL_DIR: optional subdirectory for agent-relative paths
- STRICT_MODEL_STORE=1: enforce no network downloads (prod)

## Endpoints
- GET /health → basic status
- POST /models/preload {"refresh": bool} → start background warmup (202/503)
- GET /models/status → { in_progress, summary {total, done, failed}, errors[], all_ready }
- GET /gpu/info → GPU state (optional)

## Operations
- Start warmup (no refresh):
```bash
curl -s -X POST -H 'Content-Type: application/json' \
 -d '{"refresh": false}' http://127.0.0.1:8014/models/preload
```
- Force rebuild readiness:
```bash
curl -s -X POST -H 'Content-Type: application/json' \
 -d '{"refresh": true}' http://127.0.0.1:8014/models/preload
watch -n1 curl -s http://127.0.0.1:8014/models/status
```
- Check readiness summary:
```bash
curl -s http://127.0.0.1:8014/models/status | jq '{in_progress, all_ready, summary}'
```

## Preflight gating
- Gate script enforces `all_ready=true` before agents start.
- ExecStartPre runs: `deploy/systemd/preflight.sh --gate-only <instance>`
- Cached summaries: `~/.cache/justnews/preflight/summary_*.json`

## Troubleshooting
- all_ready=false: check AGENT_MODEL_MAP.json and files under MODEL_STORE_ROOT
- 503 on preload: resolve errors, re-run with refresh=true
- Logs: `journalctl -u justnews@gpu_orchestrator -e`

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

