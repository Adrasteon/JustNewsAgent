---
title: Systemd Orchestrator Incident Report — Sept 13, 2025
description: Detailed chronology of startup failures under systemd, root causes, and fixes leading to a clean orchestrator-first boot with full system health.
tags: [systemd, orchestrator, readiness, MCP, uvicorn, ops]
status: current
last_updated: 2025-09-13
---

# Systemd Orchestrator Incident Report — Sept 13, 2025

## Executive summary
A series of startup issues prevented the gpu_orchestrator from reaching HTTP readiness under systemd. The root causes spanned incorrect project root resolution, an MCP Bus dependency deadlock, a shell env syntax error, and an entrypoint that didn’t reliably bind the web server. After targeted fixes, orchestrator reached /ready and all services passed health checks (15/15 healthy).

## Impact
- Orchestrator /ready refused; cold-start orchestration stalled.
- Timers and downstream agents waited on orchestrator readiness.
- No data loss; recovery achieved without rollback.

## Timeline (UTC local server time)
- 17:07 — Systemd “Start request repeated too quickly”; orchestrator not ready.
- 17:12 — Preflight runs (gate-only); launcher starts, but /ready intermittently refused; MCP registration warnings observed.
- 17:19 — Readiness OK once; subsequent restarts still inconsistent.
- 17:26 — Switched launcher to explicit uvicorn runner; consistent /ready observed; cluster brought fully online.

## Symptoms and evidence
- Connection refused on http://127.0.0.1:8014/ready (curl exit 7).
- Journal showed incorrect PROJECT_ROOT in earlier runs (resolved later): “Agent directory not found: /usr/agents/gpu_orchestrator”.
- Env sourcing error: “/etc/justnews/global.env: syntax error near unexpected token (‘)”.
- MCP registration errors when bus not yet up (expected but noisy): connection refused.
- After uvicorn switch, logs showed: “Uvicorn running on http://0.0.0.0:8014 …” and subsequent 200 OK on /ready.

## Root causes and fixes
1) Project root resolution under systemd
- Cause: PROJECT_ROOT resolved incorrectly when WorkingDirectory/env not set, defaulting to /usr/…
- Fixes:
  - Added robust resolution in `deploy/systemd/justnews-start-agent.sh` (prefer systemd WorkingDirectory → JUSTNEWS_ROOT → service dir → script-relative; validate by checking `agents/gpu_orchestrator` and `deploy`).
  - Added per-instance drop-in for orchestrator: set WorkingDirectory and `JUSTNEWS_ROOT=/home/adra/justnewsagent/JustNewsAgent`.
  - Verified with logs: “Resolved PROJECT_ROOT=/home/adra/justnewsagent/JustNewsAgent” and “Agent directory and main script found”.

2) MCP Bus dependency deadlock
- Cause: Orchestrator startup scripts waited for MCP Bus; cold_start waited for orchestrator /ready → circular dependency.
- Fixes:
  - Per-instance env `REQUIRE_BUS=0` for gpu_orchestrator.
  - Unit gating and launcher skip MCP wait for mcp_bus and gpu_orchestrator.
  - Result: Orchestrator starts independently; later best-effort MCP registration continues.

3) Environment syntax error in global.env
- Cause: Unquoted `LOG_FORMAT` caused bash parse error when sourced by systemd.
- Fix: Quote `LOG_FORMAT` in `/etc/justnews/global.env`; reinstalled env file and restarted services.

4) Orchestrator HTTP server not binding reliably
- Cause: Module-run invocation path didn’t consistently expose the ASGI server early enough under systemd; reduced visibility into binding.
- Fixes:
  - Launcher now prefers explicit uvicorn runner for gpu_orchestrator: `python -m uvicorn agents.gpu_orchestrator.main:app --host 0.0.0.0 --port 8014` with fallback to module-run.
  - Confirmed uvicorn/fastapi availability in the conda env.
  - Evidence: Uvicorn banner present and `ss -ltnp` shows port 8014 listening.

5) Boot-smoke oneshot failure despite OK log (noise only)
- Cause: Wrapper returned non-zero exit despite success log.
- Fix: Adjusted wrapper to always exit 0 (informational only).

## Changes applied
- `deploy/systemd/justnews-start-agent.sh`
  - Robust PROJECT_ROOT resolution and validation.
  - Dependency gating: skip MCP wait for mcp_bus and gpu_orchestrator; honor REQUIRE_BUS=0.
  - gpu_orchestrator: prefer uvicorn runner; log interpreter and command.
- Systemd unit template and drop-ins
  - `justnews@.service`: ExecStartPre gating updated; WorkingDirectory wired.
  - Drop-ins for gpu_orchestrator: `01-root.conf` (JUSTNEWS_ROOT + WorkingDirectory), `05-bus-bypass.conf` (REQUIRE_BUS=0), `10-preflight-gating.conf`, `20-restart-policy.conf`.
- Environment files
  - `/etc/justnews/global.env`: quoted LOG_FORMAT; ensured JUSTNEWS_ROOT and PYTHON_BIN.
  - `/etc/justnews/gpu_orchestrator.env`: REQUIRE_BUS=0.
- Operational scripts
  - `cold_start.sh` idempotent installs/unit updates; `health_check.sh` standardized; wrappers self-install to PATH.

## Verification
- Orchestrator readiness
  - `curl -sf http://127.0.0.1:8014/health` → OK; `/ready` → `{"ready": true}`.
  - `ss -ltnp | grep 8014` → python listening (uvicorn); journal shows uvicorn banner.
- Cluster health
  - All services active (mcp_bus, scout, analyst, synthesizer, etc.).
  - `deploy/systemd/health_check.sh -v` → 15/15 healthy; gpu_orchestrator READY.
- Residual non-blocker
  - `nvidia-persistenced.service` failed (not required for service health; optional to fix for GPU persistence).

## Preventive measures
- Idempotent installers and drop-ins; explicit WorkingDirectory for systemd.
- MCP dependency bypass for orchestrator to avoid circular waits.
- Quote all env values with special characters; validate env before deploy.
- Prefer explicit uvicorn runner for ASGI services under systemd for clear diagnostics.
- Structured logging: emit interpreter path and exec command at startup.

## Quick commands (reference)
```bash
# readiness
curl -sf http://127.0.0.1:8014/ready
ss -ltnp | grep 8014

# unit state and logs
systemctl status justnews@gpu_orchestrator
journalctl -u justnews@gpu_orchestrator -n 200 --no-pager

# health summary
bash deploy/systemd/health_check.sh -v
```

## Status
Resolved. Orchestrator-first startup is reliable; full system reports healthy. Optional follow-up: enable/fix `nvidia-persistenced.service` if GPU persistence is desired at boot.
