---
title: Preflight Runbook — Model Gating and Preload Failures
description: How to use the preflight gate to enforce model readiness and troubleshoot preload failures from the GPU Orchestrator.
tags: [operations, preflight, orchestrator, models, readiness]
status: current
last_updated: 2025-09-12
---

# Preflight Runbook: Model Gating and Preload Failures

See also: [Operator Guide — Systemd](OPERATOR_GUIDE_SYSTEMD.md), [GPU Orchestrator Operations](GPU_ORCHESTRATOR_OPERATIONS.md), [MCP Bus Operations](MCP_BUS_OPERATIONS.md), [Ops Quick Reference](OPERATIONS_QUICK_REFERENCE.md)

This runbook explains how to use the preflight gate to enforce model readiness and how to diagnose and fix model preload failures surfaced by the GPU Orchestrator.

## What the preflight does
- Verifies the GPU Orchestrator is reachable on port 8014.
- Ensures the Model Store is canonical (STRICT mode, no downloads).
- Triggers/monitors a models preload warmup and gates startup until `all_ready=true`.
- Validates NVIDIA MPS is enabled and operational for GPU resource isolation.
- Exits non‑zero on failure to stop systemd from starting dependent services.

## Quick usage

Gate-only (used by systemd ExecStartPre):

```bash
/home/adra/justnewsagent/JustNewsAgent/deploy/systemd/preflight.sh --gate-only <instance>
# example
/home/adra/justnewsagent/JustNewsAgent/deploy/systemd/preflight.sh --gate-only mcp_bus
```

Full validation run (interactive ops):

```bash
sudo /home/adra/justnewsagent/JustNewsAgent/deploy/systemd/preflight.sh
```

NOPASSWD helper (install sudo exemption and optionally run):

```bash
sudo /home/adra/justnewsagent/JustNewsAgent/deploy/systemd/setup_preflight_nopasswd.sh --install
sudo /home/adra/justnewsagent/JustNewsAgent/deploy/systemd/setup_preflight_nopasswd.sh --run-gate-only mcp_bus --timeout 300
```

Environment knobs:
- `GATE_TIMEOUT` (seconds): override default wait in systemd drop-in (e.g. 300)
- `REQUIRE_BUS=0` for gpu_orchestrator instance to bypass waiting on MCP bus
- `MODEL_STORE_ROOT`, `BASE_MODEL_DIR`, `STRICT_MODEL_STORE=1`

## Symptom
- Preflight or crawler aborts due to /models/preload 503 or /models/status with `all_ready=false` and errors.

## Where to look
- Preflight output and cached summary: `$HOME/.cache/justnews/preflight/summary_*.json`
- Orchestrator status: `GET http://localhost:8014/models/status`
- Orchestrator logs (journalctl): `sudo journalctl -u justnews@gpu_orchestrator -e`
- AGENT_MODEL_MAP: `markdown_docs/agent_documentation/AGENT_MODEL_MAP.json`
 - Preflight cache: `$HOME/.cache/justnews/preflight/summary_*.json`

## Common causes and fixes
1. Missing Model Store snapshot
   - Cause: Model not materialized at expected agent path under MODEL_STORE_ROOT.
   - Fix: Sync/copy the snapshot directory; verify checksums if applicable.
   - Verify: Run preflight again; /models/status shows `ok` for that model.

2. Wrong AGENT_MODEL_MAP key or model id
   - Cause: Agent key mismatch or model id typo.
   - Fix: Correct the agent key and model identifier in AGENT_MODEL_MAP.json.
   - Verify: `curl :8014/models/preload` (refresh=true) then poll /models/status.

3. Permission issues
   - Cause: Service user can’t read Model Store directories.
   - Fix: Adjust ownership/permissions; ensure execute/search on parent directories.
   - Verify: `sudo -u <service-user> ls -l <model_path>` succeeds.

4. OOM during warmup (rare in CPU preload)
   - Cause: Large tokenizer/model init; fragmentation.
   - Fix: Ensure STRICT_MODEL_STORE=1 (no downloads); increase swap; reduce concurrent warmups if later parallelized.
   - Verify: Retry and monitor `nvidia-smi` (should be mostly idle during CPU preload).

5. MPS not operational
   - Cause: NVIDIA MPS daemon not running or misconfigured.
   - Fix: Start MPS daemon with `nvidia-cuda-mps-control -d`; verify `/tmp/nvidia-mps/` exists.
   - Verify: Check `curl :8014/gpu/info` shows `mps_enabled: true`; verify control process running.

## Quick recovery
- Refresh job to clear past failures: `POST /models/preload {"refresh": true}`
- Confirm readiness: `GET /models/status` → `all_ready: true`

## Reference: Orchestrator endpoints
- `POST /models/preload {"refresh": <bool>}` → 202/503
- `GET /models/status` → `{in_progress, summary, errors, all_ready}`
- `GET /health` → basic status

## Preventive steps
- Keep AGENT_MODEL_MAP.json accurate with minimal diff reviews.
- Use STRICT_MODEL_STORE=1, MODEL_STORE_ROOT set in environment files.
- Run preflight via systemd ExecStartPre gate before starting agents/crawls.

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

