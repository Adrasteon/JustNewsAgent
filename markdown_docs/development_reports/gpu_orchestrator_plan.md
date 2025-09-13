---
title: GPU Orchestrator Migration Plan (V4)
description: Auto-generated description for GPU Orchestrator Migration Plan (V4)
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# GPU Orchestrator Migration Plan (V4)

Centralize GPU control into a dedicated, systemd-managed Orchestrator that provides admission control, leasing, and telemetry for all CUDA work. Remove agent-local background GPU cleanup to eliminate race conditions and device instability.

## Objectives
- Stop GPU-related hardware crashes by eliminating uncoordinated, agent-level memory cleanup.
- Introduce a single authority (“GPU Orchestrator”) for VRAM policy, admission, and scheduling.
- Maintain throughput while achieving deterministic VRAM usage and safer model lifecycle.
- Provide observability, guardrails, and SLOs for GPU usage across agents.

## Scope
- In-scope: NewsReader, Synthesizer (V3), Analyst, Fact Checker, and any agent performing CUDA work.
- Out-of-scope (Phase 0–2): Replacing model implementations; centralized execution reserved for Phase 3.

## Assumptions
- Single-GPU host initially (RTX 3090, 24GB), CUDA and NVML available.
- Agents are FastAPI services and can adopt an SDK/context-manager wrapper for leases.
- MCP bus remains the inter-agent call mechanism; Orchestrator is an additional service.

## Non-Goals
- No immediate change to model architectures or precision settings beyond existing FP16/TRT usage.
- No immediate multi-GPU/MIG placement (planned for Phase 4).

---

## Architecture Overview
- GPU Orchestrator (FastAPI service, systemd-managed)
  - Responsibilities: NVML monitoring, admission/lease control, device placement, optional model preload hints, eviction only when safe/idle, metrics/telemetry.
  - State: In-memory lease table + persisted logs; optional Redis later (not required initially).
  - Policy: Per-agent quotas, priorities, timeouts, batch-size hints.
- Agent SDK (Python)
  - Context manager to request/renew/release leases around any CUDA section.
  - Minimal overhead; cooperative enforcement.

### API Contract (MCP-aligned HTTP)
- POST /register_agent
  - in: {agent, version, capabilities, requires_models: [ids]}
  - out: {status: "ok"}
- POST /preload_model
  - in: {model_id, est_vram_gb, precision, device_pref?}
  - out: {status, device}
- POST /request_lease
  - in: {agent, operation, model_id?, est_vram_gb, batch?, max_duration_ms, priority?}
  - out: {granted: bool, lease_id?, device?, reason?, estimated_wait_ms?}
- POST /renew_lease
  - in: {lease_id, add_duration_ms}
  - out: {ok: bool}
- POST /release_lease
  - in: {lease_id, peak_vram_gb, success: bool}
  - out: {ok: bool}
- GET /gpu_state
  - out: {devices: [{id, total_gb, used_gb, temp_c, leases: [...] }], policy: {...}}
- POST /evict_model
  - in: {model_id}
  - out: {status}
- GET /health
  - out: {status: "ok", uptime_s, version}

Error modes
- 409 CONFLICT for denied admissions; 408 for expired leases; 503 for degraded GPU.

### Enforcement Model Options
- Option A (Phase 2): Cooperative leases with an SDK wrapper; agents retain execution but must acquire a lease before CUDA work.
- Option B (Phase 3+): Centralized execution pools; orchestrator hosts model runtimes, agents submit RPCs for inference.

---

## Phased Rollout

### Phase 0 — Hotfix (Today)
Goal: Eliminate unsafe background cleanup and stabilize e2e.

Actions
- NewsReader: Disable background GPU memory monitor and any forced `torch.cuda.empty_cache()` loops that may run mid-inference.
  - Gate with `GPU_MONITOR_ENABLED=0` (default off) or remove monitor entirely.
- Ensure cleanup happens only after work completes (engine `.cleanup()` on shutdown or post-request), never in background during in-flight ops.
- Align Synthesizer endpoint naming between e2e and service (update e2e calls or add alias endpoint).
- Validate with `complete_production_e2e_test.py` under load; capture VRAM/temps metrics only.

Deliverables
- Safe guard/disable in `agents/newsreader/tools.py`.
- Minor e2e/service endpoint alignment.
- Test report: e2e passes without device crashes.

Exit Criteria
- Zero device faults or CUDA resets during e2e; stable throughput within 5–10%.

### Phase 1 — Monitor-Only Orchestrator (Read-Only)
Goal: Introduce the Orchestrator in advisory mode to validate policy decisions.

Actions
- Create service skeleton at `mcp_bus/gpu_orchestrator/` (FastAPI): endpoints above, decisions logged but no blocking.
- Implement NVML sampling (utilization, memory, temps) at 1s cadence.
- Agents register and optionally call `/request_lease` but proceed regardless; orchestrator logs "would grant/deny" decisions.
- Systemd unit + health check + Prometheus metrics.

Deliverables
- Orchestrator service (read-only) + unit file, docs.
- Dashboards: VRAM usage, lease demand vs. capacity, thermal headroom.

Exit Criteria
- Decision logs cover 95%+ CUDA sections; policies tuned with real demand.

### Phase 2 — Enforced Leases (Cooperative)
Goal: Enforce admission control before CUDA work.

Actions
- Add Agent SDK: `with gpu_lease(agent, operation, model_id, est_vram_gb, max_duration_ms): ...`
- Instrument CUDA sections:
  - NewsReader: engine load, embedding, model inference.
  - Synthesizer V3: BART/FLAN‑T5 pipelines, clustering steps if GPU.
  - Analyst: TensorRT runs and any PyTorch fallbacks.
  - Fact Checker: GPU-enabled operations, if any.
- Implement lease renewal for long ops; release on completion with peak VRAM.
- Implement queueing/priorities and timeouts in the Orchestrator.

Deliverables
- SDK package (internal) + usage patched in agents.
- Integration tests: deny path, timeout, renewal, release correctness.

Exit Criteria
- No CUDA work without a lease in target agents.
- P95 lease wait times within SLO (configure per agent).

### Phase 3 — Engine Pools & Selective Central Execution
Goal: Centralize the heaviest VRAM consumers to improve packing and batching.

Actions
- Host Synthesizer V3 runtime(s) in the Orchestrator (or sibling service) with batched RPC.
- Optional: Warm model pools; preloaded/pinned models with eviction policy.
- Measure throughput gains and VRAM stability vs. cooperative-only mode.

Deliverables
- Centralized inference endpoints for selected models.
- Performance report vs. Phase 2.

Exit Criteria
- Higher throughput or lower VRAM fragmentation with no instability.

### Phase 4 — Advanced Scheduling
Goal: Optimize beyond single-GPU.

Actions
- MIG partitions or multi-GPU placement; cross-GPU balancing.
- SLA tiers (latency vs. throughput) and preemption-by-checkpoint where safe.

Deliverables
- Placement strategies, config, and tests.

Exit Criteria
- Predictable performance across GPUs/MIG; maintained stability.

---

## Operationalization (systemd, Observability, SLOs)
- systemd service: Restart=always, WatchdogSec, Before=agents, After=network.target; `ExecStart` runs FastAPI app.
- Health endpoints used by systemd watchdog; readiness gate before agents start (agents DependOn orchestrator).
- Telemetry: structured logs, Prometheus counters/gauges/histograms (leases granted/denied, wait times, temps, VRAM usage, eviction counts).
- Alerts: high temp, sustained >90% VRAM, repeated denials.

## Lease Data Model (internal)
- Lease {id, agent, operation, model_id?, est_vram_gb, granted_ts, max_duration_ms, device, priority, last_renew_ts, status}
- States: requested → granted → renewing → released | expired | revoked

## Policies and Config
- Per-agent quotas and soft/hard limits.
- Priority bands: critical > normal > background.
- Cleanup/Eviction: only when idle or with explicit cooperation; never mid-inference.
- Batch-size hints: orchestrator suggests; agents may adapt.

## Failure Modes & Mitigations
- Orchestrator crash: systemd restart + lease recovery (treat unknown leases as expired after grace period).
- Deadlocks/timeouts: hard lease timeouts; auto-evict expired; circuit-breaker for hot agents.
- Thermal throttling: deny new leases and lower batch-size hints.

## Testing & Validation (Quality Gates)
- Build/Lint/Types: PASS (PEP8/ruff), type hints in orchestrator and SDK.
- Unit tests: lease happy path, denial, renewal, expiry, recovery.
- Integration tests: agents obtain leases; denial paths return 409; timeouts 408; degraded 503.
- Load test: e2e with concurrency; verify wait-time SLOs and zero device faults.
- GPU memory cleanup verified: no background forced cleanup during in-flight work.

## Work Breakdown (WBS)
- Phase 0 (1 day)
  - Disable NewsReader monitor; align Synthesizer endpoint; run e2e validation.
- Phase 1 (2–3 days)
  - Orchestrator skeleton, NVML sampling, read-only decision logging, systemd unit, dashboards.
- Phase 2 (3–5 days)
  - SDK + agent instrumentation; queueing/timeouts; integration tests.
- Phase 3 (timeboxed spike 3–5 days)
  - Centralized execution for Synthesizer V3; measure gains.
- Phase 4 (as needed)
  - MIG/multi-GPU; SLA tiers.

## Risks
- SPOF: mitigate with systemd watchdog, minimal dependencies, stateless design.
- Latency overhead: keep lease RPC <1–2 ms; batch renewals.
- Compliance drift: CI checks to ensure CUDA entrypoints use the SDK.

## Rollback Plan
- Feature flags per phase; agents can bypass leases with env var in emergencies.
- Keep Phase 0 changes (monitor disabled) regardless; it’s strictly safer.

## Success Criteria
- Zero GPU crashes during e2e for 7 consecutive full runs.
- P95 lease wait < 150 ms under nominal load; denial rate < 1% after warm-up.
- Throughput within ±10% of baseline (or better) post Phase 2.
- Clean logs: no mid-inference cache clears; no CUDA device resets.

## Implementation Pointers (Repo)
- Phase 0 edits: `agents/newsreader/tools.py` (monitor off/guard), `complete_production_e2e_test.py` or `agents/synthesizer/main.py` (endpoint alignment).
- New service (Phase 1+): `mcp_bus/gpu_orchestrator/` with `main.py`, `policy.py`, `nvml.py`, `leases.py`, `requirements.txt`.
- SDK: `agents/common/gpu_orchestrator_sdk.py` providing `gpu_lease()` context manager and HTTP client.
- Systemd unit: `justnews-gpu-orchestrator.service` under deploy assets, referenced by `start_services_daemon.sh`.

---

Document owner: Platform/Infra
Last updated: 2025-09-10

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

