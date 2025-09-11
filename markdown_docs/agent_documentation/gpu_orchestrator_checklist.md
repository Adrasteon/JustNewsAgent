## GPU Orchestrator Implementation Checklist (E2E Readiness)

Last Updated: 2025-09-11

### ✅ Completed
- Service created: `agents/gpu_orchestrator/main.py` with endpoints `/health`, `/ready`, `/gpu/info`, `/policy` (GET/POST, SAFE_MODE read-only), `/allocations` (placeholder), `/lease`, `/release`, `/metrics`
- Port assigned & documented: 8014 (added to canonical mapping)
- Systemd startup script support: `justnews-start-agent.sh` includes `gpu_orchestrator`
- Example env file: `deploy/systemd/examples/gpu_orchestrator.env.example`
- README updated with orchestrator section & port alignment
- Canonical port mapping updated (8011 analytics, 8012 archive, 8013 dashboard, 8014 orchestrator)
- Health script & enable_all ordering updated to include orchestrator

### 🟡 Pending (Required for E2E Validation)
1. Install production env file: `/etc/justnews/gpu_orchestrator.env` (copy from example)
2. Lightweight client SDK: `agents/common/gpu_orchestrator_client.py` ✅
   - `get_gpu_info()` (timeout 2s, fallback `{available:false}`) ✅
   - `get_policy()` (cache 30s, SAFE_MODE flag) ✅
3. Integrate SDK in GPU-capable agents (Analyst first) ✅ (baseline gating logic present; extend to others later)
4. Remove or disable any legacy per-agent GPU watchdog logic ✅ (auto-start disabled in `gpu_monitoring_enhanced.py` when orchestrator detected)
5. Dashboard ingestion (optional) ✅ basic proxy endpoints wired
6. Add orchestrator to any global readiness gate (script update) – PENDING
7. Tests:
   - Unit: orchestrator endpoints ✅ `test_gpu_orchestrator_endpoints.py`
   - Client fallback & caching ✅ `test_gpu_orchestrator_client.py`
   - Analyst gating ✅ `test_analyst_gpu_gating.py`
   - E2E: orchestrator up + Analyst GPU + others CPU (run & capture log) – PENDING (`e2e_orchestrator_validation.py`)
   - Smoke harness ✅ `orchestrator_analyst_smoke_test.py`
   - Automated validation script ✅ `e2e_orchestrator_analyst_run.py`

### 🔵 Nice-to-Have (Post E2E)
- NVML-based metrics (granular utilization, PCIe throughput) when out of SAFE_MODE
- Policy mutation (POST /policy) when SAFE_MODE=false with strict validation (baseline validation present)
- Event streaming (Server-Sent Events or WebSocket) for dashboard live updates
- Advanced allocation strategy (priority queues, fractional memory accounting)

### ❗ Risks / Watchpoints
- Avoid accidental GPU initialization in orchestrator when SAFE_MODE=true
- Ensure timeouts small so agents never block on orchestrator (fail closed → CPU)
- Maintain single authoritative port mapping to prevent regression (update health_check & enable_all kept aligned)

### 🧪 E2E Success Criteria
| Criterion | Description | Status |
|-----------|-------------|--------|
| Orchestrator systemd start | `systemctl start justnews@gpu_orchestrator` succeeds & `/health` 200 | Pending runtime
| `/gpu/info` works | Returns `available` key & list (even empty) | ✅ Code path
| SAFE_MODE enforced | POST /policy blocked / read-only note | ✅ Implemented
| Agents respect SAFE_MODE | Analyst uses GPU, others CPU, no stray allocs | Pending integration
| Fallback path | Killing orchestrator causes agents to continue on CPU | Pending test
| E2E run stable | Full small pipeline run w/out GPU crash | Pending test
| Lease SAFE_MODE behavior | `/lease` returns note and no GPU index when SAFE_MODE=true | ✅ Tested (`test_gpu_orchestrator_leasing.py`)

### 📌 Next Action (Recommended Order)
 Add orchestrator to global readiness gate script (Pending item 6) ✅ `health_check.sh` now queries /ready
2. Run mini E2E (5–10 articles) with SAFE_MODE=true capturing lease denial note (test-level validated) ✅ `run_safe_mode_demo.py` (cycle_on shows denied)
3. Toggle SAFE_MODE=false; validate policy mutation & lease GPU assignment (runtime) ✅ `cycle_off` shows `safe_mode:false`, `lease.granted:true`, active_leases=1
4. Capture `/metrics` snapshot pre/post lease cycles for dashboard reference (include active_leases gauge) ✅ `metrics_snapshot.json` & `metrics_snapshot.txt` generated via `generate_orchestrator_metrics_snapshot.py`
5. Implement NVML enrichment (guarded by SAFE_MODE & availability)
   - ✅ Scaffold added: optional `ENABLE_NVML=true` env initializes NVML in lifespan, enriches `/gpu/info` & `/metrics` with flags (only when SAFE_MODE=false)
   - ✅ Lease TTL added (`GPU_ORCHESTRATOR_LEASE_TTL`, default 3600s) with opportunistic purge & `lease_expired_total` metric

## 🔄 Fresh Start + Mini E2E Procedure
1. Stop all agents (`systemctl stop 'justnews@*'`) or use project stop script.
2. Clear stale artifacts (optional): remove `orchestrator_demo_results/*` if confusing.
3. Ensure `/etc/justnews/gpu_orchestrator.env` (copy from example) has `SAFE_MODE=true` for initial protective posture.
4. Start orchestrator only: `systemctl start justnews@gpu_orchestrator` then verify `/health` & `/ready`.
5. Run `python run_safe_mode_demo.py` (captures safe-mode lease denial baseline).
6. Edit env: set `SAFE_MODE=false` (and optionally `ENABLE_NVML=true`), restart orchestrator.
7. Re-run `python run_safe_mode_demo.py` then `python generate_orchestrator_metrics_snapshot.py`.
8. Run `python scripts/mini_orchestrator_analyst_flip.py` to capture analyst decision flip JSON.
9. (Optional) Execute `pytest -k nvml_flags` to validate NVML gauge exposure when enabled.
10. Archive artifacts under `orchestrator_demo_results/` into documentation or release notes.
11. (Optional) Adjust `GPU_ORCHESTRATOR_LEASE_TTL` in env (0 disables TTL) and confirm `lease_expired_total` metric after forced expiry (can monkeypatch timestamps in test).
6. Optional: add SSE/WebSocket event streaming prototype (low-frequency state push)

---
Authoritative tracking document for GPU Orchestrator readiness.