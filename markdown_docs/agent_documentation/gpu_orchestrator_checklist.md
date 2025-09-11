## GPU Orchestrator Implementation Checklist (E2E Readiness)

Last Updated: 2025-09-11

### ✅ Completed
- Service created: `agents/gpu_orchestrator/main.py` with endpoints `/health`, `/ready`, `/gpu/info`, `/policy` (GET/POST, SAFE_MODE read-only), `/allocations` (placeholder)
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
- Lease / allocation endpoints (`/lease`, `/release`) with token-based session
- NVML-based metrics (granular utilization, PCIe throughput) when out of SAFE_MODE
- Policy mutation (POST /policy) when SAFE_MODE=false with strict validation
- Event streaming (Server-Sent Events or WebSocket) for dashboard live updates
- Prometheus exporter `/metrics`

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

### 📌 Next Action (Recommended Order)
1. Implement client SDK
2. Wire Analyst to read orchestrator metrics (log receipt)
3. Add simple unit tests (mock subprocess for nvidia-smi)
4. Deploy env + start systemd instance
5. Run mini E2E (5–10 articles) with SAFE_MODE=true
6. Expand to full E2E with Analyst GPU path

---
Authoritative tracking document for GPU Orchestrator readiness.